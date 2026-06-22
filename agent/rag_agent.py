"""
RAG Agent 封装

整合文档加载、向量索引、LLM、持久化记忆、工具调用。
"""

import os
from typing import List, Optional, Dict

from llama_index.core import Settings, Document
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.react import ReActAgentWorker
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from knowledge.local_embedding import LocalEmbedding
from agent.llm_adapter import create_llm

import config
from knowledge import loader, indexer
from agent.memory import PersistentChatMemory


class RAGAgent:
    """
    智能问答 Agent。

    用法:
        agent = RAGAgent()
        agent.upload_file("简历.pdf")
        agent.chat("张三有什么项目经验？", session_id="abc123")
    """

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        # ── LLM（通过适配器创建，自动从环境变量/.env加载配置） ──
        self.llm = create_llm(api_key=api_key, api_base=api_base)
        Settings.llm = self.llm

        # ── Embedding（纯本地，无需模型下载） ──
        embed_model = LocalEmbedding(dim=config.EMBED_DIM)
        Settings.embed_model = embed_model

        # ── 索引 ──
        indexer.init_chroma()
        self._tools = self._build_tools()

        # ── Agent ──
        self._agent = self._create_agent()

        # ── 记忆 ──
        self.memory = PersistentChatMemory()
        self.current_session_id: Optional[str] = None

        # ── 启动时检测 LLM 连接 ──
        self._warmup_llm()

    def _warmup_llm(self):
        """启动时检测 LLM 连接是否正常"""
        try:
            resp = self.llm.complete("回复OK即可")
            print(f"✅ LLM 连接正常: {str(resp)[:40]}")
        except Exception as e:
            print(f"⚠️  LLM 启动检测失败: {type(e).__name__}: {e}")
            print("   首次对话时可能需要等待连接重建。")

    # ── Agent 初始化 ──────────────────────────────

    def _build_tools(self) -> List:
        """创建 Agent 工具列表"""
        tools = []

        # Tool 1: 知识库检索
        query_engine = indexer.get_query_engine()
        rag_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="knowledge_base",
            description=(
                "检索知识库中的文档实际内容，返回最相关的原文片段。\n"
                "当用户询问某份文件的具体信息、查找文档中的某个主题或细节时，"
                "必须使用此工具。输入应为自然语言问题，例如：\n"
                "  - 「张三的项目经验是什么？」\n"
                "  - 「Happy-LLM 这本书讲了什么？」\n"
                "  - 「文档中关于 Python 的内容」\n"
                "注意：这是按语义相似度检索内容，不是按文件名搜索。"
            ),
        )
        tools.append(rag_tool)

        # Tool 2: 知识库摘要/统计
        summary_tool = FunctionTool.from_defaults(
            fn=self._get_knowledge_summary,
            name="knowledge_summary",
            description=(
                "获取知识库的整体概况，包括已上传的文件列表、"
                "文档数量和内容概况。\n"
                "当用户问「知识库里有什么」、「上传了哪些文件」、「有几份文档」"
                "等知识库整体情况时使用。\n"
                "不要用这个工具来查找具体内容——那是 knowledge_base 的职责。"
            ),
        )
        tools.append(summary_tool)

        return tools

    def _get_knowledge_summary(self) -> str:
        """
        获取知识库概况（由 Agent 作为工具调用）。
        """
        stats = indexer.get_collection_stats()
        files = indexer.list_uploaded_files()

        if stats.get("status") == "not_initialized" or stats.get("total_vectors", 0) == 0:
            return "知识库为空，尚未上传任何文件。"

        parts = [
            f"📊 知识库统计",
            f"文件数量: {stats.get('files_count', 0)} 个",
            f"文档片段数: {stats.get('total_vectors', 0)} 个",
            f"",
            f"📁 已上传的文件:",
        ]
        for f in files:
            parts.append(f"  - {f}")

        return "\n".join(parts)

    def _create_agent(self) -> AgentRunner:
        """创建 AgentRunner 实例"""
        agent_worker = ReActAgentWorker.from_tools(
            tools=self._tools,
            llm=self.llm,
            verbose=True,
            max_iterations=15,
        )
        agent = AgentRunner(
            agent_worker=agent_worker,
            memory=ChatMemoryBuffer.from_defaults(
                token_limit=config.MAX_HISTORY_TOKENS * 2,  # 给 agent 内部 buffer 更大的空间
            ),
        )
        return agent

    # ── 对话 ──────────────────────────────────────

    def chat(self, message: str, session_id: Optional[str] = None) -> str:
        """
        对话接口。

        Args:
            message: 用户消息
            session_id: 会话 ID（None 自动创建）

        Returns:
            str: Agent 回复
        """
        # 确保有会话
        if session_id is None:
            session_id = self.memory.create_session()
        self.current_session_id = session_id

        # 确保会话存在
        if not self.memory.session_exists(session_id):
            self.memory.create_session()
            self.memory.rename_session(session_id, "")
            # 修正: create_session 生成新 id，所以用传入的 id
            self._init_session_memory(session_id)

        # 初始化会话记忆（仅首次）
        if self.memory.get_message_count(session_id) == 0:
            self._init_session_memory(session_id)
        else:
            # 检查是否需要压缩
            if self.memory.needs_compression(session_id):
                self._compress_memory(session_id)
            # 加载历史到 agent buffer
            self._load_session_to_agent(session_id)

        # 保存用户消息
        self.memory.save_message(session_id, "user", message)

        # Agent 处理（带自动重试）
        import traceback

        for attempt in range(2):
            try:
                response = self._agent.chat(message)
                reply = str(response)
                break  # 成功，跳出重试循环
            except Exception as e:
                error_detail = traceback.format_exc()
                # 控制台打印完整异常
                print(f"\n{'='*50}")
                print(f"❌ Agent 调用出错 (第{attempt+1}次)")
                print(f"   异常类型: {type(e).__name__}")
                print(f"   异常信息: {e}")
                print(f"   完整 traceback:")
                print(f"{error_detail}")
                print(f"{'='*50}\n")

                if attempt == 0:
                    # 第一次失败，重建 Agent 后重试
                    print("🔄 重建 Agent 并重试...")
                    self._agent = self._create_agent()
                    self._init_session_memory(session_id)
                else:
                    # 第二次仍失败，给用户显示详细错误
                    reply = f"抱歉，处理时出错: {type(e).__name__}: {e}"

        # 保存助手回复
        self.memory.save_message(session_id, "assistant", reply)

        return reply

    def _init_session_memory(self, session_id: str):
        """初始化新会话的记忆"""
        # 插入一条系统消息告知 agent 其角色
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "你是一个智能文档问答助手。你的核心职责是基于知识库中的文档回答用户问题。\n\n"
                "【必须遵守的规则】\n"
                "1. 当用户询问知识库中的具体信息或文档内容时，**必须**使用 knowledge_base 工具\n"
                "   检索知识库，不得依赖自身知识回答。\n"
                "2. 当用户问「知识库里有什么」、「总结一下文档」等全局性问题时，使用\n"
                "   knowledge_summary 工具获取文件列表和统计信息。\n"
                "3. 每次回答前先判断是否需要检索知识库，需要时先用工具再回答。\n"
                "4. 检索后如果知识库中确实没有相关信息，如实告知用户。\n"
                "5. 闲聊或问候可以直接回应，不需要使用工具。"
            ),
        )
        self._agent.memory.put(system_msg)

    def _load_session_to_agent(self, session_id: str):
        """从 SQLite 加载历史到 agent 的 memory buffer"""
        # 清空当前 buffer
        self._agent.reset()

        # 先放 system prompt
        self._init_session_memory(session_id)

        # 加载历史消息
        history = self.memory.load_history(session_id)
        for msg in history:
            self._agent.memory.put(msg)

    def _compress_memory(self, session_id: str):
        """压缩会话历史：将旧消息摘要化"""
        history = self.memory.load_history(session_id)
        if len(history) <= 4:  # 消息太少，不压缩
            return

        # 取前面的消息（保留最近 4 条）
        to_compress = history[:-4]
        recent = history[-4:]

        # 用 LLM 生成摘要
        text_to_summarize = "\n".join(
            f"{'用户' if m.role == MessageRole.USER else '助手'}: {m.content}"
            for m in to_compress
        )

        summary_prompt = (
            f"请对以下对话内容进行简洁的中文摘要，保留关键信息（用户问题、涉及的主题、重要结论）：\n\n"
            f"{text_to_summarize}\n\n摘要："
        )

        try:
            response = self.llm.complete(summary_prompt)
            summary = str(response).strip()

            # 保存摘要
            self.memory.save_summary(
                session_id=session_id,
                summary_text=summary,
                message_count=len(to_compress),
                token_count=self.memory.estimate_tokens(session_id),
            )

            # 重建 SQLite 中的历史为摘要 + 最近消息
            conn = self.memory._get_conn()
            try:
                # 删除旧消息
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
                # 插入摘要消息
                conn.execute(
                    "INSERT INTO conversations (session_id, role, content, session_name) "
                    "VALUES (?, 'system', ?, '')",
                    (session_id, f"[历史摘要] {summary}"),
                )
                # 重新插入最近消息
                for msg in recent:
                    role = "user" if msg.role == MessageRole.USER else "assistant"
                    conn.execute(
                        "INSERT INTO conversations (session_id, role, content) "
                        "VALUES (?, ?, ?)",
                        (session_id, role, msg.content),
                    )
                conn.commit()
            finally:
                conn.close()

        except Exception as e:
            print(f"记忆压缩失败: {e}")

    # ── 会话管理 ──────────────────────────────────

    def new_session(self) -> str:
        """开启新会话"""
        self._agent.reset()
        session_id = self.memory.create_session()
        self.current_session_id = session_id
        self._init_session_memory(session_id)
        return session_id

    def switch_session(self, session_id: str) -> bool:
        """
        切换到指定会话。

        Returns:
            bool: 切换是否成功
        """
        if not self.memory.session_exists(session_id):
            return False

        # 检查是否需要压缩
        if self.memory.needs_compression(session_id):
            self._compress_memory(session_id)

        self.current_session_id = session_id
        self._load_session_to_agent(session_id)
        return True

    def get_sessions(self) -> List[Dict]:
        """获取会话列表"""
        return self.memory.get_session_list()

    def delete_session(self, session_id: str):
        """删除会话"""
        self.memory.delete_session(session_id)
        if self.current_session_id == session_id:
            self.current_session_id = None

    # ── 知识库管理 ──────────────────────────────

    def upload_file(self, file_path: str) -> Dict:
        """
        上传文件并加入知识库。

        Args:
            file_path: 文件路径

        Returns:
            Dict: {success, filename, nodes_added, error?}
        """
        filename = os.path.basename(file_path)

        try:
            # 解析文档
            documents = loader.load_document(file_path)
            if not documents:
                return {
                    "success": False,
                    "filename": filename,
                    "nodes_added": 0,
                    "error": "文档解析结果为空",
                }

            # 构建索引
            nodes_added = indexer.build_or_update_index(documents, filename)

            return {
                "success": True,
                "filename": filename,
                "nodes_added": nodes_added,
            }
        except ValueError as e:
            return {"success": False, "filename": filename, "error": str(e)}
        except Exception as e:
            return {"success": False, "filename": filename, "error": f"处理失败: {str(e)}"}

    def get_uploaded_files(self) -> List[str]:
        """获取已上传文件列表"""
        return indexer.list_uploaded_files()

    def get_knowledge_stats(self) -> Dict:
        """获取知识库统计"""
        return indexer.get_collection_stats()

    def clear_knowledge(self):
        """清空知识库（谨慎使用）"""
        indexer.clear_knowledge_base()
        # 重新初始化
        indexer.init_chroma()
        self._tools = self._build_tools()
        self._agent = self._create_agent()
        if self.current_session_id:
            self._init_session_memory(self.current_session_id)

    # ── 资源管理 ──────────────────────────────────

    def get_current_session_id(self) -> Optional[str]:
        """返回当前会话 ID"""
        return self.current_session_id
