"""
RAG Agent 封装

整合文档加载、向量索引、LLM、持久化记忆、工具调用。
"""

import os
import re
from typing import List, Optional, Dict

from llama_index.core import Settings, Document
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.react import ReActAgentWorker
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from knowledge.local_embedding import LocalEmbedding
from knowledge.query_decomposer import QueryDecomposer
from knowledge.hybrid_retriever import HybridRetriever
from knowledge.reranker import Reranker
from agent.llm_adapter import create_llm

import config
from knowledge import loader, indexer
from agent.memory import PersistentChatMemory


# 针对 DeepSeek 优化的 ReAct 系统提示模板
_DEEPSEEK_REACT_SYSTEM_HEADER = """\
You are a helpful AI assistant designed to answer questions using the tools available to you.

## Tools

You have access to the following tools:
{tool_desc}

## Output Format — STRICTLY REQUIRED

You MUST follow the format below. THIS IS MANDATORY.

### When you need to use a tool — start with "Thought:":

Thought: <your reasoning in the user's language>
Action: <tool name — one of {tool_names}>
Action Input: <JSON kwargs, e.g. {{"input": "query text"}}>

### When you can answer directly — start with "Thought:":

Thought: <your reasoning in the user's language>
Answer: <your final answer>

### CRITICAL RULES (violation will break the system):
1. Your response MUST start with "Thought:" as the FIRST word.
2. After you get an "Observation:" back from a tool, continue with another "Thought:" step.
3. When you have enough information, end with:
   Thought: ...
   Answer: <final answer>

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""


# 后置过滤：从 Agent 回复中剥离 ReAct 追踪文本，只保留最终答案
def _strip_react_trace(text: str) -> str:
    # 从 Agent 原始输出中提取最终答案，去掉 ReAct 中间步骤
    # 1) 如果包含行首 "Answer:"，取最后一个 "Answer:" 之后的内容（应对多步推理）
    if "\nAnswer:" in text:
        return text.rsplit("\nAnswer:", 1)[-1].strip()
    if text.startswith("Answer:"):
        return text.split("Answer:", 1)[-1].strip()
    # 2) 如果包含 "Action:"，说明工具调用步骤泄露了，尝试取 "Action:" 之前的内容
    if "Action:" in text:
        parts = text.split("Action:", 1)
        before_action = parts[0].strip()
        # 如果 Action 前面是 Thought，去掉 Thought 前缀
        before_action = re.sub(r'^Thought:\s*', '', before_action).strip()
        return before_action if before_action else text
    # 3) 去掉孤立 "Thought:" 前缀
    text = re.sub(r'^Thought:\s*', '', text).strip()
    return text


class FixedReActAgentWorker(ReActAgentWorker):
    """修正版 — DeepSeek 输出不总是以 "Thought:" 开头，原版 _infer_stream_chunk_is_final
    会误判为首个 chunk 就是最终答案，导致 ReAct 循环被提前截断。

    修复：只有确认包含 "Answer:" 时才视为最终 chunk，否则继续收集。
    """

    def _infer_stream_chunk_is_final(
        self, chunk, missed_chunks_storage: list
    ) -> bool:
        full_text = (chunk.message.content or "").strip()
        if not full_text:
            return False
        if len(full_text) < 7:
            missed_chunks_storage.append(chunk)
            return False
        if "Answer:" in full_text:
            missed_chunks_storage.clear()
            return True
        return False


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

        # ── Embedding（根据配置选择类型） ──
        if config.EMBED_TYPE == "semantic":
            from knowledge.semantic_embedding import SemanticEmbedding
            embed_model = SemanticEmbedding()
        else:
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

        # ── Token 用量跟踪 ──
        self._session_token_usage: Dict[str, Dict[str, int]] = {}

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

        # Tool 1: 知识库检索（支持查询分解 + 混合检索 + 重排序）
        if config.ENABLE_QUERY_DECOMPOSITION or config.ENABLE_HYBRID_RETRIEVAL or config.ENABLE_RERANKING:
            rag_tool = FunctionTool.from_defaults(
                fn=self._enhanced_retrieve,
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
        else:
            query_engine = indexer.get_query_engine(similarity_score_cutoff=config.SIMILARITY_CUTOFF)
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

    def _enhanced_retrieve(self, query: str) -> str:
        """
        增强版检索（Agent 工具调用入口）。

        管线: 查询分解 → 混合检索 → 重排序 → 格式化返回
        """
        # Step 1: 查询分解
        if config.ENABLE_QUERY_DECOMPOSITION:
            decomposer = QueryDecomposer(self.llm)
            sub_queries = decomposer.decompose(query)
        else:
            sub_queries = [query]

        # Step 2: 对每个子查询执行检索
        all_results = {}  # text → max_score
        node_texts = indexer.get_node_texts()

        for sq in sub_queries:
            if config.ENABLE_HYBRID_RETRIEVAL and node_texts:
                # 混合检索：向量 + BM25
                def vector_query(q, k):
                    qe = indexer.get_query_engine(similarity_top_k=k)
                    resp = qe.query(q)
                    return [
                        (node.text, node.score or 0)
                        for node in resp.source_nodes
                    ]

                hybrid = HybridRetriever(node_texts, vector_query)
                results = hybrid.retrieve(sq, top_k=config.SIMILARITY_TOP_K)
                for r in results:
                    all_results[r] = max(all_results.get(r, 0), 0.5)
            else:
                # 标准向量检索
                qe = indexer.get_query_engine(
                    similarity_top_k=config.SIMILARITY_TOP_K,
                    similarity_score_cutoff=config.SIMILARITY_CUTOFF,
                )
                resp = qe.query(sq)
                for node in resp.source_nodes:
                    text = node.text
                    all_results[text] = max(all_results.get(text, 0), node.score or 0)

        # Step 3: 重排序
        if config.ENABLE_RERANKING and len(all_results) > config.SIMILARITY_TOP_K:
            candidates = sorted(all_results, key=all_results.get, reverse=True)
            reranker = Reranker(self.llm)
            reranked = reranker.rerank(query, candidates, top_k=config.SIMILARITY_TOP_K)
        else:
            reranked = sorted(all_results, key=all_results.get, reverse=True)[:config.SIMILARITY_TOP_K]

        # Step 4: 格式化返回
        if not reranked:
            return "知识库中未找到相关信息。"

        parts = []
        for i, text in enumerate(reranked, 1):
            parts.append(f"[片段 {i}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def _create_agent(self) -> AgentRunner:
        """创建 AgentRunner 实例"""
        custom_formatter = ReActChatFormatter(
            system_header=_DEEPSEEK_REACT_SYSTEM_HEADER,
        )
        agent_worker = FixedReActAgentWorker.from_tools(
            tools=self._tools,
            llm=self.llm,
            verbose=True,
            max_iterations=15,
            react_chat_formatter=custom_formatter,
        )
        agent = AgentRunner(
            agent_worker=agent_worker,
            memory=ChatMemoryBuffer.from_defaults(
                token_limit=config.MAX_HISTORY_TOKENS * 2,
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

        # 记录 Agent 处理前的 token 用量
        usage_before = self.llm.get_token_usage()

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
                    # 不清除工具，只重置 memory 和 buffer
                    print("🔄 重置记忆并重试...")
                    self._agent.reset()
                    self._init_session_memory(session_id)
                else:
                    # 第二次仍失败，给用户显示详细错误
                    reply = f"抱歉，处理时出错: {type(e).__name__}: {e}"

        # 跟踪 token 用量
        usage_after = self.llm.get_token_usage()
        self._track_session_usage(session_id, usage_before, usage_after)

        # 保存助手回复（过滤掉可能泄露的 ReAct 追踪文本）
        reply = _strip_react_trace(reply)
        self.memory.save_message(session_id, "assistant", reply)

        return reply

    def chat_stream(self, message: str, session_id: Optional[str] = None):
        """
        流式对话生成器。逐块 yield 累积回复文本，前端可实时渲染。

        应对两种场景：
        1) LLM 输出了 "Answer:" 标记 → response_gen 自然流式，逐 chunk yield
        2) 未输出 "Answer:"（伪流式） → 收集后手动拆分为渐进块再 yield
        """
        # 会话管理（同 chat()）
        if session_id is None:
            session_id = self.memory.create_session()
        self.current_session_id = session_id

        if not self.memory.session_exists(session_id):
            self.memory.create_session()
            self.memory.rename_session(session_id, "")
            self._init_session_memory(session_id)

        if self.memory.get_message_count(session_id) == 0:
            self._init_session_memory(session_id)
        else:
            if self.memory.needs_compression(session_id):
                self._compress_memory(session_id)
            self._load_session_to_agent(session_id)

        # 保存用户消息
        self.memory.save_message(session_id, "user", message)

        # 记录 Agent 处理前的 token 用量
        usage_before = self.llm.get_token_usage()

        # Agent 流式处理（带自动重试）
        import traceback

        for attempt in range(2):
            try:
                stream_response = self._agent.stream_chat(message)
                gen = stream_response.response_gen

                # ── peek 前 2 个 chunk，判断是真流式还是伪流式 ──
                first: str = ""
                second: str = ""
                try:
                    d = next(gen)
                    if d:
                        first = d
                except StopIteration:
                    pass

                try:
                    d = next(gen)
                    if d:
                        second = d
                except StopIteration:
                    pass

                # 拼起来
                full_response = first + second

                # 收集到的实际 chunk 数
                real_chunks = (1 if first else 0) + (1 if second else 0)

                if real_chunks <= 2 and full_response and len(full_response) > 30:
                    # ── 伪流式: response_gen 只吐了 1-2 个 chunk（一次全量）──
                    # 手动拆分为渐进块并逐步 yield
                    step = max(3, len(full_response) // 40)
                    text_so_far = ""
                    for i in range(0, len(full_response), step):
                        text_so_far = full_response[:i + step]
                        yield text_so_far
                    full_response = text_so_far
                else:
                    # ── 真流式: 继续消费剩余 chunk，逐步 yield ──
                    # 先把已取到的 yield 出去（如果只有 first，那 second 是空的）
                    if first:
                        yield first
                    if second:
                        full_response = second
                        yield second
                    # 继续消费剩余的 gen
                    for delta in gen:
                        if delta:
                            full_response += delta
                            yield full_response

                break  # 成功，跳出重试循环

            except Exception as e:
                error_detail = traceback.format_exc()
                print(f"\n{'='*50}")
                print(f"❌ Agent 流式调用出错 (第{attempt+1}次)")
                print(f"   异常类型: {type(e).__name__}")
                print(f"   异常信息: {e}")
                print(f"{'='*50}\n")

                if attempt == 0:
                    print("🔄 重置记忆并重试...")
                    self._agent.reset()
                    self._init_session_memory(session_id)
                else:
                    yield f"抱歉，处理时出错: {type(e).__name__}: {e}"
                    full_response = ""

        # 跟踪 token 用量
        usage_after = self.llm.get_token_usage()
        self._track_session_usage(session_id, usage_before, usage_after)

        # 保存助手回复（后置清理 ReAct 追踪文本）
        cleaned = _strip_react_trace(full_response)
        self.memory.save_message(session_id, "assistant", cleaned or full_response)

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
        """压缩会话历史：将旧消息摘要化（委托给 memory.py 的 compress 方法）"""
        self.memory.compress(session_id, self.llm)

    # ── Token 用量跟踪 ──────────────────────────────

    def _track_session_usage(self, session_id: str, before: dict, after: dict):
        """对比 before/after 的累积 token 数，差值归集到当前会话"""
        if session_id not in self._session_token_usage:
            self._session_token_usage[session_id] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._session_token_usage[session_id]["prompt_tokens"] += after["prompt_tokens"] - before["prompt_tokens"]
        self._session_token_usage[session_id]["completion_tokens"] += after["completion_tokens"] - before["completion_tokens"]

    def get_session_token_usage(self, session_id: str) -> Dict[str, int]:
        """获取指定会话的 token 消耗统计"""
        return self._session_token_usage.get(session_id, {"prompt_tokens": 0, "completion_tokens": 0})

    def get_total_token_usage(self) -> Dict[str, int]:
        """获取所有会话的累积 token 消耗"""
        total_prompt = sum(s["prompt_tokens"] for s in self._session_token_usage.values())
        total_completion = sum(s["completion_tokens"] for s in self._session_token_usage.values())
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }

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

    def get_session_history(self, session_id: str) -> List[Dict]:
        """获取指定会话的历史消息列表，用于前端显示"""
        messages = self.memory.load_history(session_id)
        history = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "assistant"
            history.append({"role": role, "content": msg.content})
        return history

    def get_sessions(self) -> List[Dict]:
        """获取会话列表"""
        return self.memory.get_session_list()

    def delete_session(self, session_id: str):
        """删除会话"""
        self.memory.delete_session(session_id)
        self._session_token_usage.pop(session_id, None)
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

    def delete_file(self, filename: str) -> bool:
        """从知识库中删除指定文件"""
        return indexer.remove_file(filename)

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
