"""
RAG Agent 封装

整合文档加载、向量索引、LLM、持久化记忆、工具调用。
"""

import os
import re
import threading
from typing import List, Optional, Dict

from llama_index.core import Settings, Document
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.react import ReActAgentWorker
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from knowledge.local_embedding import LocalEmbedding
from knowledge.query_decomposer import QueryDecomposer
from knowledge.query_rewriter import QueryRewriter
from knowledge.hybrid_retriever import HybridRetriever
from knowledge.reranker import Reranker
from agent.llm_adapter import create_llm

import config
from knowledge import loader, indexer
from agent.memory import PersistentChatMemory


# 针对 DeepSeek 优化的 ReAct 系统提示模板（中文版，更严格）
_DEEPSEEK_REACT_SYSTEM_HEADER = """\
You are a helpful AI assistant designed to answer questions using the tools available to you.

## Tools

You have access to the following tools:
{tool_desc}

## Output Format — STRICTLY REQUIRED

You MUST output in EXACTLY one of the two formats below. NO OTHER FORMAT IS ACCEPTABLE.

### Format 1 — When you need to search the knowledge base:

Thought: <your reasoning in Chinese>
Action: {tool_names}
Action Input: <JSON with "input" or "query" key>

✅ CORRECT example (must use tool name exactly as given):
Thought: 用户问的是张三的项目经验，我需要检索知识库
Action: knowledge_base
Action Input: {{"query": "张三 项目经验"}}

### Format 2 — When you can answer directly:

Thought: <your reasoning in Chinese>
Answer: <your final answer in Chinese>

✅ CORRECT example:
Thought: 用户只是问好，不需要检索
Answer: 你好！有什么可以帮助你的？

### CRITICAL RULES (violation WILL break the system):
1. Your response MUST start with "Thought:" as the FIRST word — nothing before it.
2. After "Action:" put ONLY the tool name — no extra words, no colons, no quotes.
3. After "Action Input:" put ONLY valid JSON.
4. After getting "Observation:" back, continue with another "Thought:" step.
5. When you have enough information, end with "Thought:" + "Answer:".
6. NEVER explain the format — just output it.
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


# 生命周期步骤提取：将 LlamaIndex 的推理步骤转为前端状态文本
def _extract_lifecycle_step(step, user_message: str = "") -> dict | None:
    """从 Agent 推理步骤中提取生命周期信息，返回简化的 status 文本。

    返回格式：{"type": "action|observation|response", "status": "状态描述"}
    或者 None（应跳过此步骤，例如用户自己的输入）。
    """
    if isinstance(step, ActionReasoningStep):
        # 截断过长的思考内容
        thought = step.thought
        if len(thought) > 120:
            thought = thought[:120] + "…"
        return {
            "type": "action",
            "status": f"🤔 {thought} → 🔧 {step.action}",
        }
    if isinstance(step, ObservationReasoningStep):
        # 跳过用户自己消息的回显（由 add_user_step_to_reasoning 注入）
        if step.observation == user_message:
            return None
        # 提取工具返回的简要信息（第一行或文件匹配数等）
        obs = step.observation[:200].replace("\n", " ").strip()
        if len(step.observation) > 200:
            obs += "…"
        return {
            "type": "observation",
            "status": f"📥 {obs}",
        }
    if isinstance(step, ResponseReasoningStep):
        return {
            "type": "response",
            "status": "💡 正在生成回答…",
        }
    return None


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

        # ── 并发锁（保护所有修改共享状态的方法） ──
        self._lock = threading.Lock()

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
        增强版检索管线（Agent 工具调用入口）。

        完整管线（按执行顺序）：
        1. 查询改写     → LLM 将口语化问题转为检索友好语句  [ENABLE_QUERY_REWRITING]
        2. 查询分解     → 复杂问题拆为 2-3 个子查询          [ENABLE_QUERY_DECOMPOSITION]
        3. 混合检索     → 每个子查询分别执行向量+BM25检索     [ENABLE_HYBRID_RETRIEVAL]
        4. 父文档映射   → 命中的子块 → 映射回所属父块去重   [ENABLE_PARENT_RETRIEVAL]
        5. 重排序       → 关键词+LLM 两阶段精排              [ENABLE_RERANKING]
        6. 格式化返回   → 返回 top-K 片段给 Agent
        """
        # ── Step 1: 查询改写 ──
        if config.ENABLE_QUERY_REWRITING:
            rewriter = QueryRewriter(self.llm)
            rewritten_query = rewriter.rewrite(query)
            if rewritten_query != query:
                print(f"  🔄 查询改写: 「{query}」→「{rewritten_query}」")
            search_query = rewritten_query
        else:
            search_query = query

        # ── Step 2: 查询分解 ──
        if config.ENABLE_QUERY_DECOMPOSITION:
            decomposer = QueryDecomposer(self.llm)
            sub_queries = decomposer.decompose(search_query)
        else:
            sub_queries = [search_query]

        # ── Step 3: 对每个子查询执行检索 ──
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

        # ── Step 4: 父文档映射（子块→父块去重） ──
        if config.ENABLE_PARENT_RETRIEVAL:
            parent_mapping = indexer.get_node_parent_mapping()
            if parent_mapping:
                parent_agg = {}
                for child_text, score in all_results.items():
                    parent_text = parent_mapping.get(child_text, child_text)
                    parent_agg[parent_text] = max(parent_agg.get(parent_text, 0), score)
                all_results = parent_agg

        # ── Step 5: 重排序（去掉条件限制，始终执行排序） ──
        if config.ENABLE_RERANKING and len(all_results) > 1:
            candidates = sorted(all_results, key=all_results.get, reverse=True)
            reranker = Reranker(self.llm)
            reranked = reranker.rerank(query, candidates, top_k=config.SIMILARITY_TOP_K)
        else:
            reranked = sorted(all_results, key=all_results.get, reverse=True)[:config.SIMILARITY_TOP_K]

        # ── Step 6: 格式化返回 ──
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
        with self._lock:
            return self._chat_impl(message, session_id)

    def _chat_impl(self, message: str, session_id: Optional[str] = None) -> str:
        """chat() 的实际实现（被锁包裹）"""
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
                    # 重置并重新加载历史上下文（避免重试后"失忆"）
                    print("🔄 重置记忆并重试...")
                    self._load_session_to_agent(session_id)
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
        流式对话生成器（升级版）。

        流程：
          1. 会话管理与记忆加载（同前）
          2. 手动驱动 Agent 逐步推理，每次 yield ("lifecycle", step_data) 展示生命周期
          3. 推理完成后以模拟流式 yield ("token", 累积文本)
          4. 最后 yield ("done", session_id)

        Yields:
            ("lifecycle", dict) — Agent 推理步骤（思考/工具/观察/回答）
            ("token", str)     — 最终回答增量文本
            ("done", session_id) — 结束信号
        """
        # ── 会话管理（同 chat()） ──
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

        import traceback

        full_text = ""
        for attempt in range(2):
            try:
                # 手动逐步骤驱动 Agent，产出生命周期事件 + token + done
                for evt_type, evt_data in self._run_steps(message):
                    if evt_type == "token":
                        full_text = evt_data
                    yield (evt_type, evt_data)
                break  # 成功

            except Exception as e:
                error_detail = traceback.format_exc()
                print(f"\n{'='*50}")
                print(f"❌ Agent 流式调用出错 (第{attempt+1}次)")
                print(f"   异常类型: {type(e).__name__}")
                print(f"   异常信息: {e}")
                print(f"{'='*50}\n")

                if attempt == 0:
                    print("🔄 重置记忆并重试...")
                    self._load_session_to_agent(session_id)
                else:
                    yield ("token", f"抱歉，处理时出错: {type(e).__name__}: {e}")
                    yield ("done", self.current_session_id or session_id)
                    full_text = ""
                    break

        # 跟踪 token 用量
        usage_after = self.llm.get_token_usage()
        self._track_session_usage(session_id, usage_before, usage_after)

        # 保存助手回复（后置清理 ReAct 追踪文本）
        cleaned = _strip_react_trace(full_text)
        self.memory.save_message(session_id, "assistant", cleaned or full_text)

    # ── 手动步骤执行（供 chat_stream 内部使用） ────────────

    def _run_steps(self, message: str):
        """以手动逐步骤方式驱动 Agent，产出 (event_type, data) 事件。

        Yields:
            ("lifecycle", dict) — 推理步骤
            ("token", str)      — 最终回答累积文本
            ("done", session_id) — 结束
        """
        task = self._agent.create_task(message)
        is_last = False
        emitted_count = 0  # 已发出的 reasoning steps 数量

        while not is_last:
            step_output = self._agent.run_step(task.task_id)
            is_last = step_output.is_last

            # 获取当前所有推理步骤
            current_task = self._agent.state.get_task(task.task_id)
            reasoning = current_task.extra_state["current_reasoning"]

            # 发出新增的步骤
            for i in range(emitted_count, len(reasoning)):
                rs = reasoning[i]
                info = _extract_lifecycle_step(rs, message)
                if info is not None:
                    yield ("lifecycle", info)
            emitted_count = len(reasoning)

        # 获取最终回答
        response = self._agent.finalize_response(task.task_id)
        full_text = str(response)

        # 模拟流式输出最终答案
        if full_text:
            chunk_size = max(3, len(full_text) // 40)
            for i in range(0, len(full_text), chunk_size):
                yield ("token", full_text[:i + chunk_size])

        yield ("done", self.current_session_id)

    def _init_session_memory(self, session_id: str):
        """初始化新会话的记忆"""
        # 插入一条系统消息告知 agent 其角色
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "你是一个**严格基于知识库**的文档问答助手。\n\n"
                "【最高优先级规则——必须遵守】\n"
                "1. 🔍 **所有知识类问题都必须先检索知识库**\n"
                "   无论用户问什么（概念解释、事实查询、总结归纳），\n"
                "   只要不是打招呼/闲聊，都必须先用 knowledge_base 工具检索知识库。\n"
                "   即使你觉得自己知道答案，也必须先查知识库确认。\n\n"
                "2. 当用户问「知识库里有什么」、「总结一下文档」等全局性问题时，\n"
                "   使用 knowledge_summary 工具获取文件列表和统计信息。\n\n"
                "3. 检索后如果知识库中确实没有相关信息，如实告知用户，\n"
                "   不要用你自己的知识编造答案。\n\n"
                "4. 打招呼/闲聊可以直接回应，不需要使用工具。\n"
                "   （如「你好」、「再见」、「谢谢」）"
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
        with self._lock:
            return self._new_session_impl()

    def _new_session_impl(self) -> str:
        """new_session 实际实现（被锁包裹）"""
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
        with self._lock:
            return self._switch_session_impl(session_id)

    def _switch_session_impl(self, session_id: str) -> bool:
        """switch_session 实际实现（被锁包裹）"""
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
        with self._lock:
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
