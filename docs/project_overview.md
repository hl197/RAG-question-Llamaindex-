# RAG 智能问答 Agent — 项目文档

> 基于 LlamaIndex + DeepSeek + ChromaDB 的检索增强生成（RAG）系统，支持多格式文档上传、智能问答、SSE 流式输出、持久化记忆。

---

## 一、技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **LLM** | DeepSeek Chat (API) | 通过 OpenAI SDK 协议调用，支持流式/非流式，自动指数退避重试 |
| **Embedding** | sentence-transformers (384d) | paraphrase-multilingual-MiniLM-L12-v2，多语言语义模型；备选 LocalEmbedding（256d 纯 numpy，零依赖） |
| **向量数据库** | ChromaDB (PersistentClient) | 本地持久化向量存储，cosine 距离，增量索引 |
| **Agent 框架** | LlamaIndex (ReAct Agent) | 自定义 ReActAgentWorker，修复 DeepSeek 流式截断问题 |
| **后端框架** | FastAPI + Uvicorn | 12 个 RESTful API + SSE 流式端点，异步非阻塞 |
| **前端** | 原生 SPA (HTML/CSS/JS) | 三栏布局，暗色主题，SSE 流式逐字渲染，marked.js Markdown 渲染 |
| **持久化** | SQLite | 对话历史持久化 + 自动摘要压缩（4000 token 阈值） |
| **评估** | RAGAS | 4 项指标：faithfulness / answer_relevancy / context_precision / context_recall |
| **文档解析** | pymupdf4llm / python-docx / python-pptx | 支持 PDF / DOCX / PPTX / TXT / MD / CSV |

---

## 二、四层架构

```
┌─────────────────────────────────────────────────────────┐
│                    API 层 (server.py)                     │
│  FastAPI: REST + SSE + 静态文件服务                       │
│  12 个端点: 上传/删除文件、会话管理、流式对话、知识库控制  │
├─────────────────────────────────────────────────────────┤
│                   Agent 层 (rag_agent.py)                 │
│  ReAct Agent (Thought→Action→Observation→Answer)         │
│  工具注册: knowledge_base / knowledge_summary             │
│  两阶段重试: 第一次失败自动重置记忆重试                   │
├─────────────────────────────────────────────────────────┤
│              LLM 层 + 知识层 (knowledge/)                  │
│  ┌──────────┐  ┌──────────────────────────────────────┐  │
│  │DeepSeek  │  │ 检索增强管线 (Phase 3.5):               │  │
│  │  LLM     │  │  查询改写 → 查询分解 → 混合检索 →        │  │
│  │(自定义   │  │  父文档映射 → 重排序 → ChromaDB          │  │
│  │ 封装)    │  │  QueryRewriter → QueryDecomposer →       │  │
│  │          │  │  HybridRetriever → ParentMapping →       │  │
│  │          │  │  Reranker → ChromaDB                     │  │
│  └──────────┘  └──────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              持久化层 (agent/memory.py)                   │
│  SQLite 存储: 会话列表 + 消息历史 + 摘要压缩             │
│  file_registry.json: 文件去重注册表                      │
└─────────────────────────────────────────────────────────┘
```

---

## 三、运行流程

### 启动
```
用户启动 server.py
  → FastAPI 应用初始化
  → lifespan 事件触发 Agent 预热
    → DeepSeekLLM 初始化（OpenAI SDK + 自定义 httpx 客户端）
    → Settings.embed_model 初始化（语义模型或本地模型）
    → ChromaDB PersistentClient 连接 / 创建
    → VectorStoreIndex 加载 / 创建索引
    → PersistentChatMemory SQLite 连接
    → ReAct AgentRunner 创建（注册 knowledge_base/knowledge_summary 工具）
    → LLM 连接健康检测
  → 浏览器自动打开 http://localhost:7860
```

### 用户使用
```
网页操作 → HTTP API → RAGAgent → LLM/ChromaDB → SSE 流式响应 → 前端渲染

上传文件:  浏览器拖拽文件 → POST /api/files/upload (XHR + 进度条)
          → agent.upload_file() → loader.load_document()
          → 父块切分 (2048 token) → 子块切分 (512 token)
          → 子块向量化 (SemanticEmbedding 384d)
          → ChromaDB insert_nodes() (子块存储, metadata含父块原文)
          → file_registry 记录 → 进度条显示"处理完成"

提问:      浏览器输入消息 → POST /api/chat/stream (SSE)
          → agent.chat_stream()
          → PersistentChatMemory 加载/压缩历史
          → ReAct Agent 推理循环
            → 工具调用: knowledge_base 检索
            → DeepSeek LLM 生成回答（流式）
          → SSE 逐 token 推送 → 前端 marked.js 实时渲染
          → 回答保存到 SQLite
```

---

## 四、数据流

### 文件上传数据流
```
PDF/DOCX/TXT
    │
    ▼
loader.load_document(file_path)
    │  依次尝试: PDF (pymupdf4llm) → DOCX → PPTX → TXT/MD/CSV
    │  返回: List[Document]
    ▼
ParentSplitter(chunk_size=2048, overlap=0)
    │  先切父块（大粒度，用于 LLM 上下文）
    ▼
ChildSplitter(chunk_size=512, overlap=128)
    │  每块父块再切子块（小粒度，用于向量检索）
    ▼
SemanticEmbedding.encode(sub_nodes) → 384维向量
    │  sentence-transformers 多语言模型
    │  子块 metadata 中存储 parent_text（父块原文）
    ▼
ChromaDB (knowledge_base collection)
    │  cosine 距离索引，只存子块
    ▼
file_registry.json 记录文件名（去重）
```

### 问答数据流
```
用户输入 "张三是谁"
    │
    ▼
PersistentChatMemory
    │  SQLite 加载历史消息
    │  超 4000 token → 自动摘要压缩
    ▼
ReAct Agent (Thought→Action→Observation→Answer)
    │
    ├─── 判断是否需要检索知识库
    │
    ├── knowledge_base 工具调用 (完整 6 步管线):
    │     │
    │     ├── QueryRewriter.rewrite(query)
    │     │    口语查询 → 关键词优化（LLM 改写）
    │     │
    │     ├── QueryDecomposer.decompose(rewritten)
    │     │    复杂问题 → [子问题1, 子问题2, ...]
    │     │
    │     ├── HybridRetriever.retrieve(sub_query)
    │     │    向量检索 (top-24) + BM25 关键词检索
    │     │    → RRF 融合 → top-8
    │     │
    │     ├── Parent Document Mapping
    │     │    命中子块 → 映射回所属父块 → 去重
    │     │    解决: 小粒度检索精度 + 大粒度上下文完整性
    │     │
    │     ├── Reranker.rerank(query, candidates)
    │     │    关键词评分 (0.3) + LLM 相关性评分 (0.7)
    │     │    → top-5 精排结果
    │     │
    │     └── 返回片段 → Agent 观察
    │
    └── DeepSeek LLM 生成回答
          │  ReAct 格式解码 → 提取 Answer 部分
          │  流式逐 token 输出
          ▼
    SSE 推送 → 前端 EventSource 逐字渲染
          │  marked.js → Markdown 转 HTML
          ▼
    浏览器实时显示 + 保存到 SQLite
```

---

## 五、Agent 相关知识

### 5.1 ReAct Agent 是什么

ReAct（Reasoning + Acting）是一种结合推理和行动的 Agent 范式。模型在每步推理中交替执行：

- **Thought**: 分析当前状态，决定下一步做什么
- **Action**: 调用工具（检索知识库/查统计）
- **Observation**: 工具返回的结果
- **Answer**: 收集足够信息后给出最终答案

本项目的 ReAct 提示模板针对 DeepSeek 优化，要求输出严格以 `Thought:` 开头。

### 5.2 自定义修复: FixedReActAgentWorker

DeepSeek 的流式输出行为与 OpenAI 不同——有时输出不以 `Thought:` 开头。原版 `ReActAgentWorker._infer_stream_chunk_is_final` 会误判首个 chunk 为"最终答案"，提前终止 ReAct 循环。

修复方案：继承 `ReActAgentWorker` 重写该方法，**只有确认 chunk 包含 `Answer:` 时才视为推理结束**，否则继续收集。

### 5.3 工具注册

Agent 注册了两个核心工具：

| 工具名 | 类型 | 功能 | 触发场景 |
|--------|------|------|---------|
| `knowledge_base` | QueryEngineTool | 检索知识库文档片段 | 用户问具体知识性问题 |
| `knowledge_summary` | FunctionTool | 返回知识库统计概览 | 用户问"有什么文件"/"知识库情况" |

### 5.4 检索增强管线 (Phase 3.5)

这是项目中最重要的优化模块，专门解决 RAG 的核心痛点。从 Phase 3 升级到 3.5，
新增了查询改写和父文档检索两个阶段，形成完整的 6 步管线：

```
用户 Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Phase 3.5a: QueryRewriter                            │
│  LLM 将口语化问题改写为检索友好的关键词查询          │
│  "我想知道张三做了什么" → "张三 项目经历 工作职责"   │
│  解决: 向量检索对口语/修饰词多的问题匹配效果差       │
├─────────────────────────────────────────────────────┤
│ Phase 3.5b: QueryDecomposer                          │
│  LLM 判断是否为复杂查询（含多个问号/连接词）         │
│  是 → 拆分为 2-3 个子查询，分别检索后合并结果        │
│  解决: "asyncio 和 GIL 有什么关系" 类跨主题问题       │
├─────────────────────────────────────────────────────┤
│ Phase 3.5c: HybridRetriever                          │
│  向量检索 (semantic) → top-24                        │
│  BM25 关键词检索 → top-24                            │
│  RRF (倒数排名融合) 合并 → top-8                     │
│  解决: 纯语义检索对精确术语/编号/代码匹配不足        │
├─────────────────────────────────────────────────────┤
│ Phase 3.5d: Parent Document Mapping                  │
│  命中子块 (512 token) → 映射回父块 (2048 token)      │
│  去重后返回给 LLM                                    │
│  解决: 小粒度检索精度 + 大粒度上下文完整性           │
├─────────────────────────────────────────────────────┤
│ Phase 3.5e: Reranker                                 │
│  关键词评分 (0.3权重) + LLM 语义评分 (0.7权重)      │
│  始终执行精排（不再因候选少而跳过）                  │
│  解决: 前几阶段召回量大，需要精排提升 top-K 准确率   │
└─────────────────────────────────────────────────────┘
    │
    ▼
最终上下文 → LLM 生成回答
```

### 5.5 记忆系统

两级记忆架构：

| 层级 | 存储 | 作用 | 机制 |
|------|------|------|------|
| **短期** | ChatMemoryBuffer | Agent 工作记忆 | LlamaIndex 内置的对话上下文 |
| **长期** | SQLite (PersistentChatMemory) | 跨会话持久化 | 每次对话保存到 conversations.db |
| **压缩** | LLM 摘要 | 防超长上下文 | 超 4000 token 时自动压缩历史为摘要 |

### 5.6 安全与容错

- **指数退避重试**: API 限速 (429/503) 时自动重试，最长等待 8s
- **Agent 自愈重试**: 第一次流式调用失败 → 重置 Agent 和记忆 → 自动重试
- **API Key 三级加载**: 环境变量 → .env → Fernet+PBKDF2 加密文件
- **SSL 证书**: 使用 `certifi` 自动定位，不依赖系统环境变量
- **代理绕过**: DeepSeek/OpenAI API 域名自动加入 NO_PROXY，不受系统代理干扰

---

## 六、项目亮点（简历要点）

### 🏆 技术亮点

1. **端到端 RAG 系统自研**
   - 从文档解析 → 向量化 → 检索增强 → LLM 生成 → 前端展示，全链路自实现
   - 无需 Gradio/Streamlit 等快速原型框架，采用 FastAPI + 原生 SPA，生产可用

2. **检索增强管线深度优化**（Phase 3 → 3.5）
   - 查询改写（QueryRewriter）：LLM 口语转关键词，提升向量检索命中率
   - 查询分解器解决跨主题复杂问题
   - 混合检索融合语义 + 关键词，弥补纯向量对精确术语匹配的不足
   - **父文档检索**：子块（512 token）检索精度 + 父块（2048 token）上下文完整性
   - LLM 两阶段重排序提升检索精准度
   - 对比纯向量检索，在 RAGAS 评测中显著提升 4 项指标

3. **Custom LLM 封装**
   - 绕过 LlamaIndex OpenAI 模型名校验，自定义 DeepSeek LLM 封装
   - 支持任意 OpenAI 兼容 API（DeepSeek/Qwen/GLM 等）
   - 内置指数退避重试、SSL 证书自动发现、代理绕过

4. **ReAct Agent 流式修复**
   - 发现并修复了 DeepSeek × LlamaIndex 的流式推理截断 bug
   - 确保长程多步推理中 Agent 不会过早终止

5. **SSE 流式架构**
   - FastAPI StreamingResponse + 前端 ReadableStream
   - 逐 token 推送，marked.js 实时 Markdown 渲染
   - 用户体验接近 ChatGPT 的流式效果

6. **双 Embedding 策略**
   - SemanticEmbedding（384维 sentence-transformers）：理解语义，同义词匹配
   - LocalEmbedding（256维 纯numpy）：零依赖，快速启动
   - 通过配置一键切换

7. **API Key 安全体系**
   - 三级加载：环境变量 → .env → Fernet+PBKDF2 加密文件
   - PBKDF2 600K 迭代派密钥，加密文件可提交仓库

### 🛠 工程亮点

- **全类型文档解析**: PDF（含表格/图片）、DOCX、PPTX、TXT、MD、CSV
- **增量索引**: 文件注册表去重，避免重复索引
- **持久化记忆**: SQLite 存储对话历史，4000 token 自动摘要压缩
- **会话管理**: 多会话支持，历史回溯，删除会话
- **RAGAS 质量评估**: faithfulness / relevancy / precision / recall 四项指标
- **响应式前端**: 三栏布局，移动端自适应，暗色主题，拖拽上传

---

## 七、评估体系

使用 RAGAS 框架对系统进行量化评估，在独立 ChromaDB 环境中运行，不污染生产数据。

| 指标 | 说明 | 测量内容 |
|------|------|---------|
| **faithfulness** | 忠实度 | 生成答案是否基于检索到的上下文，有无幻觉 |
| **answer_relevancy** | 答案相关性 | 答案与问题的相关程度 |
| **context_precision** | 上下文精度 | 检索到的文档片段中，有多少是真正相关的 |
| **context_recall** | 上下文召回率 | 所有相关文档片段中，检索到了多少 |

4 篇测试文档 + 10 个问答对，覆盖单文档事实查询、跨文档综合、多跳推理等场景。

```bash
conda activate all-in-rag
python evaluation/run_eval.py          # 运行评估
python evaluation/run_eval.py --keep   # 保留临时数据
```

---

## 八、快速启动

```bash
# 1. 安装依赖
conda activate all-in-rag
python -m pip install -r requirements.txt

# 2. 配置 API Key
export DEEPSEEK_API_KEY="sk-xxxxx"

# 3. 启动服务
python server.py
# 浏览器自动打开 http://localhost:7860

# 4. 可选：下载语义模型（国内网络需使用镜像）
python scripts/download_model.py
```

---

## 九、项目目录结构

```
RAG-question/
├── server.py                    # FastAPI 入口（12 API + SSE + 静态文件）
├── config.py                    # 全局配置 + API Key 三级加载
├── agent/                       # Agent 核心层
│   ├── rag_agent.py             # Agent 中枢：工具/记忆/对话
│   ├── deepseek_llm.py          # DeepSeek LLM 自定义封装
│   ├── llm_adapter.py           # LLM 配置工厂
│   └── memory.py                # SQLite 持久化记忆 + 摘要压缩
├── knowledge/                   # 知识层（检索增强管线 Phase 3.5）
│   ├── loader.py                # 多格式文档解析
│   ├── indexer.py               # ChromaDB 索引管理（父文档检索）
│   ├── semantic_embedding.py    # sentence-transformers 语义 Embedding
│   ├── local_embedding.py       # 纯 NumPy 本地 Embedding（零下载）
│   ├── query_rewriter.py        # LLM 查询改写（口语→关键词）
│   ├── query_decomposer.py      # LLM 查询分解（复杂→子查询）
│   ├── hybrid_retriever.py      # 向量 + BM25 混合检索
│   └── reranker.py              # 关键词 + LLM 两阶段重排序
├── evaluation/                  # RAGAS 评估
│   ├── evaluator.py             # 评估核心
│   ├── test_data.py             # 测试文档 + QA 对
│   └── run_eval.py              # 评估 CLI 入口
├── static/                      # 前端 SPA
│   └── index.html               # 三栏布局，暗色主题，SSE 流式渲染
└── docs/                        # 文档
    └── project_overview.md      # 本文档
```

---

> **关键词**: RAG · LlamaIndex · DeepSeek · ChromaDB · FastAPI · ReAct Agent · SSE 流式 · sentence-transformers · SQLite · RAGAS
