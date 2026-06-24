# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG 智能问答 Agent — 基于 LlamaIndex + DeepSeek + ChromaDB 的检索增强生成系统。FastAPI 后端 + 原生 SPA 前端，用户上传文档后系统自动解析、向量化后存入 ChromaDB，基于 ReAct Agent 实现智能问答（支持 SSE 流式输出）。

## Commands

```bash
# 启动 FastAPI 服务（自动打开 http://localhost:7860）
conda activate all-in-rag
python server.py

# 安装/更新依赖
python -m pip install -r requirements.txt

# RAG 质量评估（RAGAS 4 项指标）
python evaluation/run_eval.py          # 运行评估
python evaluation/run_eval.py --keep   # 评估后保留临时数据

# 加密 API Key（用于生产环境提交）
python scripts/encrypt_key.py
export KEY_PASSPHRASE='你的口令'

# 手动下载语义模型（国内网络走镜像）
python scripts/download_model.py
```

## Architecture

### Layered Architecture (四层架构)

```
FastAPI Server (server.py)        ← API 层 (REST + SSE)
    ├── static/index.html         ← 前端 SPA（原生 HTML/CSS/JS）
    └── static/marked.min.js      ← 本地化 Markdown 渲染（避免 CDN 被拦截）
    ↓
RAG Agent (rag_agent.py)         ← Agent 层 (ReAct)
    ├── knowledge_base (增强检索)
    └── knowledge_summary (知识库摘要)
    ↓               ↓
DeepSeekLLM      Knowledge Layer      ← LLM层 + 知识层
(deepseek_llm.py)  ├── Loader (loader.py)
                   ├── Indexer (indexer.py, KnowledgeIndex 类)
                   ├── Embedding (semantic_embedding.py / local_embedding.py)
                   ├── QueryDecomposer (query_decomposer.py)
                   ├── HybridRetriever (hybrid_retriever.py)
                   └── Reranker (reranker.py)
    ↓
Persistent Memory (memory.py)     ← 持久化层 (SQLite)
    ↓
Evaluation (evaluation/)          ← 评估层 (RAGAS)
```

### Key Files & Responsibilities

| File | Role |
|------|------|
| `server.py` | FastAPI 后端：12 个 REST 端点 + SSE 流式聊天 + 静态文件服务 |
| `static/index.html` | 原生 SPA 前端：三栏布局（知识库\|聊天\|会话），暗色主题，SSE 流式渲染 |
| `static/marked.min.js` | Markdown 渲染库（本地化，规避 CDN 跟踪防护拦截） |
| `config.py` | 全局配置 + API Key 三级加载 + Embedding 类型切换 + 日志系统 |
| `agent/rag_agent.py` | Agent 中枢：工具注册、对话分发、增强检索管线 |
| `agent/deepseek_llm.py` | 自定义 LLM 封装，OpenAI SDK 调用 DeepSeek API |
| `agent/llm_adapter.py` | LLM 配置工厂，统一创建 DeepSeekLLM 实例 |
| `agent/memory.py` | SQLite 持久化记忆 + 自动摘要压缩（4000 token 阈值） |
| `knowledge/loader.py` | 多格式文档解析 PDF/DOCX/PPTX/TXT/MD/CSV |
| `knowledge/indexer.py` | KnowledgeIndex 类封装 ChromaDB 生命周期，向后兼容模块级函数 |
| `knowledge/local_embedding.py` | 纯 NumPy Embedding（字符 n-gram + 随机投影，零下载） |
| `knowledge/semantic_embedding.py` | sentence-transformers 语义 Embedding（384d，中英文，离缓存加载） |
| `knowledge/query_decomposer.py` | LLM 驱动的复杂查询拆分为子查询 |
| `knowledge/hybrid_retriever.py` | 向量 + BM25 混合检索（RRF 融合） |
| `knowledge/reranker.py` | 关键词 + LLM 两阶段重排序 |
| `evaluation/evaluator.py` | RAGAS 评估核心：独立 ChromaDB 索引 → 检索生成 → RAGAS evaluate() |
| `evaluation/test_data.py` | 4 篇测试文档 + 10 个问答对（含 ground truth） |
| `evaluation/run_eval.py` | 评估 CLI 入口（含 HF 离线环境变量注入） |
| `scripts/encrypt_key.py` | API Key Fernet 加密工具 |
| `tests/` | 单元测试：LLM、LLM Adapter、Memory |
| `docs/project_overview.md` | 项目详细介绍（含技术栈、架构图、数据流、亮点，供简历参考） |

### Project Structure

```
RAG-question/
├── server.py                    # FastAPI 入口（12 API + SSE + 静态文件）
├── config.py                    # 全局配置 + API Key 三级加载
├── requirements.txt             # Python 依赖
├── CLAUDE.md                    # 项目文档（本文件）
├── ERROR_LOG.md                 # 错误排查记录
├── README.md                    # 面向开发者的快速入门
│
├── agent/                       # Agent 核心层
│   ├── rag_agent.py             # Agent 中枢：工具/记忆/对话
│   ├── deepseek_llm.py          # DeepSeek LLM 自定义封装
│   ├── llm_adapter.py           # LLM 配置工厂
│   └── memory.py                # SQLite 持久化记忆 + 摘要压缩
│
├── knowledge/                   # 知识层（检索增强管线）
│   ├── loader.py                # 多格式文档解析（PDF/DOCX/PPTX/TXT/MD/CSV）
│   ├── indexer.py               # ChromaDB 索引/检索（KnowledgeIndex 类）
│   ├── local_embedding.py       # 纯 NumPy 本地 Embedding（256d，可配置）
│   ├── semantic_embedding.py    # sentence-transformers 语义 Embedding（384d）
│   ├── query_decomposer.py      # LLM 查询分解
│   ├── hybrid_retriever.py      # 向量 + BM25 混合检索
│   └── reranker.py              # 关键词 + LLM 两阶段重排序
│
├── evaluation/                  # RAGAS 评估
│   ├── evaluator.py             # 评估核心
│   ├── test_data.py             # 4 篇测试文档 + 10 个 QA 对
│   └── run_eval.py              # 评估 CLI 入口
│
├── static/                      # 前端 SPA
│   ├── index.html               # 三栏布局，暗色主题，SSE 流式渲染
│   └── marked.min.js            # Markdown 渲染（本地化）
│
├── tests/                       # 单元测试
│   ├── conftest.py              # 共享 fixture
│   ├── test_deepseek_llm.py     # LLM 封装测试
│   ├── test_llm_adapter.py      # LLM 适配器测试
│   └── test_memory.py           # 记忆模块测试
│
├── docs/                        # 项目文档
│   └── project_overview.md      # 详细介绍（技术栈/架构/数据流/亮点）
│
├── scripts/                     # 工具脚本
│   └── encrypt_key.py           # API Key 加密
│
├── config/                      # 加密配置
│   └── keys.enc                 # Fernet 加密的 API Key
│
└── data/                        # 运行时数据（gitignored）
    ├── chroma/                  # ChromaDB 向量数据
    ├── conversations.db         # SQLite 对话历史
    ├── file_registry.json       # 文件去重注册表
    └── app.log                  # 应用日志
```

### Data Flow

**文件上传**: `POST /api/files/upload → server.upload_file() → RAGAgent.upload_file() → loader.load_document() → SentenceSplitter 切分 → SemanticEmbedding 向量化 → indexer.build_or_update_index() → ChromaDB insert_nodes()`

**用户提问（JSON）**: `POST /api/chat → RAGAgent.chat() → 加载历史记忆 → ReActAgentWorker 推理循环 → (知识库检索/直接回答) → DeepSeek API → 保存回复`

**用户提问（SSE 流式）**: `POST /api/chat/stream → RAGAgent.chat_stream() → text/event-stream → 前端 ReadableStream 逐 token 渲染`

### API Endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | SPA 首页 |
| POST | `/api/chat` | JSON 同步对话 |
| POST | `/api/chat/stream` | SSE 流式对话 |
| GET | `/api/sessions` | 会话列表 |
| POST | `/api/sessions` | 创建新会话 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| GET | `/api/sessions/{id}/history` | 会话历史 |
| POST | `/api/files/upload` | 上传文件（multipart） |
| GET | `/api/files` | 文件列表 |
| DELETE | `/api/files/{name}` | 删除文件 |
| GET | `/api/knowledge/stats` | 知识库统计 |
| POST | `/api/knowledge/clear` | 清空知识库 |
| GET | `/api/usage` | Token 用量统计 |

### Key Design Decisions

- **FastAPI 替代 Gradio**: RESTful API + SSE 流式 + 原生 SPA 前端，零构建依赖，更好的定制性和生产可用性
- **双 Embedding 策略**: `EMBED_TYPE="local"` 用字符 n-gram（零下载）或 `"semantic"` 用 sentence-transformers（384d，RAGAS answer_relevancy +18%）
- **增强检索管线（Phase 3）**: 查询分解 → 混合检索（向量+BM25）→ 重排序（关键词+LLM），可通过 `config.py` 开关控制
- **自定义 LLM 封装**: 绕过 LlamaIndex OpenAI 模型名校验，支持任意 DeepSeek 模型名
- **FixedReActAgentWorker**: 继承 ReActAgentWorker 重写 `_infer_stream_chunk_is_final`，修复 DeepSeek 流式输出误判为"最终答案"提前终止推理的问题
- **两级记忆**: ChatMemoryBuffer（短期工作记忆）+ SQLite（长期持久化）+ 摘要压缩（4000 token 阈值）
- **增量索引**: file_registry.json 去重，仅首次上传时建立索引
- **API Key 安全**: 三级加载（env → .env → Fernet+PBKDF2 加密文件），加密文件可提交仓库
- **RAGAS 评估**: 独立 ChromaDB collection，4 项指标（faithfulness / relevancy / precision / recall）
- **marked.js 本地化**: 静态文件内置 Markdown 渲染库，规避浏览器跟踪防护对 CDN 的拦截

### Constraints & Known Issues

- `indexer.py` 的 `_init_embedding()` 会检查 `Settings.embed_model`，若已由 `rag_agent.py` 设置则不覆盖。新增 embedding 类型时必须保证两处一致。
- `LocalEmbedding` 和 `DeepSeekLLM` 继承 Pydantic BaseModel，内部字段必须用 `object.__setattr__` 赋值
- `LocalEmbedding` 字符级 n-gram 无语义理解能力，跨文档综合题检索效果差。建议生产环境使用 `EMBED_TYPE="semantic"`
- SemanticEmbedding 模型下载到 `~/.cache/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。首次下载需走镜像：
  `curl -sL --max-time 600 -o model.safetensors "https://hf-mirror.com/.../resolve/main/model.safetensors"`（或使用 `python scripts/download_model.py`）
- `indexer.py` 已重构为 `KnowledgeIndex` 类，模块级函数保留作为向后兼容包装
- ChromaDB 数据目录在 `data/chroma/`，已 gitignored；切换 Embedding 类型或维度时必须清空此目录重建索引
- 会话历史存储在 `data/conversations.db`，已 gitignored
- `.env` 文件包含明文 API Key，已 gitignored
- `docx` / `pptx` 文件加载依赖 `python-docx` / `python-pptx` 库，`requirements.txt` 已包含
- 浏览器跟踪防护可能拦截 CDN 资源（如 marked.js），已通过本地化 `static/marked.min.js` 解决

### Environment Variables (from ~/.bashrc)

- `DEEPSEEK_API_KEY` — DeepSeek API 密钥
- `KEY_PASSPHRASE` — 用于解密 `config/keys.enc` 的口令
- ~~`SSL_CERT_FILE`~~ — 已不依赖：`deepseek_llm.py` 使用 `certifi` 自动定位证书
- `HF_ENDPOINT` — HuggingFace 镜像（`https://hf-mirror.com`），首次下载模型时使用

### Configuration (`config.py`)

- `LLM_MODEL` — 模型名，默认 `deepseek-chat`
- `LLM_MAX_TOKENS` — 单次回答最大 token 数，默认 `8192`
- `LLM_CONTEXT_WINDOW` — 上下文窗口大小，默认 `65536`
- `EMBED_TYPE` — `"semantic"`（语义模型 384d）或 `"local"`（纯 numpy 256d）
- `CHUNK_SIZE / CHUNK_OVERLAP` — 文档分块参数（512 / 128）
- `SIMILARITY_TOP_K / SIMILARITY_CUTOFF` — 检索参数（8 / 0.3）
- `ENABLE_QUERY_DECOMPOSITION / HYBRID_RETRIEVAL / RERANKING` — Phase 3 检索增强开关

## Development Habits

### 错误调试必须记录到 ERROR_LOG.md
每次排查 Bug 后，无论是否找到根因，都必须将调试过程和结果追加到 `ERROR_LOG.md`，格式：
```
## N. 错误标题

**错误**: 用户看到的错误信息

**根因**: 分析后确定的根本原因

**解决过程**: 尝试了哪些方案、哪些有效哪些无效

**关键文件**: 涉及修改的文件列表
```
