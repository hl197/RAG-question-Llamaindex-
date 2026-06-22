# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG 智能问答 Agent — 基于 LlamaIndex + DeepSeek + ChromaDB 的检索增强生成系统。用户通过 Gradio Web 界面上传文档，系统自动解析、向量化后存入 ChromaDB，基于 ReAct Agent 实现智能问答。

## Commands

```bash
# 启动 Web 界面（访问 http://localhost:7860）
conda activate all-in-rag
python app.py

# 安装/更新依赖
python -m pip install -r requirements.txt

# 加密 API Key（用于生产环境提交）
python scripts/encrypt_key.py                    # 交互式输入
python scripts/encrypt_key.py sk-xxx             # 直接传 Key
export KEY_PASSPHRASE='你的口令'                 # 运行时解密

# 生成项目介绍文档
python scripts/generate_report.py

# 注意：Windows Git Bash 中需要使用完整 Python 路径
# C:\Users\86182\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe app.py
# 推荐使用 conda: conda activate all-in-rag
```

## Architecture

### Layered Architecture (四层架构)

```
Gradio Web UI (app.py)           ← 前端层
    ↓
RAG Agent (rag_agent.py)         ← Agent 层 (ReAct)
    ├── knowledge_base (RAG 检索)
    └── knowledge_summary (知识库摘要)
    ↓               ↓
DeepSeekLLM      Knowledge Layer      ← LLM层 + 知识层
(deepseek_llm.py)  ├── Loader (loader.py)
                   ├── Indexer (indexer.py)
                   └── LocalEmbedding (local_embedding.py)
    ↓
Persistent Memory (memory.py)     ← 持久化层 (SQLite)
```

### Key Files & Responsibilities

| File | Role |
|------|------|
| `app.py` | Gradio 6.x 前端，双栏布局（左侧知识库+会话管理，右侧聊天） |
| `config.py` | 全局配置 + API Key 三级加载策略（环境变量 → .env → keys.enc 解密） |
| `agent/rag_agent.py` | Agent 中枢：工具注册、对话分发、记忆管理 |
| `agent/deepseek_llm.py` | 自定义 LLM 封装，OpenAI SDK 调用 DeepSeek API，绕开 LlamaIndex 模型名校验 |
| `agent/memory.py` | SQLite 持久化记忆 + 自动摘要压缩（4000 token 阈值） |
| `knowledge/loader.py` | 多格式文档解析 PDF/DOCX/PPTX/TXT/MD/CSV |
| `knowledge/indexer.py` | ChromaDB 增量索引 + 相似度检索 |
| `knowledge/local_embedding.py` | 纯 NumPy Embedding（随机投影 + 4-gram 特征，零下载） |

### Data Flow

**文件上传**: `Gradio UploadButton → upload_file() → loader.load_document() → indexer.build_or_update_index() → ChromaDB`

**用户提问**: `输入框 → RAGAgent.chat() → 加载历史记忆 → ReActAgentWorker 推理循环 → (知识库检索/直接回答) → DeepSeek API → 保存回复`

### Key Design Decisions

- **自定义 LLM 封装**: 绕过 LlamaIndex OpenAI 模型名校验，支持任意 DeepSeek 模型名
- **本地 Embedding**: 随机投影 + 字符 n-gram 特征，零外部依赖（仅 NumPy），解决中国网络 HuggingFace 不可用问题
- **两级记忆**: ChatMemoryBuffer（短期工作记忆）+ SQLite（长期持久化）+ 摘要压缩（4000 token 阈值）
- **增量索引**: file_registry.json 去重，仅首次上传时建立索引
- **API Key 安全**: 三级加载（env → .env → Fernet+PBKDF2 加密文件），加密文件可提交仓库

### Constraints & Known Issues

- `LocalEmbedding` 和 `DeepSeekLLM` 继承 Pydantic BaseModel，内部字段必须用 `object.__setattr__` 赋值
- Gradio 6.x 移除了 `bubble_full_width`、`theme`、`css`（移到 `launch()`）等参数
- `QueryEngineTool.from_defaults()` 和 `FunctionTool.from_defaults()` 不接受 `metadata=` 参数，需分开传 `name=` 和 `description=`
- ChromaDB 数据目录在 `data/chroma/`，已 gitignored
- 会话历史存储在 `data/conversations.db`，已 gitignored
- `.env` 文件包含明文 API Key，已 gitignored

### Environment Variables (from ~/.bashrc)

- `DEEPSEEK_API_KEY` — DeepSeek API 密钥
- `KEY_PASSPHRASE` — 用于解密 `config/keys.enc` 的口令

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
