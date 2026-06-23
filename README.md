# RAG 智能问答助手

基于 **LlamaIndex + DeepSeek + ChromaDB** 的检索增强生成（RAG）系统。上传文档 → 自动向量化 → AI 智能问答，支持 SSE 流式实时回复。

## 快速开始

```bash
# 1. 环境准备
conda activate all-in-rag
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"    # 方式 A: 环境变量
# 或编辑 .env 文件写入 DEEPSEEK_API_KEY=xxx        # 方式 B: .env 文件

# 3. 启动服务
python server.py
# 浏览器访问 http://localhost:7860
```

## 功能

| 功能 | 说明 |
|------|------|
| 📄 多格式文档上传 | PDF / DOCX / PPTX / TXT / MD / CSV |
| 🔍 智能检索 | 查询分解 → 混合检索（向量+BM25）→ 重排序 |
| 💬 流式对话 | SSE 实时逐字渲染，支持 Markdown |
| 🧠 持久记忆 | SQLite 跨会话记忆 + 自动摘要压缩 |
| 📊 质量评估 | RAGAS 4 项指标（忠实度/相关性/精准度/召回率） |
| 🎨 现代 UI | 暗色主题三栏布局，响应式设计 |

## API 概览

```
POST /api/chat              JSON 同步对话
POST /api/chat/stream       SSE 流式对话
GET  /api/sessions          会话列表
POST /api/files/upload      上传文件（multipart）
GET  /api/files             文件列表
GET  /api/knowledge/stats   知识库统计
GET  /api/usage             Token 用量
```

完整 API 文档：启动后访问 `http://localhost:7860/docs`（自动生成 Swagger UI）。

## 项目结构

```
RAG-question/
├── server.py              # FastAPI 入口
├── config.py              # 全局配置
├── agent/                 # Agent 核心层（ReAct 推理 + LLM + 记忆）
├── knowledge/             # 知识层（文档解析 + 向量索引 + 检索管线）
├── evaluation/            # RAGAS 质量评估
├── static/index.html      # 前端 SPA
├── tests/                 # 单元测试
├── scripts/               # 工具脚本
└── data/                  # 运行时数据（gitignored）
```

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | DeepSeek (deepseek-chat) |
| Agent 框架 | LlamaIndex 0.12+ (ReAct) |
| Embedding | sentence-transformers (384d) / 纯 NumPy 本地 (512d) |
| 向量数据库 | ChromaDB |
| 后端 | FastAPI + SSE |
| 前端 | 原生 HTML/CSS/JS (零构建依赖) |
| 记忆 | SQLite + ChatMemoryBuffer |

## 评估

```bash
python evaluation/run_eval.py          # 运行 RAGAS 评估
python evaluation/run_eval.py --keep   # 评估后保留临时数据
```

## AI 开发指南

本项目为 AI 编码助手（Claude Code）维护了 `CLAUDE.md`，包含架构详图、设计决策、约束条件和错误排查记录。
