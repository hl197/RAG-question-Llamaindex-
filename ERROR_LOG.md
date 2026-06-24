写一个专门的llm类用于从环境变量中加载api，供其他类调用llm
写一个数据结构类，用来规范各部分的参数接口规范



# 错误记录与解决方案

> 本文档记录开发过程中遇到的所有错误及其解决方案。
> 最后更新: 2026-06-22

---

## 1. DeepSeek 无 Embedding API

**错误**:
```
POST /v1/embeddings → 404 Not Found
```
DeepSeek API 不支持 `/v1/embeddings` 端点，无法使用 `llama_index.embeddings.deepseek.DeepSeekEmbedding`。

**解决过程**:
1. 尝试 `HuggingFaceEmbedding`（本地模型 `BAAI/bge-small-zh-v1.5`）→ huggingface.co 被墙，下载卡死
2. 设置 `HF_ENDPOINT=https://hf-mirror.com` → 下载进度条不动
3. **最终方案**: 手写纯 numpy 本地 Embedding → `knowledge/local_embedding.py`
4. 配置项从 `EMBED_MODEL = "BAAI/bge-small-zh-v1.5"` 改为 `EMBED_DIM = 256`

**关键文件**: `knowledge/local_embedding.py`

---

## 2. Gradio 6.x API 不兼容

**错误**:
```
TypeError: Chatbot.__init__() got an unexpected keyword argument 'bubble_full_width'
TypeError: Blocks.__init__() got an unexpected keyword argument 'theme'
TypeError: Blocks.__init__() got an unexpected keyword argument 'css'
TypeError: Blocks.__init__() got an unexpected keyword argument 'title'
TypeError: launch() got an unexpected keyword argument 'title'
```

**原因**: 安装的 Gradio 为 6.x 版本，移除了 `bubble_full_width` 参数，且 `theme`/`css`/`title` 需要从 `Blocks()` 移到 `launch()` 调用。`title` 在 6.x 中已被移除。

**解决**:
- 删除所有 `Chatbot` 中的 `bubble_full_width`
- `theme` 参数移到 `launch(theme=...)` 中
- `css` 参数移到 `launch(css=CSS)` 中
- `title` 参数直接删除

**关键文件**: `app.py`

---

## 3. Gradio 6.x → huggingface-hub 版本冲突

**错误**:
```
ImportError: huggingface-hub>=0.34.0,<1.0 required, but 0.36.2+xxx is installed
```
Gradio 6.19 要求 `huggingface-hub>=1.2.0`，而 `transformers` 要求 `<1.0`。

**解决**: 移除 `transformers` 和 `HuggingFaceEmbedding` 依赖，改用纯 numpy LocalEmbedding，不再需要 huggingface-hub。

---

## 4. llama-index-vector-stores-chroma 未安装

**错误**:
```
ModuleNotFoundError: No module named 'llama_index.vector_stores.chroma'
```

**解决**:
```powershell
C:\Users\86182\.conda\envs\all-in-rag\python.exe -m pip install llama-index-vector-stores-chroma
```
安装时需注意版本锁定，避免自动升级 llama-index-core:
```powershell
python -m pip install "llama-index-vector-stores-chroma<0.5" "llama-index-core<0.13"
```

---

## 5. Chinese Quotation Marks SyntaxError

**错误**:
```
SyntaxError: invalid character '……' (U+2026)
```
`__init__.py` 中使用中文引号 `""` (U+201C/U+201D) 作为 Python 字符串的引号，被解释器当作非法字符。

**解决**: 将所有 `""` 和 `''` 替换为 `[]` 或标准 ASCII 引号。

---

## 6. Pydantic v2 严格属性校验 (BaseEmbedding)

**错误**:
```
ValueError: "LocalEmbedding" object has no field "dim"
```
`LocalEmbedding` 继承自 `BaseEmbedding`（Pydantic v2 BaseModel），直接 `self.dim = dim` 赋值失败，因为 `dim` 不是声明的 Pydantic 字段。

**根因**: 所有继承 Pydantic BaseModel 的类（包括 LlamaIndex 的所有 `Settings` 类）在 `__init__` 之后不允许 `setattr` 未声明的属性。

**解决**: 使用 `object.__setattr__(self, "attr_name", value)` 绕过 Pydantic 的字段校验。

**关键文件**: `knowledge/local_embedding.py` — `__init__` 中所有内部属性都用 `object.__setattr__`

---

## 7. Pydantic v2 严格属性校验 (DeepSeekLLM)

**错误**:
```
ValueError: "DeepSeekLLM" object has no field "_client"
```
同 #6，`DeepSeekLLM` 继承自 `LLM`（BaseModel），`self._client = OpenAIClient(...)` 失败。

**解决**: 同样使用 `object.__setattr__(self, "_client", ...)`

**关键文件**: `agent/deepseek_llm.py`

---

## 8. LlamaIndex OpenAI 模型名白名单校验

**错误**:
```
ValueError: Unknown model 'deepseek-chat'. Please provide a valid OpenAI model name in: o1, gpt-4o, ...
```
`llama_index.llms.openai.OpenAI` 内部调用 `openai_modelname_to_contextsize()` 校验模型名，`deepseek-chat` 不在白名单中。

**解决**: 手写自定义 `DeepSeekLLM` 类，继承 `llama_index.core.llms.LLM`，底层用原生 `openai.OpenAI` SDK 调用 DeepSeek API，完全绕过 llama-index 的模型名校验。

**关键文件**: `agent/deepseek_llm.py`

---

## 9. DeepSeekLLM 缺少抽象方法实现

**错误**:
```
TypeError: Can't instantiate abstract class DeepSeekLLM without an implementation for abstract methods 
'achat', 'acomplete', 'astream_chat', 'astream_complete'
```
`llama_index.core.llms.LLM` 基类有 8 个抽象方法，只实现了同步 4 个（chat/complete/stream_chat/stream_complete），缺少异步 4 个。

**解决**: 补充 `achat`、`acomplete`、`astream_chat`、`astream_complete`，内部用 `asyncio.to_thread()` 委托同步方法。

**关键文件**: `agent/deepseek_llm.py`

---

## 10. QueryEngineTool.from_defaults() 参数错误

**错误**:
```
TypeError: QueryEngineTool.from_defaults() got an unexpected keyword argument 'metadata'
```

**原因**: 新版 LlamaIndex 的 `QueryEngineTool.from_defaults()` 不接受 `metadata=ToolMetadata(...)`，需要分开传 `name=` 和 `description=`。

**解决**: 
```python
# 错误
QueryEngineTool.from_defaults(query_engine=..., metadata=ToolMetadata(name=..., description=...))

# 正确
QueryEngineTool.from_defaults(query_engine=..., name="knowledge_base", description="...")
```

**关键文件**: `agent/rag_agent.py` — `_build_tools()` 方法

---

## 11. FunctionTool.from_defaults() 参数错误

**错误**:
```
TypeError: FunctionTool.from_defaults() got an unexpected keyword argument 'metadata'
```

**原因**: 同 #10，`FunctionTool.from_defaults()` 也不接受 `metadata=`，需要 `name=` 和 `description=`。

**解决**: 同样改为分开传参数。

**关键文件**: `agent/rag_agent.py` — `_build_tools()` 方法

---

## 12. `assert isinstance(embed_model, BaseEmbedding)` 失败

**错误**:
```
AssertionError at resolve_embed_model()
```

**原因**: `LocalEmbedding` 最初是普通类，没有继承 `BaseEmbedding`，`Settings.embed_model = embed_model` 时会校验类型。

**解决**: 让 `LocalEmbedding` 继承 `llama_index.core.embeddings.BaseEmbedding` 并实现抽象方法 `_get_text_embedding`、`_get_text_embeddings`、`_get_query_embedding` 等。

**关键文件**: `knowledge/local_embedding.py`

---

## 13. pymupdf4llm 导入路径变更

**错误**:
```
上传失败: 处理失败: 请安装 pymupdf4llm: pip install pymupdf4llm
```
但 `pymupdf4llm` 实际已安装。

**根因**: `pymupdf4llm` 1.27.2.3 版本中 `LlamaMarkdownReader` 从 `pymupdf4llm.llama` 移到了 `pymupdf4llm` 顶层。

**解决**: 
```python
# 错误
from pymupdf4llm.llama import LlamaMarkdownReader

# 正确
from pymupdf4llm import LlamaMarkdownReader
```

**关键文件**: `knowledge/loader.py`

---

## 14. 系统 HTTP_PROXY 环境变量干扰 API 直连

**错误**:
```
对话时 → Connection error 或 Authentication Fails (governor)
```
但直接测试 `api.deepseek.com` 的网络连接是通的（DNS 解析成功、TCP 连得上）。

**根因**: 系统设置了 `HTTP_PROXY=http://127.0.0.1:9674`，且代理客户端未运行。Python 的 httpx 库在创建客户端时自动读取该环境变量走代理，导致请求被拒绝。

**解决过程**:
1. ❌ `httpx.Client(proxy=None)` — 无效，httpx 仍读取环境变量
2. ❌ `httpx.Client(mounts={})` — 无效，内部默认传输层仍读取环境变量
3. ❌ `os.environ.pop("HTTP_PROXY")` — 只对当前进程有效，openai SDK 内部有缓存
4. ❌ `NO_PROXY=api.deepseek.com` — httpx 部分版本不识别
5. ✅ `httpx.HTTPTransport(proxy=None)` — 在传输层彻底禁用代理

**方案对比**:
| 方法 | 效果 | 说明 |
|------|------|------|
| `os.environ.pop()` | 部分有效 | 在当前进程清除，但 httpx 有内部缓存 |
| `httpx.Client(proxy=None)` | ❌ 无效 | 文档说绕过代理，实际仍读环境变量 |
| `httpx.Client(mounts={})` | ❌ 无效 | 挂载点清空后默认传输层仍读代理 |
| `httpx.HTTPTransport(proxy=None)` | ✅ 有效 | 在传输层彻底禁用代理读取 |
| `NO_PROXY` 环境变量 | 辅助有效 | 需配合传输层方案一起使用 |
| `create_llm()` 适配器 | 架构改进 | 统一集中管理配置加载 |

**关键文件**: `agent/deepseek_llm.py`、`config.py`、`agent/llm_adapter.py`（新文件）

---

## 15. DeepSeek API 间歇性 Authentication Fails

**错误**:
```
第一次调用连接正常，后续某些调用出现:
Authentication Fails (governor)
```
或反过来：第一次失败，第二次成功。

**根因**: 疑似 DeepSeek API 服务端网关（governor）的间歇性问题。`governor` 是 DeepSeek 的 API 网关组件，在某些请求频率或上下文长度下会返回认证失败。该错误在直接测试时无法稳定复现，仅在 Agent 的多轮调用中出现。

**解决**:
1. 添加连接预热（Agent 初始化时发一条测试请求）
2. 添加自动重试机制（第一次失败时重建 Agent 再试）
3. 改进错误日志输出（显示完整 traceback 和异常类型）

**关键文件**: `agent/rag_agent.py`

---

## 16. SemanticEmbedding 加载时直连 HuggingFace 超时

**错误**:
```
HTTPSConnectionPool(host='huggingface.co', port=443): 
Max retries exceeded with url: /.../modules.json
(ConnectTimeoutError)
```

**根因**: `huggingface_hub` 库在被任何代码首次导入时即读取环境变量并缓存配置。`datasets`/`ragas` 等库在 `run_eval.py` 设置 `HF_ENDPOINT`/`HF_HUB_OFFLINE` 之前已被导入，导致后续 `SemanticEmbedding.__init__` 中设置环境变量无效。

**解决过程**:
1. ❌ 在 `semantic_embedding.py` 模块级设置 env var — 无效，`huggingface_hub` 已被 `datasets` 先导入
2. ❌ 在 `semantic_embedding.py.__init__` 中设置并传 `local_files_only=True` — 无效，transformers AutoProcessor 版本不兼容
3. ❌ 仅设 `HF_HUB_OFFLINE=1` — 模型权重已缓存但 tokenizer/processor 配置缺失
4. ✅ **三重保护**：
   - bash 层：`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com`
   - Python 层：`run_eval.py` 顶部（`import sys` 之前）用 `os.environ.setdefault()` 注入三项
   - 模块层：`semantic_embedding.py` 顶部同样 setdefault（兜底）

**关键文件**: `evaluation/run_eval.py`（第 12-15 行）、`knowledge/semantic_embedding.py`

---

## 汇总：按文件相关

| 文件 | 错误数 | 主要问题类型 |
|------|--------|-------------|
| `knowledge/local_embedding.py` | 3 | Pydantic 字段校验 (#6)、BaseEmbedding 继承 (#12) |
| `agent/deepseek_llm.py` | 4 | Pydantic 字段校验 (#7)、缺少抽象方法 (#9)、httpx 代理绕过 (#14) |
| `agent/rag_agent.py` | 3 | Tool 构造参数错误 (#10, #11)、重试机制 (#15) |
| `agent/llm_adapter.py` | 1 | 新文件：LLM 配置适配器 (#14) |
| `app.py`（已废弃，被 server.py 替代） | 4 | Gradio 6.x API 不兼容 (#2) |
| `server.py`（新增） | — | FastAPI 替代 Gradio，SSE 流式 + REST API |
| `config.py` | 2 | Embedding 配置项变更 (#1)、NO_PROXY 设置 (#14) |
| `knowledge/loader.py` | 1 | pymupdf4llm 导入路径变更 (#13) |
| `knowledge/semantic_embedding.py` | 1 | HF 离线加载环境变量注入顺序 (#16) |
| `evaluation/run_eval.py` | 1 | HF 离线加载环境变量注入顺序 (#16) |

---

## 17. SSL_CERT_FILE 指向错误的 conda 安装

**错误**:
```
FileNotFoundError: [Errno 2] No such file or directory
ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
```

**根因**: 系统存在两个 conda 安装：`D:\Miniconda3`（base）和 `C:\Users\86182\.conda\`（all-in-rag 环境）。`~/.bash_profile` 中 `SSL_CERT_FILE=$CONDA_HOME/Library/ssl/cacert.pem` 指向 base conda 的证书路径，但实际运行环境是 `all-in-rag`，httpx 读取 `SSL_CERT_FILE` 时找到的路径在 base conda 下不存在。

**解决过程**:
1. ❌ 修改 `SSL_CERT_FILE` 指向 `all-in-rag` 环境路径 — 治标不治本，每次切换环境都要改
2. ✅ **在 `deepseek_llm.py` 中使用 `certifi.where()` 自动定位证书**，不再依赖 `SSL_CERT_FILE` 环境变量

**关键文件**: `agent/deepseek_llm.py`（httpx.HTTPTransport 添加 `verify=certifi.where()`）

---

## 18. HuggingFace 模型下载超时 + ChromaDB 维度不匹配

**错误**: 服务器启动后上传文件失败（500 Internal Server Error），日志显示多个错误：
1. `HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded` — huggingface.co 被墙
2. `SSL: UNEXPECTED_EOF_WHILE_READING` — 镜像站 hf-mirror.com SSL 连接被重置
3. 上传文件返回 500 — ChromaDB 维度不匹配（LocalEmbedding 256d 旧 collection vs SemanticEmbedding 384d）

**根因**:
1. 国内网络 huggingface.co 被 GFW 阻断，且代理（127.0.0.1:9674）也无法建立 SSL 握手
2. `hf-mirror.com` 的 CDN 域名（`cas-bridge.xethub.hf-mirror.org`、`us.aws.cdn.hf-mirror.org`）DNS 解析失败
3. `knowledge/indexer.py` 的 `_init_embedding()` 硬编码了 `LocalEmbedding`，即使 `config.EMBED_TYPE="semantic"` 也会被覆盖，导致 ChromaDB 试图以 256d 创建 collection，后续 SemanticEmbedding（384d）插入时爆维度错误

**解决过程**:
1. ❌ 设置 `HF_ENDPOINT=https://hf-mirror.com` 并去掉 `HF_HUB_OFFLINE=1` — 仍 SSL 错误
2. ❌ 设置代理 `HTTP_PROXY=http://127.0.0.1:9674` — huggingface.co SSL 握手被重置
3. ❌ `git clone https://hf-mirror.com/...` — 仓库克隆成功但 LFS 大文件 CDN 域名不可达
4. ✅ **从 hf-mirror.com 用 `curl -sL --max-time 600` 直接下载 model.safetensors（449MB）和 tokenizer.json（8.7MB）等 LFS 文件到 `~/.cache/sentence-transformers/`**
5. ✅ **修改 `indexer.py._init_embedding()`**: 先检查 `Settings.embed_model` 是否已设置（由 `rag_agent.py` 决定），已设置则不覆盖
6. ✅ **marked.js 本地化**: 浏览器跟踪防护拦截 CDN，改为 `static/marked.min.js`

**关键文件**:
- `knowledge/semantic_embedding.py` — 从 `~/.cache/sentence-transformers/` 本地路径加载模型
- `knowledge/indexer.py` — `_init_embedding()` 不再硬编码 LocalEmbedding
- `static/index.html` — marked.js 改为本地引用
- `config.py` — 新增 `LLM_MAX_TOKENS=8192`、`LLM_CONTEXT_WINDOW`、`EMBED_TYPE` 等配置项
