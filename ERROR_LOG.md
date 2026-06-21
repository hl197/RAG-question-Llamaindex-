# 错误记录与解决方案

> 本文档记录开发过程中遇到的所有错误及其解决方案。
> 最后更新: 2026-06-21

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

## 汇总：按文件相关

| 文件 | 错误数 | 主要问题类型 |
|------|--------|-------------|
| `knowledge/local_embedding.py` | 3 | Pydantic 字段校验 (#6)、BaseEmbedding 继承 (#12) |
| `agent/deepseek_llm.py` | 3 | Pydantic 字段校验 (#7)、缺少抽象方法 (#9) |
| `agent/rag_agent.py` | 2 | Tool 构造参数错误 (#10, #11) |
| `app.py` | 4 | Gradio 6.x API 不兼容 (#2) |
| `config.py` | 1 | Embedding 配置项变更 (#1) |
