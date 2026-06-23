"""
语义 Embedding —— 基于 sentence-transformers 的真实语义模型。

与 LocalEmbedding（字符 n-gram + 随机投影）不同，本模块使用预训练的
Transformer 模型提取文本语义向量，能够理解同义词、近义表达和上下文语义。

首次使用会自动从 HF-Mirror 下载模型（~470MB），之后加载本地缓存。

使用方式：
    from knowledge.semantic_embedding import SemanticEmbedding
    embed = SemanticEmbedding()  # 默认 paraphrase-multilingual-MiniLM-L12-v2
    vec = embed.get_text_embedding("Python 异步编程")
"""

import asyncio
import os
from typing import List

# 必须在导入 sentence_transformers 之前设置，否则会直连 HuggingFace 超时
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 不强制离线模式，第一次使用会从镜像站下载模型（~470MB）
# 下载后自动缓存，后续运行时即使没网也能加载；
# 如需完全无网络运行，可在启动前设置:
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

from llama_index.core.embeddings import BaseEmbedding


class SemanticEmbedding(BaseEmbedding):
    """
    基于 sentence-transformers 的语义 Embedding 模型。

    默认使用 paraphrase-multilingual-MiniLM-L12-v2：
    - 维度: 384
    - 语言: 中文 + 英文 + 50+ 语言
    - 大小: ~470MB（首次下载后缓存）
    - 离线可用: 是（下载后无需网络）
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        **kwargs,
    ):
        from pathlib import Path
        from sentence_transformers import SentenceTransformer

        # 优先从本地缓存加载（国内网络可先用 download_model.py 从 ModelScope 下载）
        _local_cache = Path.home() / ".cache" / "sentence-transformers" / model_name
        if _local_cache.exists():
            model_path = str(_local_cache)
        else:
            model_path = model_name

        model = SentenceTransformer(model_path)
        dim = model.get_embedding_dimension() or model.get_sentence_embedding_dimension()

        super().__init__(embed_dim=dim, **kwargs)
        object.__setattr__(self, "_model", model)

    @classmethod
    def class_name(cls) -> str:
        return "SemanticEmbedding"

    # ── BaseEmbedding 抽象方法实现 ──────────────

    def _get_text_embedding(self, text: str) -> List[float]:
        """单条文本嵌入"""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量文本嵌入"""
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步单条嵌入"""
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入"""
        return await asyncio.to_thread(self._get_text_embeddings, texts)

    # ── 查询嵌入 ──────────────────────────────────

    def _get_query_embedding(self, query: str) -> List[float]:
        """查询嵌入"""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步查询嵌入"""
        return await self._aget_text_embedding(query)
