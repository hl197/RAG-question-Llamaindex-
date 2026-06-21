"""
纯本地 Embedding 实现 —— 零下载、零依赖、完全离线。

原理：基于字符特征 + 随机投影（Random Projection），
将文本转换为固定维度的向量，支持中文和多语言。

无需任何模型文件，只依赖 numpy。
"""

import hashlib
import math
from typing import List, Optional

import numpy as np
from llama_index.core.embeddings import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    """
    本地 Embedding 模型，继承 LlamaIndex BaseEmbedding。

    将文本转为固定维度的向量，使用随机投影 + 字符 n-gram 特征。
    所有向量 L2 归一化后可用于余弦相似度检索。
    """

    def __init__(
        self,
        dim: int = 256,
        seed: int = 42,
        **kwargs,
    ):
        """
        Args:
            dim: 输出向量维度（256~512 较合适，越大越精确）
            seed: 随机种子，保证多次运行结果一致
        """
        super().__init__(embed_dim=dim, **kwargs)
        # Pydantic 模型不允许 setattr 未声明的字段，使用 object.__setattr__ 绕过
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "_rng", np.random.RandomState(seed))
        # 字符 n-gram 的哈希空间大小（2^20 ≈ 100万，足够避免大量碰撞）
        object.__setattr__(self, "hash_space", 2 ** 20)

        # 随机投影矩阵：hash_space × dim
        object.__setattr__(self, "_projection", self._build_projection())

    def _build_projection(self):
        """构建随机投影矩阵（纯函数，无副作用）"""
        scale = math.sqrt(3)
        n = self.hash_space
        d = self.dim
        proj = self._rng.choice(
            [0, scale, -scale],
            size=(n, d),
            p=[2/3, 1/6, 1/6],
        )
        return proj.astype(np.float32)

    def _hash_feature(self, feature: str) -> int:
        """将特征字符串哈希到 [0, hash_space) 范围"""
        h = hashlib.md5(feature.encode("utf-8")).hexdigest()
        return int(h, 16) % self.hash_space

    def _extract_features(self, text: str) -> dict:
        """
        提取字符特征及其权重。

        特征类型：
        - 单字（char unigram）: 权重 1
        - 二字组（char bigram）: 权重 2
        - 三字组（char trigram）: 权重 1
        """
        if not text or not text.strip():
            return {}

        text = text.strip()
        features = {}

        # 单字（unigram）
        for ch in text:
            if ch.strip():
                features[f"1:{ch}"] = features.get(f"1:{ch}", 0) + 1

        # 二字组（bigram）
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            if bigram.strip():
                features[f"2:{bigram}"] = features.get(f"2:{bigram}", 0) + 2

        # 三字组（trigram）
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            if trigram.strip():
                features[f"3:{trigram}"] = features.get(f"3:{trigram}", 0) + 1

        return features

    def _build_vector(self, text: str) -> List[float]:
        """将文本转为向量（核心逻辑）"""
        if not text or not text.strip():
            return [0.0] * self.dim

        features = self._extract_features(text)

        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in features.items():
            idx = self._hash_feature(feat)
            w = math.sqrt(weight)
            vec += self._projection[idx] * w

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec.tolist()

    # ── BaseEmbedding 抽象方法实现 ──────────────

    def _get_text_embedding(self, text: str) -> List[float]:
        """单条文本嵌入（BaseEmbedding 抽象方法）"""
        return self._build_vector(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量文本嵌入"""
        return [self._build_vector(t) for t in texts]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步单条嵌入"""
        return self._build_vector(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入"""
        return [self._build_vector(t) for t in texts]

    # ── 查询嵌入（用于 query 嵌入，与文档嵌入用同一逻辑） ──

    def _get_query_embedding(self, query: str) -> List[float]:
        """查询嵌入（BaseEmbedding 可选覆盖，不覆盖则 fallback 到 _get_text_embedding）"""
        return self._build_vector(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步查询嵌入"""
        return self._build_vector(query)

    @classmethod
    def class_name(cls) -> str:
        return "LocalEmbedding"
