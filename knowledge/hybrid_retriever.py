"""
混合检索器 —— 向量检索 + BM25 关键词检索，加权融合。

解决纯向量检索在精确关键词匹配上的不足（如术语、编号、代码）。
使用倒数排名融合（RRF）合并两路结果，无需调参。

依赖: rank-bm25 (pip install rank-bm25)

使用方式:
    retriever = HybridRetriever(node_texts, vector_query_fn)
    results = retriever.retrieve("Python 协程", top_k=8)
"""

from typing import List, Callable, Tuple

from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    混合检索器：向量语义 + BM25 关键词。

    检索流程：
    1. 向量检索 top-K×2 候选 → score = cos_sim
    2. BM25 检索 top-K×2 候选 → score = bm25_score
    3. RRF 融合去重 → 返回 top-K
    """

    def __init__(
        self,
        node_texts: List[str],
        vector_query_fn: Callable[[str, int], List[Tuple[str, float]]],
    ):
        """
        Args:
            node_texts: 所有文档分块的文本列表（用于构建 BM25 索引）
            vector_query_fn: 向量检索函数，签名为 (query, top_k) -> [(text, score), ...]
        """
        self._node_texts = node_texts
        self._vector_query = vector_query_fn

        # 构建 BM25 索引（对中文做简单分词：按字符二元组分词）
        tokenized = [self._tokenize(t) for t in node_texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词：中文按字符二元组 + 空格分词"""
        tokens = []
        # 按空格/标点分割
        import re
        words = re.split(r'[\s，。！？、；：""''（）\n]+', text)
        for word in words:
            if not word:
                continue
            # 英文单词直接保留
            if word.encode('utf-8').isascii():
                tokens.append(word.lower())
            else:
                # 中文按二元组切分
                tokens.append(word)  # 保留完整词
                for i in range(len(word) - 1):
                    tokens.append(word[i:i + 2])
        return tokens

    def retrieve(self, query: str, top_k: int = 8, alpha: float = 0.6) -> List[str]:
        """
        混合检索。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            alpha: 向量检索权重 (0-1)，默认 0.6（偏向语义）

        Returns:
            文档文本列表
        """
        if not self._node_texts or not self._bm25:
            return []

        # 1. 向量检索（多取一些候选）
        vec_k = min(top_k * 3, len(self._node_texts))
        vector_results = self._vector_query(query, vec_k)

        # 2. BM25 检索
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # 归一化 BM25 分数到 [0, 1]
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_norm = [s / bm25_max for s in bm25_scores]

        # 3. 向量分数字典
        vector_dict = {}
        for i, (text, score) in enumerate(vector_results):
            vector_dict[text] = max(vector_dict.get(text, 0), score)

        # 4. 加权融合（文本 → 加权分数）
        scores = {}
        for idx, text in enumerate(self._node_texts):
            v_score = vector_dict.get(text, 0)
            b_score = bm25_norm[idx]
            scores[idx] = alpha * v_score + (1 - alpha) * b_score

        # 5. 排序取 top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = [self._node_texts[idx] for idx, _ in ranked[:top_k] if scores[idx] > 0]

        return results if results else [t for t, _ in vector_results[:top_k]]

    def __len__(self):
        return len(self._node_texts)
