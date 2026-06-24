"""
重排序器 —— 检索粗筛后精排，提升 top-K 精准度。

检索流程: 粗筛 top-N (20) → 精排 top-K (5~8)
精排使用 LLM 打分 + 关键词重叠双路评分，无需额外模型。

使用方式:
    reranker = Reranker(llm)
    reranked = reranker.rerank(query, candidates, top_k=5)
"""

from typing import List, Tuple


class Reranker:
    """
    两阶段重排序：
    1. 关键词快速评分（精确匹配加分）
    2. LLM 相关性打分（语义相关性）

    最终分数 = 0.3 × keyword_score + 0.7 × llm_score
    """

    def __init__(self, llm):
        self._llm = llm

    def _keyword_score(self, query: str, text: str) -> float:
        """基于关键词重叠的快速评分"""
        import re
        # 按标点和空格分词，保留 ≥2 字的词作为关键词
        tokens = re.split(r'[\s，。！？、；：""''（）()\n,.!?;:]+', query)
        query_words = {t.strip() for t in tokens if len(t.strip()) >= 2}

        if not query_words:
            return 0.0

        matched = sum(1 for w in query_words if w in text)
        return min(1.0, matched / len(query_words))

    def _llm_score(self, query: str, texts: List[str]) -> List[float]:
        """使用 LLM 对候选文本进行相关性打分"""
        candidates_text = ""
        for i, text in enumerate(texts):
            truncated = text[:200].replace("\n", " ")
            candidates_text += f"[{i}] {truncated}\n"

        prompt = (
            "评估以下文本片段与问题的相关性，为每个片段打分（0=无关, 1=高度相关）。\n"
            "只输出 JSON 数组，如 [0.8, 0.3, 0.9]，不要其他文字。\n\n"
            f"问题: {query}\n\n"
            f"{candidates_text}\n"
            "相关性分数:"
        )

        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()

            # 清理格式
            import re
            text = re.sub(r'```(?:json)?\s*', '', text)
            text = re.sub(r'```', '', text)
            text = text.strip()

            import json
            scores = json.loads(text)
            if isinstance(scores, list) and len(scores) == len(texts):
                return [float(s) for s in scores]
        except Exception:
            pass

        # 降级：仅用关键词评分
        return [self._keyword_score(query, t) for t in texts]

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        use_llm: bool = True,
    ) -> List[str]:
        """
        对候选文本重排序。

        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回数量
            use_llm: 是否使用 LLM 打分（关闭则仅用关键词）

        Returns:
            排序后的文本列表（即使 candidates 数量 ≤ top_k 也会排序后返回）
        """

        # 1. 关键词快速评分
        kw_scores = [self._keyword_score(query, t) for t in candidates]

        # 2. LLM 精排
        if use_llm and self._llm and len(candidates) > 1:
            llm_scores = self._llm_score(query, candidates)
        else:
            llm_scores = kw_scores

        # 3. 融合分数
        final_scores = [
            0.3 * kw + 0.7 * llm
            for kw, llm in zip(kw_scores, llm_scores)
        ]

        # 4. 排序取 top_k（即使候选少也重新排序）
        ranked = sorted(
            enumerate(final_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [candidates[i] for i, _ in ranked[:top_k]]
