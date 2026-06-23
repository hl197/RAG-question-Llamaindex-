"""
查询分解器 —— 将复杂查询拆分为子查询，分别检索后合并。

解决跨文档综合题（如 Q6）中单个向量无法同时覆盖
多个语义不相关问题域的缺陷。

使用方式：
    decomposer = QueryDecomposer(llm)
    sub_queries = decomposer.decompose("asyncio 和迁移学习有什么相似之处")
    # → ["asyncio 的切换开销为什么小", "迁移学习的核心思想是什么"]
"""

import re
from typing import List


class QueryDecomposer:
    """
    使用 LLM 将复杂查询分解为 2-3 个独立子查询。

    判断标准：如果问题包含多个问号、或连接词（和/与/以及/还有），
    则尝试分解；简单问题直接返回原问题。
    """

    def __init__(self, llm):
        self._llm = llm

    def _is_complex(self, question: str) -> bool:
        """启发式判断是否需要分解"""
        # 多个问号
        if question.count("?") + question.count("？") > 1:
            return True
        # 跨主题连接词
        connectors = ["和", "与", "以及", "还有", "同时", "另外", "此外"]
        # 只有当句子较长且包含连接词时才分解
        if len(question) > 30 and any(f" {c} " in question for c in connectors):
            return True
        return False

    def decompose(self, question: str) -> List[str]:
        """
        将复杂问题分解为子问题列表。

        Args:
            question: 原始问题

        Returns:
            子问题列表（至少包含原问题）
        """
        if not self._is_complex(question):
            return [question]

        prompt = (
            "将以下复杂问题拆分为 2-3 个独立的子问题，每个子问题应能单独检索回答。\n"
            "规则：\n"
            "1. 每个子问题一行，不要编号\n"
            "2. 子问题之间语义独立，不要重叠\n"
            "3. 保持原问题的关键术语不变\n"
            "4. 如果问题本身已足够简单，直接返回原问题\n"
            "\n"
            f"问题: {question}\n"
            "子问题:"
        )

        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()
        except Exception:
            return [question]

        # 解析子问题列表
        lines = []
        for line in text.split("\n"):
            # 去除编号（如 "1. "、"1) "、" - "）
            cleaned = re.sub(r'^[\d]+[\.\)、\s]\s*', '', line.strip())
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)
            if cleaned and len(cleaned) > 3:
                lines.append(cleaned)

        if len(lines) <= 1:
            return [question]

        return lines[:3]  # 最多 3 个子问题

    async def adecompose(self, question: str) -> List[str]:
        """异步版本"""
        return self.decompose(question)
