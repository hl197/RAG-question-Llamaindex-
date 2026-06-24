"""
查询改写器 —— 将用户口语化问题改写为检索友好的查询语句。

向量检索对"关键词密度高、术语完整、无冗余修饰"的查询效果最好。
口语化问题（如"我想知道张三在Happy-LLM项目里到底做了啥工作"）
直接检索效果差，改写后（"张三 Happy-LLM 项目 工作职责 架构设计"）精度显著提升。

使用方式：
    rewriter = QueryRewriter(llm)
    query = rewriter.rewrite("我想知道张三在Happy-LLM项目里做了什么")
    # → "张三 Happy-LLM 项目 工作职责"
"""

from typing import Optional


class QueryRewriter:
    """
    查询改写器：用 LLM 将口语化/复杂问题转为检索友好语句。

    改写原则：
    - 提取核心实体和关键词
    - 去掉口语化修饰词、疑问语气、客套话
    - 保留关键术语（中英文专名、技术术语）不变
    - 不过度扩展（不添加原问题没有的信息）
    """

    def __init__(self, llm):
        self._llm = llm

    def rewrite(self, question: str, max_length: int = 100) -> str:
        """
        将问题改写为检索友好的查询。

        Args:
            question: 用户原始问题
            max_length: 改写结果最大长度

        Returns:
            改写后的查询字符串（失败时返回原问题）
        """
        prompt = (
            "你是一个检索查询优化专家。将用户的自然语言问题改写成"
            "最适合向量检索的查询语句。\n\n"
            "规则：\n"
            "1. 提取核心实体和关键词，去掉口语化修饰词\n"
            "2. 去掉疑问语气（吗、呢、什么、怎么）和客套话\n"
            "3. 保留关键术语、技术名词、人名、项目名不变\n"
            "4. 不要添加原问题没有的信息\n"
            "5. 如果问题已经简洁清晰，原样返回\n"
            "6. 直接返回改写结果，不要解释、不要加引号\n\n"
            f"原问题: {question}\n"
            "改写查询:"
        )

        try:
            response = self._llm.complete(prompt)
            rewritten = response.text.strip().strip('"\'')
            if not rewritten or len(rewritten) > max_length:
                return question
            return rewritten
        except Exception:
            return question

    async def arewrite(self, question: str, max_length: int = 100) -> str:
        """异步版本"""
        return self.rewrite(question, max_length)
