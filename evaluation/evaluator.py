"""
RAGAS 评估核心

RAGEvaluator 负责：
  1. 在独立 ChromaDB collection 中索引测试文档
  2. 对每个问题执行检索 + 生成
  3. 调用 RAGAS evaluate() 计算 4 项指标并输出报告

RAGAS judge LLM 通过 instructor Mode.JSON 模式调用 DeepSeek
（ragas 默认即用 Mode.JSON，DeepSeek 兼容，无需额外配置）。
Embedding 使用项目自带的 LocalEmbedding 包装为 RAGAS 兼容接口。
"""

import asyncio
import json
import math
import os
import re
import shutil
from datetime import datetime
from typing import List, Dict, Optional

import chromadb
from datasets import Dataset
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import OpenAI

import config
from agent.llm_adapter import create_llm
from knowledge.local_embedding import LocalEmbedding
from knowledge.semantic_embedding import SemanticEmbedding

# RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding


class _LocalEmbeddingWrapper(BaseRagasEmbedding):
    """将项目 LocalEmbedding 包装为 RAGAS 兼容接口（同时支持 Langchain 接口）"""

    def __init__(self, embed_model: LocalEmbedding):
        self._embed = embed_model

    # ── RAGAS BaseRagasEmbedding 接口 ──

    def embed_text(self, text: str, **kwargs) -> List[float]:
        return self._embed.get_text_embedding(text)

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        return await self._embed.aget_text_embedding(text)

    # ── Langchain Embeddings 接口（answer_relevancy 内部使用） ──

    def embed_query(self, text: str) -> List[float]:
        return self._embed.get_text_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed.get_text_embedding(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return await self._embed.aget_text_embedding(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return [await self._embed.aget_text_embedding(t) for t in texts]


def _create_judge_llm():
    """创建 RAGAS judge LLM，指向 DeepSeek API（instructor Mode.JSON）"""
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )
    return llm_factory(config.LLM_MODEL, client=client)


class RAGEvaluator:
    """RAG 系统质量评估器（RAGAS 驱动）"""

    def __init__(self):
        self.llm = create_llm()
        Settings.llm = self.llm

        # 根据配置选择 Embedding 类型
        if config.EMBED_TYPE == "semantic":
            Settings.embed_model = SemanticEmbedding(model_name=config.SEMANTIC_MODEL_NAME)
        else:
            Settings.embed_model = LocalEmbedding(dim=config.EMBED_DIM)

        self.eval_dir = os.path.join(config.DATA_DIR, "chroma_eval")
        self.collection_name = "eval_test"

        self._client: Optional[chromadb.PersistentClient] = None
        self._index: Optional[VectorStoreIndex] = None
        self._query_engine: Optional[RetrieverQueryEngine] = None

    # ── 初始化 ──────────────────────────────────

    def setup(self):
        """创建独立的 eval ChromaDB collection 并索引测试文档"""
        from evaluation.test_data import TEST_DOCUMENTS

        self._close_chroma()

        if os.path.exists(self.eval_dir):
            shutil.rmtree(self.eval_dir)
        os.makedirs(self.eval_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.eval_dir)
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        docs = [
            Document(text=td["content"], metadata={"source_file": td["filename"]})
            for td in TEST_DOCUMENTS
        ]

        splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(docs)
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=config.SIMILARITY_TOP_K,
        )
        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)],
            llm=None,
        )

        print(f"✅ Eval 索引就绪: {len(nodes)} 个文本块, {len(TEST_DOCUMENTS)} 篇文档")

    def _close_chroma(self):
        """关闭 ChromaDB 客户端释放文件锁"""
        if self._client is not None:
            try:
                self._client._system.stop()
            except Exception:
                pass
            self._client = None
            self._index = None
            self._query_engine = None

    # ── 检索 + 生成 ─────────────────────────────

    def _retrieve(self, question: str) -> List[str]:
        """检索相关文档片段，返回文本列表"""
        response = self._query_engine.query(question)
        return [node.text for node in response.source_nodes]

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """基于检索到的文档片段生成回答"""
        context_text = "\n\n---\n\n".join(contexts)
        prompt = (
            "你是一个精准的文档问答助手。请**严格基于**以下参考资料回答问题。\n"
            '如果参考资料中没有相关信息，请如实说"未找到相关信息"。\n\n'
            f"## 参考资料\n{context_text}\n\n"
            f"## 问题\n{question}\n\n"
            "请用中文简要回答："
        )
        response = self.llm.complete(prompt)
        return response.text.strip()

    # ── 运行评估 ────────────────────────────────

    def run(self) -> Dict:
        """执行完整评估流程，返回结果字典"""
        from evaluation.test_data import TEST_QA_PAIRS

        if self._query_engine is None:
            self.setup()

        questions, answers, contexts_list, ground_truths = [], [], [], []

        print(f"\n{'='*60}")
        print(f"🔍 开始 RAGAS 评估 — {len(TEST_QA_PAIRS)} 个测试问题")
        print(f"{'='*60}\n")

        for i, qa in enumerate(TEST_QA_PAIRS, 1):
            question = qa["question"]
            print(f"📝 [{i}/{len(TEST_QA_PAIRS)}] {question}")

            contexts = self._retrieve(question)
            print(f"   📥 检索到 {len(contexts)} 个片段")

            answer = self._generate_answer(question, contexts)
            print(f"   🤖 回答: {answer[:80]}...")

            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(qa["ground_truth"])

        # 组装 RAGAS Dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        })

        print(f"\n{'='*60}")
        print(f"📊 正在调用 RAGAS evaluate() 计算指标...")
        print(f"{'='*60}\n")

        # 创建指向 DeepSeek 的 judge LLM + Embedding
        judge_llm = _create_judge_llm()
        judge_embedding = _LocalEmbeddingWrapper(Settings.embed_model)
        print(f"🔧 RAGAS Judge LLM: {config.LLM_MODEL} @ DeepSeek (instructor Mode.JSON)")
        print(f"🔧 RAGAS Embedding: LocalEmbedding ({config.EMBED_DIM}d)")

        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embedding,
        )

        return self._process_result(result, questions, answers)

    def _process_result(self, result, questions, answers) -> Dict:
        """处理评估结果，生成报告"""
        df = result.to_pandas()

        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        scores = {}
        for name in metric_names:
            if name in df.columns:
                vals = [v for v in df[name] if not (isinstance(v, float) and math.isnan(v))]
                scores[name] = round(float(sum(vals) / len(vals)), 4) if vals else 0.0

        details = []
        for i in range(len(questions)):
            detail = {"question": questions[i][:100], "answer": answers[i][:200]}
            for name in metric_names:
                if name in df.columns and i < len(df):
                    val = df[name].iloc[i]
                    detail[name] = round(float(val), 4) if not (isinstance(val, float) and math.isnan(val)) else None
            details.append(detail)

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": config.LLM_MODEL,
            "embed_dim": config.EMBED_DIM,
            "chunk_size": config.CHUNK_SIZE,
            "similarity_top_k": config.SIMILARITY_TOP_K,
            "num_questions": len(questions),
            "scores": scores,
            "details": details,
        }

        report_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    # ── 清理 ────────────────────────────────────

    def cleanup(self):
        """删除 eval ChromaDB 数据"""
        self._close_chroma()
        if os.path.exists(self.eval_dir):
            try:
                shutil.rmtree(self.eval_dir)
                print("🗑️  已清理 Eval 数据")
            except PermissionError:
                print("⚠️  无法清理 Eval 数据（文件可能被占用），请手动删除 data/chroma_eval/")

    # ── 报告输出 ────────────────────────────────

    def print_report(self, report: Dict):
        """格式化打印评估报告"""
        scores = report["scores"]

        print(f"\n{'='*60}")
        print(f"📋 RAGAS 评估报告")
        print(f"{'='*60}")
        print(f"模型: {report['model']}")
        print(f"Embedding: {config.EMBED_TYPE} ({config.SEMANTIC_MODEL_NAME if config.EMBED_TYPE == 'semantic' else f'{config.EMBED_DIM}d'}))")
        print(f"分块大小/重叠: {report['chunk_size']}/{config.CHUNK_OVERLAP}")
        print(f"检索 Top-K: {report['similarity_top_k']}")
        print(f"测试问题数: {report['num_questions']}")
        print()

        print(f"{'指标':<25} │ {'得分':<8} │ 说明")
        print(f"{'─'*25}─┼─{'─'*8}─┼─{'─'*35}")

        desc = {
            "faithfulness": ("回答忠实度", "回答中的断言有多少能在文档中找到依据"),
            "answer_relevancy": ("答案相关性", "回答是否紧扣问题，有无跑题"),
            "context_precision": ("检索精准度", "检索到的文档有多少是真正相关的"),
            "context_recall": ("检索召回率", "标准答案的信息有多少被检索覆盖"),
        }

        diagnostics = []

        for key, (label, explain) in desc.items():
            val = scores.get(key, 0)
            if val is not None and not math.isnan(val):
                filled = max(0, min(20, int(val * 20)))
                bar = "█" * filled + "░" * (20 - filled)
                print(f"{label:<24} │ {val:.4f} │ {bar} {explain}")

                if val < 0.5:
                    diagnostics.append(f"🔴 {label} 偏低 ({val:.2f})，需要重点关注")
                elif val < 0.7:
                    diagnostics.append(f"🟡 {label} 一般 ({val:.2f})，有优化空间")

        print()

        if diagnostics:
            print("📌 诊断建议:")
            for d in diagnostics:
                print(f"   {d}")
            print()
            print("   可尝试调整以下参数后重新评估:")
            print(f"   - CHUNK_SIZE (当前 {config.CHUNK_SIZE})")
            print(f"   - CHUNK_OVERLAP (当前 {config.CHUNK_OVERLAP})")
            print(f"   - SIMILARITY_TOP_K (当前 {config.SIMILARITY_TOP_K})")
            print(f"   - EMBED_DIM (当前 {config.EMBED_DIM})")
        else:
            print("✅ 所有指标良好，系统表现稳定。")

        print(f"\n📄 详细报告已保存至: evaluation/eval_report.json")
        print(f"{'='*60}\n")

    # ── 工具方法 ────────────────────────────────

    @staticmethod
    def _parse_json_list(text: str) -> List[str]:
        """从 LLM 回复中提取 JSON 数组"""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return []
