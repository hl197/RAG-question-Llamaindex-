"""
索引构建与检索模块

负责 ChromaDB 初始化、文档索引构建（增量）、查询检索。

提供两套接口：
- KnowledgeIndex 类：推荐使用，封装状态，支持多实例
- 模块级函数：向后兼容，委托给默认单例
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from knowledge.local_embedding import LocalEmbedding

import config


class KnowledgeIndex:
    """知识库索引管理器 —— 封装 ChromaDB 连接和索引生命周期。

    替代原有的模块级全局变量，消除线程安全隐患。
    """

    def __init__(self, chroma_dir: str = None):
        self._chroma_dir = chroma_dir or config.CHROMA_DIR
        self._client: Optional[chromadb.PersistentClient] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._storage_context: Optional[StorageContext] = None
        self._index: Optional[VectorStoreIndex] = None
        self._loaded_files: List[str] = []

        # 文件注册表路径
        self._registry_path = os.path.join(self._chroma_dir, "file_registry.json")

    # ── 属性 ────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._index is not None

    # ── Embedding 初始化 ─────────────────────────

    def _init_embedding(self):
        """初始化 Embedding 模型"""
        if Settings.embed_model is not None:
            return  # 已有外部设置的 embedding（如 rag_agent.py 设置的语义模型）

        if config.EMBED_TYPE == "semantic":
            from knowledge.semantic_embedding import SemanticEmbedding
            Settings.embed_model = SemanticEmbedding()
        else:
            Settings.embed_model = LocalEmbedding(dim=config.EMBED_DIM)

    # ── 文件注册表 ───────────────────────────────

    def _save_file_registry(self):
        """保存文件注册表"""
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(self._loaded_files, f, ensure_ascii=False, indent=2)

    def _load_file_registry(self) -> List[str]:
        """加载文件注册表"""
        if os.path.exists(self._registry_path):
            with open(self._registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _get_collection(self):
        """获取 ChromaDB 集合对象"""
        if self._client is None:
            raise RuntimeError("ChromaDB 未初始化，请先调用 init_chroma()")
        return self._client.get_collection("knowledge_base")

    # ── ChromaDB 初始化 ──────────────────────────

    def init_chroma(self):
        """初始化 ChromaDB 客户端 + 向量存储 + 索引"""
        if self._index is not None:
            return

        self._init_embedding()
        os.makedirs(self._chroma_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._chroma_dir)
        collection = self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

        self._vector_store = ChromaVectorStore(chroma_collection=collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

        try:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                storage_context=self._storage_context,
            )
            print(f"✅ 从 ChromaDB 加载已有索引（共 {collection.count()} 条向量）")
        except Exception:
            self._index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=self._storage_context,
            )
            print("📂 创建新的空索引")

        self._loaded_files = self._load_file_registry()

    # ── 索引构建 ─────────────────────────────────

    def build_or_update_index(self, documents: List[Document], filename: str) -> int:
        """增量添加文档到索引。返回新增节点数。"""
        if self._index is None:
            raise RuntimeError("索引未初始化，请先调用 init_chroma()")

        if filename in self._loaded_files:
            print(f"⏭️ 文件已存在索引中，跳过: {filename}")
            return 0

        splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        for doc in documents:
            doc.metadata["source_file"] = filename

        nodes = splitter.get_nodes_from_documents(documents)
        self._index.insert_nodes(nodes)

        self._loaded_files.append(filename)
        self._save_file_registry()

        print(f"✅ 已索引文件: {filename}（{len(nodes)} 个文本块）")
        return len(nodes)

    # ── 检索引擎 ─────────────────────────────────

    def get_query_engine(
        self,
        similarity_top_k: Optional[int] = None,
        similarity_score_cutoff: Optional[float] = None,
    ) -> RetrieverQueryEngine:
        """获取查询引擎"""
        if self._index is None:
            raise RuntimeError("索引未初始化，请先调用 init_chroma()")

        top_k = similarity_top_k or config.SIMILARITY_TOP_K

        retriever = VectorIndexRetriever(index=self._index, similarity_top_k=top_k)

        postprocessors = []
        if similarity_score_cutoff is not None:
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=similarity_score_cutoff)
            )

        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            llm=None,
        )

    def get_node_texts(self) -> List[str]:
        """返回索引中所有文档分块的文本列表（供混合检索使用）"""
        if self._index is None:
            return []
        try:
            nodes = self._index.docstore.docs.values()
            return [node.text for node in nodes if hasattr(node, "text")]
        except Exception:
            return []

    # ── 文件管理 ─────────────────────────────────

    def remove_file(self, filename: str) -> bool:
        """从 ChromaDB 中删除指定文件名的所有文档片段"""
        try:
            collection = self._get_collection()
            collection.delete(where={"filename": filename})

            self._loaded_files = self._load_file_registry()
            if filename in self._loaded_files:
                self._loaded_files.remove(filename)
                self._save_file_registry()

            print(f"✅ 已从知识库中删除: {filename}")
            return True
        except Exception as e:
            print(f"❌ 删除文件失败: {e}")
            return False

    def list_uploaded_files(self) -> List[str]:
        """返回已上传的文件名列表"""
        if not self._loaded_files:
            self._loaded_files = self._load_file_registry()
        return list(self._loaded_files)

    def clear_knowledge_base(self):
        """清空知识库（删除所有向量 + 注册表）"""
        self.close()

        if os.path.exists(self._chroma_dir):
            shutil.rmtree(self._chroma_dir)
            print("🗑️ 已清空 ChromaDB 数据")

        self._vector_store = None
        self._storage_context = None
        self._index = None
        self._loaded_files = []

    def get_collection_stats(self) -> Dict:
        """返回集合统计信息"""
        try:
            collection = self._get_collection()
            return {
                "total_vectors": collection.count(),
                "files_count": len(self.list_uploaded_files()),
                "files": self.list_uploaded_files(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        """关闭 ChromaDB 客户端，释放文件锁"""
        if self._client is not None:
            try:
                self._client._system.stop()
            except Exception:
                pass
            self._client = None


# ── 默认单例 + 向后兼容的模块级函数 ─────────────────


_default_index: Optional[KnowledgeIndex] = None


def _get_default() -> KnowledgeIndex:
    """获取默认 KnowledgeIndex 单例"""
    global _default_index
    if _default_index is None:
        _default_index = KnowledgeIndex()
    return _default_index


def init_chroma():
    _get_default().init_chroma()


def build_or_update_index(documents: List[Document], filename: str) -> int:
    return _get_default().build_or_update_index(documents, filename)


def get_query_engine(
    similarity_top_k: Optional[int] = None,
    similarity_score_cutoff: Optional[float] = None,
) -> RetrieverQueryEngine:
    return _get_default().get_query_engine(similarity_top_k, similarity_score_cutoff)


def get_node_texts() -> List[str]:
    return _get_default().get_node_texts()


def remove_file(filename: str) -> bool:
    return _get_default().remove_file(filename)


def list_uploaded_files() -> List[str]:
    return _get_default().list_uploaded_files()


def clear_knowledge_base():
    _get_default().clear_knowledge_base()


def get_collection_stats() -> Dict:
    return _get_default().get_collection_stats()
