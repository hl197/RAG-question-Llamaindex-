"""
索引构建与检索模块

负责 ChromaDB 初始化、文档索引构建（增量）、查询检索。
"""

import json
import os
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


# ── 全局单例 ──────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_storage_context: Optional[StorageContext] = None
_index: Optional[VectorStoreIndex] = None
_loaded_files: List[str] = []


# ── Embedding 初始化 ──────────────────────────────
def _init_embedding():
    """初始化本地 Embedding 模型（零下载，纯 numpy）"""
    if Settings.embed_model is None or not isinstance(
        Settings.embed_model, LocalEmbedding
    ):
        embed_model = LocalEmbedding(dim=config.EMBED_DIM)
        Settings.embed_model = embed_model


# ── ChromaDB 初始化 ──────────────────────────────
def init_chroma():
    """
    初始化 ChromaDB 客户端 + 向量存储 + 索引。
    如果已有数据则加载，否则创建空索引。
    """
    global _chroma_client, _vector_store, _storage_context, _index, _loaded_files

    if _index is not None:
        return  # 已初始化

    _init_embedding()

    # 确保目录存在
    os.makedirs(config.CHROMA_DIR, exist_ok=True)

    # 连接/创建 ChromaDB
    _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = _chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )

    _vector_store = ChromaVectorStore(chroma_collection=collection)
    _storage_context = StorageContext.from_defaults(vector_store=_vector_store)

    # 尝试从已有数据恢复索引
    try:
        _index = VectorStoreIndex.from_vector_store(
            vector_store=_vector_store,
            storage_context=_storage_context,
        )
        print(f"✅ 从 ChromaDB 加载已有索引（共 {collection.count()} 条向量）")
    except Exception:
        # 没有数据，创建空索引
        _index = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=_storage_context,
        )
        print("📂 创建新的空索引")

    # 加载已上传文件列表
    _loaded_files = _load_file_registry()


# ── 文件注册表（记录已上传的文件） ──────────────
_FILE_REGISTRY_PATH = os.path.join(config.CHROMA_DIR, "file_registry.json")


def _save_file_registry():
    """保存文件注册表"""
    with open(_FILE_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(_loaded_files, f, ensure_ascii=False, indent=2)


def _load_file_registry() -> List[str]:
    """加载文件注册表"""
    if os.path.exists(_FILE_REGISTRY_PATH):
        with open(_FILE_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ── 索引构建 ──────────────────────────────────────
def build_or_update_index(
    documents: List[Document],
    filename: str,
) -> int:
    """
    增量添加文档到索引。

    Args:
        documents: 解析后的 Document 列表
        filename: 来源文件名

    Returns:
        int: 新增的节点数

    Raises:
        RuntimeError: 索引未初始化
    """
    global _index

    if _index is None:
        raise RuntimeError("索引未初始化，请先调用 init_chroma()")

    if filename in _loaded_files:
        print(f"⏭️ 文件已存在索引中，跳过: {filename}")
        return 0

    # 文本分块
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # 为文档添加文件名元数据
    for doc in documents:
        doc.metadata["source_file"] = filename

    # 分块并插入索引
    nodes = splitter.get_nodes_from_documents(documents)
    _index.insert_nodes(nodes)

    # 记录已上传文件
    _loaded_files.append(filename)
    _save_file_registry()

    print(f"✅ 已索引文件: {filename}（{len(nodes)} 个文本块）")
    return len(nodes)


# ── 检索引擎 ──────────────────────────────────────
def get_query_engine(
    similarity_top_k: Optional[int] = None,
    similarity_score_cutoff: Optional[float] = None,
) -> RetrieverQueryEngine:
    """
    获取查询引擎。

    Args:
        similarity_top_k: 返回 top-k 结果（默认 config.SIMILARITY_TOP_K）
        similarity_score_cutoff: 相似度阈值过滤

    Returns:
        RetrieverQueryEngine
    """
    if _index is None:
        raise RuntimeError("索引未初始化，请先调用 init_chroma()")

    top_k = similarity_top_k or config.SIMILARITY_TOP_K

    retriever = VectorIndexRetriever(
        index=_index,
        similarity_top_k=top_k,
    )

    postprocessors = []
    if similarity_score_cutoff is not None:
        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=similarity_score_cutoff)
        )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=postprocessors,
        llm=None,  # 让 Agent 自己走 LLM
    )

    return query_engine


# ── 工具函数 ──────────────────────────────────────
def list_uploaded_files() -> List[str]:
    """返回已上传的文件名列表"""
    global _loaded_files

    if not _loaded_files:
        _loaded_files = _load_file_registry()
    return list(_loaded_files)


def clear_knowledge_base():
    """
    清空知识库（删除所有向量 + 注册表）。
    需要重新 init_chroma() 才能使用。
    """
    global _chroma_client, _vector_store, _storage_context, _index, _loaded_files

    import shutil

    if os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)
        print("🗑️ 已清空 ChromaDB 数据")

    _chroma_client = None
    _vector_store = None
    _storage_context = None
    _index = None
    _loaded_files = []


def get_collection_stats() -> Dict:
    """返回集合统计信息"""
    if _chroma_client is None:
        return {"status": "not_initialized"}

    try:
        collection = _chroma_client.get_collection("knowledge_base")
        return {
            "total_vectors": collection.count(),
            "files_count": len(list_uploaded_files()),
            "files": list_uploaded_files(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
