"""
测试索引模块 (indexer.py)

使用临时目录测试 KnowledgeIndex 的 CRUD 操作。
注意：需要 conda 环境设置正确的 SSL_CERT_FILE，否则 httpx 初始化会失败。
"""

import os
import tempfile
import time

# 修正 SSL_CERT_FILE 环境变量，解决 conda 环境下 httpx SSL 报错
if "CONDA_HOME" in os.environ and not os.environ.get("SSL_CERT_FILE", ""):
    cert_path = os.path.join(os.environ["CONDA_HOME"], "Library", "ssl", "cacert.pem")
    if os.path.exists(cert_path):
        os.environ["SSL_CERT_FILE"] = cert_path

import pytest
import config
from llama_index.core import Document


def _force_remove(path: str, max_retries: int = 3):
    """Windows 上 ChromaDB 可能延迟释放文件锁，重试删除"""
    import shutil
    for attempt in range(max_retries):
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                raise


@pytest.fixture
def indexer():
    """创建临时 KnowledgeIndex 实例（使用 LocalEmbedding 避免下载）"""
    from llama_index.core import Settings
    from knowledge.local_embedding import LocalEmbedding
    from knowledge.indexer import KnowledgeIndex

    # 使用本地 embedding（零下载、零网络请求），避免 test 卡在模型下载或 SSL 问题
    Settings.embed_model = LocalEmbedding(dim=config.EMBED_DIM)

    tmp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    idx = KnowledgeIndex(chroma_dir=tmp_dir)
    idx.init_chroma()
    yield idx
    idx.close()
    _force_remove(tmp_dir)


class TestKnowledgeIndexInit:
    """测试索引初始化"""

    def test_not_initialized_by_default(self):
        """未调用 init_chroma 前 is_initialized 应为 False"""
        from knowledge.indexer import KnowledgeIndex

        idx = KnowledgeIndex(chroma_dir=tempfile.mkdtemp())
        assert not idx.is_initialized
        idx.close()

    def test_init_chroma(self, indexer):
        """init_chroma 后应处于已初始化状态"""
        assert indexer.is_initialized

    def test_init_chroma_idempotent(self, indexer):
        """多次调用 init_chroma 不应报错"""
        indexer.init_chroma()  # 第二次调用
        assert indexer.is_initialized


class TestFileRegistry:
    """测试文件注册表读写"""

    def test_empty_registry(self, indexer):
        """空索引时应返回空文件列表"""
        files = indexer.list_uploaded_files()
        assert files == []

    def test_after_add_file(self, indexer):
        """添加文件后注册表应包含该文件名"""
        docs = [Document(text="测试内容")]
        indexer.build_or_update_index(docs, "test.txt")
        files = indexer.list_uploaded_files()
        assert "test.txt" in files

    def test_add_duplicate_skipped(self, indexer):
        """重复添加同一文件应返回 0 节点"""
        docs = [Document(text="测试内容")]
        indexer.build_or_update_index(docs, "test.txt")
        n = indexer.build_or_update_index(docs, "test.txt")
        assert n == 0  # 跳过重复


class TestBuildAndRemove:
    """测试索引构建与删除"""

    def test_build_index_nodes(self, indexer):
        """构建索引后应返回正确的节点数"""
        docs = [Document(text="这是第一段测试内容。"), Document(text="这是第二段测试内容。")]
        n = indexer.build_or_update_index(docs, "doc.txt")
        assert n > 0
        assert indexer.is_initialized

    def test_remove_nonexistent_file(self, indexer):
        """删除不存在的文件不应报错"""
        # ChromaDB 的 delete 在无匹配记录时不会报错，返回 True
        result = indexer.remove_file("nonexistent.txt")
        # 只要不抛异常就行
        assert result is True or result is False

    def test_remove_existing_file(self, indexer):
        """删除已索引文件后注册表应不再包含该文件"""
        docs = [Document(text="测试内容")]
        indexer.build_or_update_index(docs, "test.txt")
        assert "test.txt" in indexer.list_uploaded_files()

        indexer.remove_file("test.txt")
        assert "test.txt" not in indexer.list_uploaded_files()

    def test_clear_knowledge(self, indexer):
        """清空知识库后索引应处于未初始化状态"""
        docs = [Document(text="测试内容")]
        indexer.build_or_update_index(docs, "test.txt")
        assert indexer.is_initialized

        indexer.clear_knowledge_base()
        assert not indexer.is_initialized


class TestQueryEngine:
    """测试查询引擎"""

    def test_get_query_engine_before_init(self):
        """未初始化时获取引擎应报错"""
        from knowledge.indexer import KnowledgeIndex

        idx = KnowledgeIndex(chroma_dir=tempfile.mkdtemp())
        with pytest.raises(RuntimeError, match="未初始化"):
            idx.get_query_engine()
        idx.close()

    def test_get_query_engine_after_init(self, indexer):
        """初始化后应能获取查询引擎"""
        engine = indexer.get_query_engine()
        assert engine is not None

    def test_get_node_texts_empty(self, indexer):
        """空索引应返回空列表"""
        texts = indexer.get_node_texts()
        assert texts == []


class TestCollectionStats:
    """测试集合统计"""

    def test_stats_empty(self, indexer):
        """空索引的统计应有 0 向量"""
        stats = indexer.get_collection_stats()
        assert stats["total_vectors"] == 0
        assert stats["files_count"] == 0

    def test_stats_after_add(self, indexer):
        """添加文件后统计应更新"""
        docs = [Document(text="测试内容")]
        indexer.build_or_update_index(docs, "test.txt")
        stats = indexer.get_collection_stats()
        assert stats["files_count"] == 1
        assert stats["total_vectors"] > 0
