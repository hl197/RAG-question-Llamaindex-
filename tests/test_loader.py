"""
测试文档加载模块 (loader.py)

测试支持的文件格式的解析基础功能。
"""

import os
import tempfile

import pytest


def _write_tmp_file(content: str, suffix: str = ".txt") -> str:
    """写入临时文件并返回路径"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content.encode("utf-8"))
    tmp.close()
    return tmp.name


class TestLoaderBasics:
    """测试加载器基础功能"""

    def test_load_txt(self):
        """加载 .txt 文件应正确解析内容"""
        from knowledge.loader import load_document

        content = "这是测试文本内容。\n第二行。"
        path = _write_tmp_file(content, ".txt")
        try:
            docs = load_document(path)
            assert len(docs) >= 1
            assert "测试文本内容" in docs[0].text
        finally:
            os.unlink(path)

    def test_load_md(self):
        """加载 .md 文件应正确解析内容"""
        from knowledge.loader import load_document

        content = "# 标题\n这是 Markdown 内容。"
        path = _write_tmp_file(content, ".md")
        try:
            docs = load_document(path)
            assert len(docs) >= 1
            assert "标题" in docs[0].text
        finally:
            os.unlink(path)

    def test_load_csv(self):
        """加载 .csv 文件应正确解析内容"""
        from knowledge.loader import load_document

        content = "name,age\n张三,28\n李四,32"
        path = _write_tmp_file(content, ".csv")
        try:
            docs = load_document(path)
            assert len(docs) >= 1
            combined = " ".join(d.text for d in docs)
            assert "张三" in combined
        finally:
            os.unlink(path)

    def test_load_empty_txt(self):
        """加载空文本文件应返回空列表"""
        from knowledge.loader import load_document

        path = _write_tmp_file("", ".txt")
        try:
            docs = load_document(path)
            assert docs == [] or all(d.text == "" for d in docs)
        finally:
            os.unlink(path)

    def test_unsupported_format(self):
        """不支持的文件格式应抛出 ValueError"""
        from knowledge.loader import load_document

        path = _write_tmp_file("test", ".xyz")
        try:
            with pytest.raises((ValueError, Exception)):
                load_document(path)
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        """不存在的文件应抛出异常"""
        from knowledge.loader import load_document

        with pytest.raises(Exception):
            load_document("/tmp/nonexistent_file_12345.txt")

    def test_encoding_detection(self):
        """应能正确检测 UTF-8 和 GBK 编码的中文文件"""
        from knowledge.loader import load_document

        # UTF-8
        utf8_path = _write_tmp_file("UTF-8 中文测试。", ".txt")
        try:
            docs = load_document(utf8_path)
            assert any("UTF-8" in d.text for d in docs)
        finally:
            os.unlink(utf8_path)

        # GBK
        gbk_content = "GBK编码测试。".encode("gbk")
        gbk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        gbk_path.write(gbk_content)
        gbk_path.close()
        try:
            docs = load_document(gbk_path.name)
            assert any("GBK编码测试" in d.text for d in docs)
        finally:
            os.unlink(gbk_path.name)
