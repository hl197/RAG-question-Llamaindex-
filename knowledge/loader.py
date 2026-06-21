"""
多格式文档加载模块

支持解析 PDF / DOCX / PPTX / TXT / MD / CSV 文件，
统一输出为 List[llama_index.core.Document]。
"""

import os
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader


def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择解析器，返回 Document 列表。

    Args:
        file_path: 文件路径（绝对或相对路径）

    Returns:
        List[Document]: LlamaIndex Document 对象列表

    Raises:
        ValueError: 不支持的格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()

    # ── PDF → PyMuPDF4LLM（高保真 Markdown） ──
    if ext == ".pdf":
        return _load_pdf(path)

    # ── DOCX → LlamaIndex DocxReader ──
    elif ext == ".docx":
        return _load_docx(path)

    # ── PPTX → LlamaIndex PptxReader ──
    elif ext == ".pptx":
        return _load_pptx(path)

    # ── TXT / MD → 原生读取 ──
    elif ext in (".txt", ".md"):
        return _load_text(path)

    # ── CSV → LlamaIndex CSVReader ──
    elif ext == ".csv":
        return _load_csv(path)

    else:
        raise ValueError(
            f"不支持的文件格式: {ext}，支持的格式: "
            f".pdf .docx .pptx .txt .md .csv"
        )


def _load_pdf(path: Path) -> List[Document]:
    """使用 PyMuPDF4LLM 解析 PDF，保留表格/标题层级"""
    try:
        from pymupdf4llm.llama import LlamaMarkdownReader

        reader = LlamaMarkdownReader()
        docs = reader.load_data(str(path))
        _attach_source_metadata(docs, path, "pdf")
        return docs
    except ImportError as e:
        raise ImportError("请安装 pymupdf4llm: pip install pymupdf4llm") from e


def _load_docx(path: Path) -> List[Document]:
    """使用 LlamaIndex DocxReader 解析 DOCX"""
    try:
        from llama_index.readers.file import DocxReader

        reader = DocxReader()
        docs = reader.load_data(file=path)
        _attach_source_metadata(docs, path, "docx")
        return docs
    except ImportError as e:
        raise ImportError("请安装 python-docx: pip install python-docx") from e


def _load_pptx(path: Path) -> List[Document]:
    """使用 LlamaIndex PptxReader 解析 PPTX"""
    try:
        from llama_index.readers.file import PptxReader

        reader = PptxReader()
        docs = reader.load_data(file=path)
        _attach_source_metadata(docs, path, "pptx")
        return docs
    except ImportError as e:
        raise ImportError("请安装 python-pptx: pip install python-pptx") from e


def _load_text(path: Path) -> List[Document]:
    """读取纯文本 / Markdown 文件"""
    encoding = _detect_encoding(path)
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
    doc = Document(text=text)
    _attach_source_metadata([doc], path, path.suffix[1:])
    return [doc]


def _load_csv(path: Path) -> List[Document]:
    """使用 LlamaIndex CSVReader 解析 CSV"""
    try:
        from llama_index.readers.file import CSVReader

        reader = CSVReader()
        docs = reader.load_data(file=path)
        _attach_source_metadata(docs, path, "csv")
        return docs
    except ImportError as e:
        raise ImportError("请安装 llama-index-readers-file") from e


def _attach_source_metadata(docs: List[Document], path: Path, fmt: str) -> None:
    """为 Document 对象统一附加来源元数据"""
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("file_name", path.name)
        doc.metadata.setdefault("file_path", str(path.absolute()))
        doc.metadata.setdefault("file_type", fmt)
        doc.metadata.setdefault("page_label", doc.metadata.get("page_label", str(i + 1)))


def _detect_encoding(path: Path) -> str:
    """检测文件编码，优先 UTF-8 回退 GBK"""
    import chardet

    raw = path.read_bytes()
    result = chardet.detect(raw)
    return result.get("encoding", "utf-8") or "utf-8"


def get_supported_extensions() -> List[str]:
    """返回支持的扩展名列表"""
    return [".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"]
