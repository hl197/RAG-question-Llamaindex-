"""共享测试 fixture"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.memory import PersistentChatMemory


@pytest.fixture
def memory():
    """
    使用临时 SQLite 文件测试记忆模块。

    注意：不能使用 ``:memory:``，因为 PersistentChatMemory 的每个方法
    都创建独立连接，接到 ``:memory:`` 会得到多个互不共享的数据库实例。
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    mem = PersistentChatMemory(db_path=db_path)
    yield mem
    os.unlink(db_path)
