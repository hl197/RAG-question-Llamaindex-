"""
测试持久化记忆模块 (memory.py)

使用 :memory: SQLite 测试 PersistentChatMemory：
- 会话 CRUD
- 消息保存/加载
- token 估算
- 摘要压缩
"""

import time

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

import config


class TestSessionManagement:
    """测试会话管理"""

    def test_create_session(self, memory):
        """创建会话应返回有效的 session_id"""
        session_id = memory.create_session()
        assert session_id is not None
        assert len(session_id) == 8

    def test_create_session_with_name(self, memory):
        """创建会话时可指定名称"""
        session_id = memory.create_session(session_name="测试会话")
        assert memory.session_exists(session_id)

    def test_session_exists(self, memory):
        """检查已存在的会话"""
        session_id = memory.create_session()
        assert memory.session_exists(session_id)

    def test_session_not_exists(self, memory):
        """检查不存在的会话"""
        assert not memory.session_exists("nonexistent")

    def test_delete_session(self, memory):
        """删除会话后应不再存在"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "你好")
        memory.delete_session(session_id)
        assert not memory.session_exists(session_id)

    def test_get_session_list_empty(self, memory):
        """无会话时 get_session_list 应返回空列表"""
        sessions = memory.get_session_list()
        assert sessions == []

    def test_get_session_list_with_messages(self, memory):
        """有消息的会话应在列表中"""
        s1 = memory.create_session()
        s2 = memory.create_session()
        memory.save_message(s1, "user", "第一条消息")
        memory.save_message(s2, "user", "第二条消息")

        sessions = memory.get_session_list()
        assert len(sessions) == 2

    def test_get_session_list_ordered(self, memory):
        """会话列表应按最后活跃时间降序排列"""
        s1 = memory.create_session()
        memory.save_message(s1, "user", "早的消息")
        time.sleep(1)  # 确保 timestamp 不同
        s2 = memory.create_session()
        memory.save_message(s2, "user", "晚的消息")

        sessions = memory.get_session_list()
        assert sessions[0]["session_id"] == s2  # 后创建的应该在前面

    def test_get_session_list_auto_title(self, memory):
        """会话名称应自动由第一条用户消息生成（超过 30 字时截断加…）"""
        session_id = memory.create_session()
        # 40 个字符，超过 SESSION_TITLE_PREFIX_LEN(30)，触发截断
        long_msg = "这是一个非常长的用户消息用于测试自动标题截取功能是否正常显示完毕"
        assert len(long_msg) > config.SESSION_TITLE_PREFIX_LEN
        memory.save_message(session_id, "user", long_msg)

        sessions = memory.get_session_list()
        assert sessions[0]["session_name"] == long_msg[:config.SESSION_TITLE_PREFIX_LEN] + "…"

    def test_get_session_list_fallback_title(self, memory):
        """无用户消息的会话应显示为"新会话" """
        session_id = memory.create_session()
        memory.save_message(session_id, "assistant", "回复")
        sessions = memory.get_session_list()
        assert sessions[0]["session_name"] == "新会话"

    def test_rename_session(self, memory):
        """重命名会话应生效（需在第一条用户消息发送之后重命名，否则 auto-title 会覆盖）"""
        session_id = memory.create_session()
        # 先发一条用户消息触发 auto-title
        memory.save_message(session_id, "user", "hello")
        # 再重命名覆盖自动生成的标题
        memory.rename_session(session_id, "新名称")
        sessions = memory.get_session_list()
        assert sessions[0]["session_name"] == "新名称"

    def test_delete_nonexistent_session(self, memory):
        """删除不存在的会话应不报错"""
        memory.delete_session("nonexistent")  # 不应抛出异常


class TestMessageManagement:
    """测试消息管理"""

    def test_save_and_load_message(self, memory):
        """保存消息后应能加载"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "你好")
        memory.save_message(session_id, "assistant", "你好！有什么可以帮助你的？")

        history = memory.load_history(session_id)
        assert len(history) == 2
        assert history[0].role == MessageRole.USER
        assert history[0].content == "你好"
        assert history[1].role == MessageRole.ASSISTANT
        assert history[1].content == "你好！有什么可以帮助你的？"

    def test_load_history_empty(self, memory):
        """空会话应返回空列表"""
        session_id = memory.create_session()
        history = memory.load_history(session_id)
        assert history == []

    def test_load_history_nonexistent(self, memory):
        """不存在的会话应返回空列表"""
        history = memory.load_history("nonexistent")
        assert history == []

    def test_get_message_count(self, memory):
        """消息计数应准确"""
        session_id = memory.create_session()
        assert memory.get_message_count(session_id) == 0

        memory.save_message(session_id, "user", "你好")
        assert memory.get_message_count(session_id) == 1

        memory.save_message(session_id, "assistant", "回复")
        assert memory.get_message_count(session_id) == 2

    def test_get_message_count_nonexistent(self, memory):
        """不存在的会话应返回 0"""
        assert memory.get_message_count("nonexistent") == 0

    def test_save_messages_batch(self, memory):
        """批量保存消息应全部成功写入"""
        session_id = memory.create_session()
        messages = [
            ("user", "问题1", None),
            ("assistant", "回答1", {"score": 0.95}),
            ("user", "问题2", None),
            ("assistant", "回答2", {"score": 0.87}),
        ]
        memory.save_messages_batch(session_id, messages)

        assert memory.get_message_count(session_id) == 4
        history = memory.load_history(session_id)
        assert len(history) == 4

    def test_load_history_order(self, memory):
        """消息应按时间正序排列"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "第一条")
        memory.save_message(session_id, "assistant", "回复1")
        memory.save_message(session_id, "user", "第二条")

        history = memory.load_history(session_id)
        assert history[0].content == "第一条"
        assert history[2].content == "第二条"

    def test_save_message_with_metadata(self, memory):
        """保存消息时可附带 metadata"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "查询", metadata={"source": "test", "score": 0.99})
        history = memory.load_history(session_id)
        assert history[0].content == "查询"

    def test_save_message_auto_title(self, memory):
        """第一条用户消息应自动生成会话标题（超过 30 字时截断加…）"""
        session_id = memory.create_session()
        # 35 个字符，超过 SESSION_TITLE_PREFIX_LEN(30)
        msg = "这是我的第一条用户消息用于测试自动标题截取显示功能是否正常完毕"
        assert len(msg) > config.SESSION_TITLE_PREFIX_LEN
        memory.save_message(session_id, "user", msg)
        sessions = memory.get_session_list()
        expected = msg[:config.SESSION_TITLE_PREFIX_LEN] + "…"
        assert sessions[0]["session_name"] == expected

    def test_message_count_excludes_system(self, memory):
        """get_message_count 不应计入 system 消息"""
        session_id = memory.create_session()  # 创建时插入了一条 system 消息
        assert memory.get_message_count(session_id) == 0


class TestTokenEstimation:
    """测试 token 估算"""

    def test_estimate_tokens_empty(self, memory):
        """空会话应返回 0 token"""
        session_id = memory.create_session()
        assert memory.estimate_tokens(session_id) == 0

    def test_estimate_tokens_chinese(self, memory):
        """中文字符应估算为 2 token/字"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "你好世界")  # 4 个汉字
        assert memory.estimate_tokens(session_id) == 8  # 4 * 2

    def test_estimate_tokens_english(self, memory):
        """英文字符应估算为约 0.5 token/字符"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "hello")  # 5 个英文字符
        assert memory.estimate_tokens(session_id) == 2  # 5 // 2 = 2

    def test_estimate_tokens_mixed(self, memory):
        """中英文混合应正确估算"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "你好world")
        assert memory.estimate_tokens(session_id) == 6  # 2*2 + 5//2 = 4 + 2 = 6

    def test_estimate_tokens_multiple_messages(self, memory):
        """多条消息的 token 应累加"""
        session_id = memory.create_session()
        memory.save_messages_batch(session_id, [
            ("user", "abc", None),
            ("assistant", "def", None),
        ])
        # 3//2 + 3//2 = 1 + 1 = 2
        assert memory.estimate_tokens(session_id) == 2


class TestSummaryCompression:
    """测试摘要压缩"""

    def test_save_and_get_summary(self, memory):
        """保存摘要后应能获取"""
        session_id = memory.create_session()
        memory.save_summary(session_id, "这是测试摘要", message_count=5, token_count=100)
        summary = memory.get_summary(session_id)
        assert summary == "这是测试摘要"

    def test_get_summary_none(self, memory):
        """无摘要时应返回 None"""
        session_id = memory.create_session()
        assert memory.get_summary(session_id) is None

    def test_get_summary_latest(self, memory):
        """多次保存摘要应返回最新的"""
        session_id = memory.create_session()
        memory.save_summary(session_id, "摘要1", 3, 50)
        time.sleep(1)  # 确保 timestamp 不同
        memory.save_summary(session_id, "摘要2", 6, 100)
        assert memory.get_summary(session_id) == "摘要2"

    def test_needs_compression_false(self, memory):
        """token 未超过阈值时不应压缩"""
        session_id = memory.create_session()
        memory.save_message(session_id, "user", "你好")
        assert not memory.needs_compression(session_id)

    def test_needs_compression_true(self, memory):
        """token 超过阈值时应压缩"""
        session_id = memory.create_session()
        big_text = "测试" * (config.MAX_HISTORY_TOKENS // 2 + 1)
        memory.save_message(session_id, "user", big_text)
        assert memory.needs_compression(session_id)

    def test_summary_after_delete_session(self, memory):
        """删除会话后摘要也应删除"""
        session_id = memory.create_session()
        memory.save_summary(session_id, "摘要", 3, 50)
        memory.delete_session(session_id)
        assert memory.get_summary(session_id) is None
