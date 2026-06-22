"""
持久化记忆模块

基于 SQLite 实现跨会话的对话历史持久化 + 摘要压缩。
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole

import config


class PersistentChatMemory:
    """
    持久化对话记忆。

    职责：
    - 创建/切换/删除会话
    - 保存/加载消息历史
    - 自动摘要压缩（超出 token 阈值时）
    """

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._token_cache: Dict[str, int] = {}
        self._init_db()

    # ── 数据库连接管理 ────────────────────────────

    @contextmanager
    def _connect(self):
        """
        上下文管理器：自动管理 SQLite 连接生命周期。
        每次调用新建连接，用完自动关闭，天然线程安全。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """创建数据库表（如不存在）"""
        import os
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_name TEXT DEFAULT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_session_id
                    ON conversations(session_id);

                CREATE INDEX IF NOT EXISTS idx_session_created
                    ON conversations(session_id, created_at);
            """)

    # ── Token 估算工具 ────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """
        估算单条消息的 token 数（中文字符 ≈ 2 token，英文 ≈ 1 token）。
        """
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return chinese_chars * 2 + other_chars // 2

    def estimate_tokens(self, session_id: str) -> int:
        """
        估算会话消息的总 token 数（优先返回缓存值，仅在缓存未命中时全表扫描）。
        """
        if session_id in self._token_cache:
            return self._token_cache[session_id]

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM conversations "
                "WHERE session_id = ? AND role != 'system' "
                "ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()

            total = sum(self._count_tokens(row["content"]) for row in rows)
            self._token_cache[session_id] = total
            return total

    # ── 会话管理 ──────────────────────────────────

    def create_session(self, session_name: Optional[str] = None) -> str:
        """
        创建新会话。

        Args:
            session_name: 会话名称（可选，默认为空）

        Returns:
            str: 新会话 ID
        """
        session_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, session_name) "
                "VALUES (?, 'system', ?, ?)",
                (session_id, session_name or "", session_name),
            )
            conn.commit()
        return session_id

    def get_session_list(self) -> List[Dict]:
        """
        获取所有会话列表（按最新消息排序）。

        Returns:
            List[Dict]: 每个元素包含 session_id, session_name, message_count, last_active, first_message
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    c.session_id,
                    c.session_name,
                    COUNT(*) as message_count,
                    MAX(c.created_at) as last_active,
                    (
                        SELECT content FROM conversations
                        WHERE session_id = c.session_id
                          AND role = 'user'
                        ORDER BY created_at ASC LIMIT 1
                    ) as first_message
                FROM conversations c
                WHERE c.role != 'system'
                GROUP BY c.session_id
                ORDER BY last_active DESC
            """).fetchall()

            results = []
            for row in rows:
                name = row["session_name"]
                if not name and row["first_message"]:
                    name = row["first_message"][:config.SESSION_TITLE_PREFIX_LEN]
                    if len(row["first_message"]) > config.SESSION_TITLE_PREFIX_LEN:
                        name += "…"

                results.append({
                    "session_id": row["session_id"],
                    "session_name": name or "新会话",
                    "message_count": row["message_count"],
                    "last_active": row["last_active"],
                })
            return results

    def rename_session(self, session_id: str, new_name: str):
        """重命名会话"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET session_name = ? WHERE session_id = ?",
                (new_name, session_id),
            )
            conn.commit()

    def delete_session(self, session_id: str):
        """删除会话及其所有消息"""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM summaries WHERE session_id = ?", (session_id,)
            )
            conn.commit()
        # 清除对应的 token 缓存
        self._token_cache.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            return row is not None

    # ── 消息管理 ──────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """
        保存一条消息到数据库。

        Args:
            session_id: 会话 ID
            role: 'user' 或 'assistant'
            content: 消息内容
            metadata: 额外元数据（可选）
        """
        with self._connect() as conn:
            if role == "user":
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM conversations "
                    "WHERE session_id = ? AND role = 'user'",
                    (session_id,),
                ).fetchone()
                if row["cnt"] == 0:
                    title = content[:config.SESSION_TITLE_PREFIX_LEN]
                    if len(content) > config.SESSION_TITLE_PREFIX_LEN:
                        title += "…"
                    conn.execute(
                        "UPDATE conversations SET session_name = ? "
                        "WHERE session_id = ?",
                        (title, session_id),
                    )

            conn.execute(
                "INSERT INTO conversations (session_id, role, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            conn.commit()

        # 增量更新 token 缓存
        tokens = self._count_tokens(content)
        self._token_cache[session_id] = self._token_cache.get(session_id, 0) + tokens

    def save_messages_batch(
        self,
        session_id: str,
        messages: List[Tuple[str, str, Optional[Dict]]],
    ):
        """
        批量保存消息。

        Args:
            session_id: 会话 ID
            messages: [(role, content, metadata), ...]
        """
        with self._connect() as conn:
            for role, content, metadata in messages:
                conn.execute(
                    "INSERT INTO conversations (session_id, role, content, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
                )
            conn.commit()

        # 增量更新 token 缓存
        total_tokens = sum(self._count_tokens(content) for _, content, _ in messages)
        self._token_cache[session_id] = self._token_cache.get(session_id, 0) + total_tokens

    def load_history(self, session_id: str) -> List[ChatMessage]:
        """
        加载会话的历史消息（按时间顺序）。

        Args:
            session_id: 会话 ID

        Returns:
            List[ChatMessage]: 可用于 ChatMemoryBuffer 的消息列表
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations "
                "WHERE session_id = ? AND role != 'system' "
                "ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()

            messages = []
            for row in rows:
                role = MessageRole.USER if row["role"] == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=row["content"]))
            return messages

    def get_message_count(self, session_id: str) -> int:
        """获取会话的消息总数"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversations "
                "WHERE session_id = ? AND role != 'system'",
                (session_id,),
            ).fetchone()
            return row["cnt"] if row else 0

    # ── 摘要管理 ──────────────────────────────────

    def get_summary(self, session_id: str) -> Optional[str]:
        """获取最近一次摘要"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_text FROM summaries "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return row["summary_text"] if row else None

    def save_summary(self, session_id: str, summary_text: str, message_count: int, token_count: int):
        """保存摘要"""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (session_id, summary_text, message_count, token_count) "
                "VALUES (?, ?, ?, ?)",
                (session_id, summary_text, message_count, token_count),
            )
            conn.commit()

    def needs_compression(self, session_id: str) -> bool:
        """检查是否需要进行摘要压缩"""
        token_count = self.estimate_tokens(session_id)
        return token_count > config.MAX_HISTORY_TOKENS

    # ── 摘要压缩 ──────────────────────────────────

    def compress(self, session_id: str, llm=None) -> bool:
        """
        压缩会话历史：将旧消息摘要化。

        将历史消息（除最近 4 条外）合并生成摘要，替换数据库中的原始消息，
        从而减少 token 占用。

        Args:
            session_id: 会话 ID
            llm: 用于生成摘要的 LLM 实例（传 None 则不执行压缩）

        Returns:
            bool: 是否成功执行了压缩
        """
        history = self.load_history(session_id)
        if len(history) <= 4:
            return False

        if llm is None:
            return False

        to_compress = history[:-4]
        recent = history[-4:]

        text_to_summarize = "\n".join(
            f"{'用户' if m.role == MessageRole.USER else '助手'}: {m.content}"
            for m in to_compress
        )

        summary_prompt = (
            f"请对以下对话内容进行简洁的中文摘要，保留关键信息（用户问题、涉及的主题、重要结论）：\n\n"
            f"{text_to_summarize}\n\n摘要："
        )

        try:
            response = llm.complete(summary_prompt)
            summary = str(response).strip()

            self.save_summary(
                session_id=session_id,
                summary_text=summary,
                message_count=len(to_compress),
                token_count=self.estimate_tokens(session_id),
            )

            # 重建 SQLite 中的历史为摘要 + 最近消息
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "INSERT INTO conversations (session_id, role, content, session_name) "
                    "VALUES (?, 'system', ?, '')",
                    (session_id, f"[历史摘要] {summary}"),
                )
                for msg in recent:
                    role = "user" if msg.role == MessageRole.USER else "assistant"
                    conn.execute(
                        "INSERT INTO conversations (session_id, role, content) "
                        "VALUES (?, ?, ?)",
                        (session_id, role, msg.content),
                    )
                conn.commit()

            # 数据已变，缓存失效，清除
            self._token_cache.pop(session_id, None)
            return True

        except Exception as e:
            print(f"记忆压缩失败: {e}")
            return False
