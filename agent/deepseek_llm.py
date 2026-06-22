"""
DeepSeek LLM 封装 —— 兼容 LlamaIndex LLM 接口。

使用 OpenAI SDK 协议调用 DeepSeek API，绕开 llama-index OpenAI 包装类的模型名校验。
"""

import asyncio
import httpx
import random
import time
from typing import Any, Optional, Sequence, List, Generator

from llama_index.core.llms import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from openai import OpenAI as OpenAIClient


class DeepSeekLLM(LLM):
    """
    自定义 DeepSeek LLM。

    通过 OpenAI SDK 访问 DeepSeek API，绕开 llama-index 内置 OpenAI 类的模型名白名单校验。
    支持 deepseek-chat、deepseek-v4-flash 等任意模型名。
    """

    # Pydantic 字段声明（继承 BaseModel 需要显式标注类型）
    model: str = "deepseek-chat"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 65536

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        context_window: int = 65536,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key or "",
            api_base=api_base or "https://api.deepseek.com",
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            **kwargs,
        )
        # Pydantic 不允许直接 setattr 未声明的字段，用 object.__setattr__ 存客户端
        # 使用独立 httpx 客户端，显式禁用代理（HTTPTransport(proxy=None) 才能绕过
        # 系统 HTTP_PROXY/HTTPS_PROXY 环境变量，Client(mounts={}) 无效）
        object.__setattr__(self, "_http_client", httpx.Client(
            transport=httpx.HTTPTransport(proxy=None),
            follow_redirects=True,
        ))
        object.__setattr__(self, "_client", OpenAIClient(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=self._http_client,
        ))

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    # ── 指数退避重试 ─────────────────────────────

    def _call_with_retry(self, api_call_func, max_retries=3):
        """带指数退避的 API 调用"""
        last_error = None
        for attempt in range(max_retries):
            try:
                return api_call_func()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # 只在 429/503/服务端错误时重试
                if any(code in error_str for code in ['429', '503', 'rate', 'limit', 'too many', 'server error', 'governor']):
                    if attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ API 限速，{sleep_time:.1f}s 后重试 (第{attempt+1}次)...")
                        time.sleep(sleep_time)
                        continue
                    raise  # 非限速错误直接抛出
                raise
        raise last_error  # 重试用完仍失败

    # ── 核心接口 ──────────────────────────────────

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """对话生成（带指数退避重试）"""
        openai_messages = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content or ""
            openai_messages.append({"role": role, "content": content})

        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        reply = response.choices[0].message.content or ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=reply),
            raw=response,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """文本补全（包装为单轮对话，带指数退避重试）"""
        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        text = response.choices[0].message.content or ""
        return CompletionResponse(text=text, raw=response)

    # ── 流式接口（必要实现） ──────────────────────

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """流式对话生成（流式创建带指数退避重试）"""
        openai_messages = [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content or "",
            }
            for msg in messages
        ]

        stream = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        )

        return self._stream_chat_to_gen(stream)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """流式文本补全（流式创建带指数退避重试）"""
        stream = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        )

        return self._stream_complete_to_gen(stream)

    # ── 异步接口（LLM 抽象方法要求实现） ─────────

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """异步对话生成（委托同步版本）"""
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """异步文本补全"""
        return await asyncio.to_thread(self.complete, prompt, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """异步流式对话"""
        return self.stream_chat(messages, **kwargs)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        """异步流式补全"""
        return self.stream_complete(prompt, **kwargs)

    # ── 流式辅助 ──────────────────────────────────

    def _stream_chat_to_gen(self, stream) -> ChatResponseGen:
        """将 OpenAI 流式响应转成 ChatResponseGen"""
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_text += delta.content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=full_text,
                    ),
                    delta=delta.content,
                    raw=chunk,
                )

    def _stream_complete_to_gen(self, stream) -> CompletionResponseGen:
        """将 OpenAI 流式响应转成 CompletionResponseGen"""
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_text += delta.content
                yield CompletionResponse(text=full_text, delta=delta.content, raw=chunk)

    # ── 序列化 ──

    @classmethod
    def class_name(cls) -> str:
        return "DeepSeekLLM"
