"""
DeepSeek LLM 封装 —— 兼容 LlamaIndex LLM 接口。

使用 OpenAI SDK 协议调用 DeepSeek API，绕开 llama-index OpenAI 包装类的模型名校验。
"""

import asyncio
import certifi
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

    # Token 用量累积（从 API 响应中提取）
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

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
            transport=httpx.HTTPTransport(proxy=None, verify=certifi.where()),
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

    # ── Token 用量查询 ──────────────────────────

    def get_token_usage(self) -> dict:
        """获取从启动到现在的累积 token 消耗"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

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

        # 提取 token 用量
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=reply),
            raw=response,
            additional_kwargs={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
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

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return CompletionResponse(
            text=text,
            raw=response,
            additional_kwargs={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

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
        last_usage = None
        for chunk in stream:
            if chunk.usage:
                last_usage = chunk.usage
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

        # 流结束后累积 token 用量（在最后一个 chunk 中返回）
        if last_usage:
            self.total_prompt_tokens += last_usage.prompt_tokens or 0
            self.total_completion_tokens += last_usage.completion_tokens or 0

    def _stream_complete_to_gen(self, stream) -> CompletionResponseGen:
        """将 OpenAI 流式响应转成 CompletionResponseGen"""
        full_text = ""
        last_usage = None
        for chunk in stream:
            if chunk.usage:
                last_usage = chunk.usage
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_text += delta.content
                yield CompletionResponse(text=full_text, delta=delta.content, raw=chunk)

        if last_usage:
            self.total_prompt_tokens += last_usage.prompt_tokens or 0
            self.total_completion_tokens += last_usage.completion_tokens or 0

    # ── 序列化 ──

    @classmethod
    def class_name(cls) -> str:
        return "DeepSeekLLM"
