"""
测试 DeepSeek LLM 封装 (deepseek_llm.py)

Mock OpenAI 调用，不真实调用 API。
覆盖：chat, complete, 流式接口, 异步接口, metadata, class_name。
"""

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.base.llms.types import ChatMessage, MessageRole


@pytest.fixture
def llm():
    """返回一个 DeepSeekLLM 实例（不 mock 客户端，实际调用 chat 时才 mock）"""
    from agent.deepseek_llm import DeepSeekLLM

    return DeepSeekLLM(
        api_key="test-key",
        model="deepseek-chat",
        temperature=0.5,
    )


class TestBasicProperties:
    """测试基本属性和元数据"""

    def test_metadata(self, llm):
        """metadata 应正确返回模型信息"""
        metadata = llm.metadata
        assert metadata.model_name == "deepseek-chat"
        assert metadata.context_window == 65536
        assert metadata.num_output == 4096
        assert metadata.is_chat_model is True

    def test_class_name(self):
        """class_name 应返回 DeepSeekLLM"""
        from agent.deepseek_llm import DeepSeekLLM

        assert DeepSeekLLM.class_name() == "DeepSeekLLM"

    def test_init_with_custom_params(self):
        """初始化时可自定义所有参数"""
        from agent.deepseek_llm import DeepSeekLLM

        llm = DeepSeekLLM(
            model="deepseek-v4-flash",
            api_key="custom-key",
            api_base="https://custom.api.com",
            temperature=0.1,
            max_tokens=2048,
            context_window=32768,
        )
        assert llm.model == "deepseek-v4-flash"
        assert llm.api_key == "custom-key"
        assert llm.api_base == "https://custom.api.com"
        assert llm.temperature == 0.1
        assert llm.max_tokens == 2048
        assert llm.context_window == 32768

    def test_http_client_created(self, llm):
        """_http_client 和 _client 应在 __init__ 中创建"""
        assert hasattr(llm, "_http_client")
        assert hasattr(llm, "_client")

    def test_http_client_no_proxy(self, llm):
        """_http_client 应使用禁用代理的 HTTPTransport"""
        import httpx
        assert isinstance(llm._http_client, httpx.Client)
        assert isinstance(llm._http_client._transport, httpx.HTTPTransport)


class TestChat:
    """测试 chat 方法"""

    def test_chat_returns_chat_response(self, llm):
        """chat() 应返回 ChatResponse"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "你好！有什么可以帮助你的？"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            messages = [ChatMessage(role=MessageRole.USER, content="你好")]
            response = llm.chat(messages)

        assert response.message.role == MessageRole.ASSISTANT
        assert response.message.content == "你好！有什么可以帮助你的？"
        assert response.raw is mock_response

    def test_chat_sends_correct_params(self, llm):
        """chat() 应传递正确的参数给 OpenAI SDK"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "回复"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response) as mock_create:
            messages = [ChatMessage(role=MessageRole.USER, content="问题")]
            llm.chat(messages)

            mock_create.assert_called_once_with(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "问题"}],
                temperature=0.5,
                max_tokens=4096,
            )

    def test_chat_with_multiple_messages(self, llm):
        """chat() 应支持多轮对话"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "最终回复"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            messages = [
                ChatMessage(role=MessageRole.USER, content="第一轮"),
                ChatMessage(role=MessageRole.ASSISTANT, content="第一轮回复"),
                ChatMessage(role=MessageRole.USER, content="第二轮"),
            ]
            response = llm.chat(messages)

        assert response.message.content == "最终回复"

    def test_chat_empty_content(self, llm):
        """chat() 返回空内容时应处理"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            messages = [ChatMessage(role=MessageRole.USER, content="")]
            response = llm.chat(messages)

        assert response.message.content == ""


class TestComplete:
    """测试 complete 方法"""

    def test_complete_returns_completion_response(self, llm):
        """complete() 应返回 CompletionResponse"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "补全文本"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            response = llm.complete("请补全")

        assert response.text == "补全文本"
        assert response.raw is mock_response

    def test_complete_sends_correct_params(self, llm):
        """complete() 应传递正确的参数"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "补全"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response) as mock_create:
            llm.complete("请补全此句")

            mock_create.assert_called_once_with(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "请补全此句"}],
                temperature=0.5,
                max_tokens=4096,
            )

    def test_complete_empty_prompt(self, llm):
        """complete() 空 prompt 应正常工作"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            response = llm.complete("")

        assert response.text == ""


class TestStreamChat:
    """测试流式 chat"""

    def test_stream_chat_yields_chunks(self, llm):
        """stream_chat() 应逐块 yield ChatResponse（content 为累积文本，delta 为增量）"""
        chunks = []
        # deltas 是增量文本，content 会累积拼接
        for delta in ["你", "好", "！"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = delta
            chunks.append(chunk)

        end_chunk = MagicMock()
        end_chunk.choices = [MagicMock()]
        end_chunk.choices[0].delta.content = None
        chunks.append(end_chunk)

        with patch.object(llm._client.chat.completions, "create", return_value=iter(chunks)):
            messages = [ChatMessage(role=MessageRole.USER, content="你好")]
            responses = list(llm.stream_chat(messages))

        assert len(responses) == 3
        assert responses[0].message.content == "你"
        assert responses[0].delta == "你"
        assert responses[1].message.content == "你好"
        assert responses[1].delta == "好"
        assert responses[2].message.content == "你好！"
        assert responses[2].delta == "！"

    def test_stream_chat_empty_stream(self, llm):
        """stream_chat() 流中无内容时应不 yield"""
        end_chunk = MagicMock()
        end_chunk.choices = [MagicMock()]
        end_chunk.choices[0].delta.content = None

        with patch.object(llm._client.chat.completions, "create", return_value=iter([end_chunk])):
            messages = [ChatMessage(role=MessageRole.USER, content="")]
            responses = list(llm.stream_chat(messages))

        assert responses == []


class TestStreamComplete:
    """测试流式 complete"""

    def test_stream_complete_yields_chunks(self, llm):
        """stream_complete() 应逐块 yield CompletionResponse（text 为累积文本，delta 为增量）"""
        chunks = []
        for delta in ["补", "全", "完成"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = delta
            chunks.append(chunk)

        end_chunk = MagicMock()
        end_chunk.choices = [MagicMock()]
        end_chunk.choices[0].delta.content = None
        chunks.append(end_chunk)

        with patch.object(llm._client.chat.completions, "create", return_value=iter(chunks)):
            responses = list(llm.stream_complete("补"))

        assert len(responses) == 3
        assert responses[0].text == "补"
        assert responses[0].delta == "补"
        assert responses[1].text == "补全"
        assert responses[1].delta == "全"
        assert responses[2].text == "补全完成"
        assert responses[2].delta == "完成"


class TestAsyncMethods:
    """测试异步方法"""

    @pytest.mark.asyncio
    async def test_achat(self, llm):
        """achat() 应异步返回 ChatResponse"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "异步回复"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            messages = [ChatMessage(role=MessageRole.USER, content="异步问题")]
            response = await llm.achat(messages)

        assert response.message.content == "异步回复"
        assert response.message.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_acomplete(self, llm):
        """acomplete() 应异步返回 CompletionResponse"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "异步补全"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            response = await llm.acomplete("请补全")

        assert response.text == "异步补全"

    @pytest.mark.asyncio
    async def test_astream_chat(self, llm):
        """astream_chat() 应委托给同步 stream_chat"""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "流"
        end_chunk = MagicMock()
        end_chunk.choices = [MagicMock()]
        end_chunk.choices[0].delta.content = None

        with patch.object(llm._client.chat.completions, "create", return_value=iter([chunk, end_chunk])):
            messages = [ChatMessage(role=MessageRole.USER, content="")]
            # astream_chat 是 async def，返回 coroutine；await 后拿到同步 generator
            gen = await llm.astream_chat(messages)
            responses = list(gen)

        assert len(responses) == 1
        assert responses[0].message.content == "流"
        assert responses[0].delta == "流"

    @pytest.mark.asyncio
    async def test_astream_complete(self, llm):
        """astream_complete() 应委托给同步 stream_complete"""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "流补全"
        end_chunk = MagicMock()
        end_chunk.choices = [MagicMock()]
        end_chunk.choices[0].delta.content = None

        with patch.object(llm._client.chat.completions, "create", return_value=iter([chunk, end_chunk])):
            gen = await llm.astream_complete("")
            responses = list(gen)

        assert len(responses) == 1
        assert responses[0].text == "流补全"
        assert responses[0].delta == "流补全"

    @pytest.mark.asyncio
    async def test_achat_multiple_calls(self, llm):
        """多次异步调用应正常工作"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "回答"
        mock_response.choices = [mock_choice]

        with patch.object(llm._client.chat.completions, "create", return_value=mock_response):
            r1 = await llm.achat([ChatMessage(role=MessageRole.USER, content="问题1")])
            r2 = await llm.achat([ChatMessage(role=MessageRole.USER, content="问题2")])

        assert r1.message.content == "回答"
        assert r2.message.content == "回答"
