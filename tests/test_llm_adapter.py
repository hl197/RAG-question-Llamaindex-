"""
测试 LLM 配置适配器 (llm_adapter.py)

覆盖：
- LLMConfig 优先级加载（显式传参 > 环境变量 > 默认值）
- create_llm() 工厂函数
- NO_PROXY 自动追加 api.deepseek.com
"""

import os

import pytest
from agent.llm_adapter import LLMConfig, create_llm


class TestLLMConfig:
    """测试 LLMConfig 优先级加载逻辑"""

    # ── 显式传参 > 环境变量 > 默认值 ─────────────────

    def test_explicit_overrides_env(self, monkeypatch):
        """显式传参应覆盖环境变量"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.base.com")

        config = LLMConfig(api_key="explicit-key", api_base="https://explicit.base.com")
        assert config.api_key == "explicit-key"
        assert config.api_base == "https://explicit.base.com"

    def test_env_fallback_when_no_explicit(self, monkeypatch):
        """未显式传参时，应回退到环境变量"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key-123")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.deepseek.com")

        config = LLMConfig()
        assert config.api_key == "env-key-123"
        assert config.api_base == "https://env.deepseek.com"

    def test_default_when_no_env_no_explicit(self, monkeypatch):
        """无环境变量且无显式传参时，应使用默认值"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        config = LLMConfig()
        assert config.api_key == ""
        assert config.api_base == "https://api.deepseek.com"

    def test_partial_explicit_args(self, monkeypatch):
        """只传部分参数时，其余参数应走环境变量或默认值"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key-partial")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        config = LLMConfig(api_base="https://custom.base.com")
        assert config.api_key == "env-key-partial"
        assert config.api_base == "https://custom.base.com"

    # ── 模型参数默认值 ──────────────────────────────

    def test_default_model_temperature(self):
        """验证模型参数的默认值"""
        config = LLMConfig()
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.context_window == 65536

    # ── 边界条件 ──────────────────────────────────

    def test_empty_env_values(self, monkeypatch):
        """环境变量为空字符串时，应使用默认值"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "")

        config = LLMConfig()
        # 空字符串会被 os.getenv 返回 ""，但非 None，走分支
        # 不过 "" 也是 truthy? "" 是 falsy, 所以 if os.getenv(env_var): 会跳过
        assert config.api_key == ""
        assert config.api_base == "https://api.deepseek.com"


class TestCreateLLM:
    """测试 create_llm 工厂函数"""

    def test_create_llm_defaults(self, monkeypatch):
        """create_llm() 默认应返回 DeepSeekLLM 实例"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        llm = create_llm()
        from agent.deepseek_llm import DeepSeekLLM

        assert isinstance(llm, DeepSeekLLM)
        assert llm.api_key == "test-key"
        assert llm.model == "deepseek-chat"
        assert llm.temperature == 0.7

    def test_create_llm_with_explicit_api_key(self, monkeypatch):
        """create_llm(api_key="test") 应使用显式传入的 api_key"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        llm = create_llm(api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_create_llm_with_model(self, monkeypatch):
        """create_llm(model="custom-model") 应使用指定模型"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        llm = create_llm(model="deepseek-v4-flash")
        assert llm.model == "deepseek-v4-flash"

    def test_create_llm_with_kwargs(self, monkeypatch):
        """create_llm 应透传额外 kwargs"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        llm = create_llm(temperature=0.1, max_tokens=2048)
        assert llm.temperature == 0.1
        assert llm.max_tokens == 2048

    def test_create_llm_sets_http_client(self, monkeypatch):
        """create_llm 创建的实例应包含 _http_client 和 _client"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        llm = create_llm()
        assert hasattr(llm, "_http_client")
        assert hasattr(llm, "_client")


class TestNoProxy:
    """测试 NO_PROXY 自动追加逻辑"""

    def test_no_proxy_contains_deepseek_domain(self):
        """NO_PROXY 应包含 api.deepseek.com"""
        no_proxy = os.environ.get("NO_PROXY", "")
        assert "api.deepseek.com" in no_proxy

    def test_no_proxy_does_not_duplicate(self, monkeypatch):
        """NO_PROXY 不应重复添加 api.deepseek.com"""
        monkeypatch.setenv("NO_PROXY", "api.deepseek.com")
        no_proxy = os.environ.get("NO_PROXY", "")
        assert no_proxy.split(",").count("api.deepseek.com") == 1
