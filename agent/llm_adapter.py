"""
LLM 配置与适配器

集中管理 LLM 配置加载，优先级：
  1. 显式传入的参数（最高优先级）
  2. 环境变量（DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL）
  3. .env 文件
  4. 代码内置默认值（最低优先级）
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import config as project_config  # 项目全局配置

# 自动加载项目根目录的 .env 文件
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

# 注: NO_PROXY 已由 config.py 统一设置，此处不重复

@dataclass
class LLMConfig:
    """LLM 配置容器"""

    # API 认证
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # 模型参数（优先使用 config.py 中的值）
    model: str = project_config.LLM_MODEL
    temperature: float = 0.7
    max_tokens: int = project_config.LLM_MAX_TOKENS
    context_window: int = project_config.LLM_CONTEXT_WINDOW

    # 环境变量映射：字段名 → (环境变量名, 默认值)
    _ENV_MAP = {
        "api_key": ("DEEPSEEK_API_KEY", ""),
        "api_base": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    }

    def __post_init__(self):
        """按优先级填充配置"""
        for field_name, (env_var, default) in self._ENV_MAP.items():
            existing = getattr(self, field_name)
            if existing is not None:
                continue  # 1. 显式传入
            env_val = os.getenv(env_var)
            if env_val:
                setattr(self, field_name, env_val)  # 2. 环境变量
                continue
            setattr(self, field_name, default)  # 3. 默认值


def create_llm(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> "DeepSeekLLM":
    """
    LLM 工厂函数。

    按优先级加载配置：显式参数 > 环境变量 > .env > 默认值。
    通常无需传参即可使用（自动从环境加载）。

    Args:
        api_key: DeepSeek API Key（不传则从环境变量 / .env 读取）
        api_base: API 地址（不传则默认 https://api.deepseek.com）
        model: 模型名（不传则默认 deepseek-chat）
        **kwargs: 其他参数传给 DeepSeekLLM

    Returns:
        DeepSeekLLM 实例
    """
    from agent.deepseek_llm import DeepSeekLLM

    # 只传非 None 的参数，让 LLMConfig 使用默认值
    config_kwargs = {}
    if api_key is not None:
        config_kwargs["api_key"] = api_key
    if api_base is not None:
        config_kwargs["api_base"] = api_base
    config = LLMConfig(**config_kwargs)

    # model 单独处理（不在 _ENV_MAP 中，需显式传入或使用默认值）
    if model is not None:
        config.model = model

    # 合并其他 kwargs 覆盖 config
    merged = {
        "model": config.model,
        "api_key": config.api_key,
        "api_base": config.api_base,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "context_window": config.context_window,
    }
    merged.update(kwargs)

    return DeepSeekLLM(**merged)
