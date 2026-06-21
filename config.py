"""
全局配置文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 显式指定 .env 路径（避免工作目录不一致导致加载失败）
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

# ===== DeepSeek API =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ===== 模型配置 =====
LLM_MODEL = "deepseek-chat"
EMBED_DIM = 256  # 本地 Embedding 向量维度（纯 numpy，无需下载模型）

# ===== 分块参数 =====
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# ===== 检索参数 =====
SIMILARITY_TOP_K = 5

# ===== 数据目录 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
DB_PATH = os.path.join(DATA_DIR, "conversations.db")

# ===== 记忆参数 =====
MAX_HISTORY_TOKENS = 4000  # 触发摘要压缩的 token 阈值
SESSION_TITLE_PREFIX_LEN = 30  # 会话标题截取长度

# ===== 上传文件限制 =====
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"}

# ===== Gradio 配置 =====
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False  # 设为 True 可生成外网链接
