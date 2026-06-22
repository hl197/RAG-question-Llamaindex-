"""
全局配置文件

API Key 加载优先级：
  1. 环境变量 DEEPSEEK_API_KEY（最高优先级）
  2. .env 文件（开发环境）
  3. config/keys.enc 加密文件（生产环境/仓库提交）
     - 需要环境变量 KEY_PASSPHRASE 提供解密口令
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# 添加 DeepSeek/OpenAI API 域名到 no_proxy，绕过系统代理直连
_existing_no_proxy = os.environ.get("NO_PROXY", "")
_domains = "api.deepseek.com,api.openai.com"
if _existing_no_proxy:
    os.environ["NO_PROXY"] = f"{_existing_no_proxy},{_domains}"
else:
    os.environ["NO_PROXY"] = _domains

# 显式指定 .env 路径（避免工作目录不一致导致加载失败）
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_env_path))


def _try_decrypt_key() -> str:
    """尝试从 config/keys.enc 解密 API Key"""
    enc_file = Path(__file__).parent / "config" / "keys.enc"
    passphrase = os.getenv("KEY_PASSPHRASE", "")

    if not enc_file.exists() or not passphrase:
        return ""

    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        encrypted_data = enc_file.read_bytes()
        salt = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))
        return Fernet(key).decrypt(ciphertext).decode("utf-8")
    except Exception:
        return ""


# ===== DeepSeek API =====
_try_env = os.getenv("DEEPSEEK_API_KEY") or ""
_try_enc = _try_decrypt_key()
DEEPSEEK_API_KEY = _try_env or _try_enc or ""
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
