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
LLM_MAX_TOKENS = 8192  # 单次回答最大 token 数
LLM_CONTEXT_WINDOW = 65536
# Embedding 模型: "local" (纯numpy随机投影) 或 "semantic" (sentence-transformers语义模型)
EMBED_TYPE = "semantic"
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 384维，中英文
EMBED_DIM = 256  # 仅 LocalEmbedding 使用（SemanticEmbedding 由模型决定维度）
# 注意：修改此值后需清空 data/chroma/ 重新建索引，否则维度不匹配

# ===== 分块参数 =====
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# ===== 检索参数 =====
SIMILARITY_TOP_K = 8
SIMILARITY_CUTOFF = 0.3  # 相似度阈值，过滤低分噪音片段

# ===== Phase 3: 检索增强开关 =====
ENABLE_QUERY_DECOMPOSITION = True  # 复杂查询自动分解为子查询
ENABLE_HYBRID_RETRIEVAL = True     # 向量 + BM25 混合检索
ENABLE_RERANKING = True            # 检索后 LLM 精排

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

# ===== Token 计费 =====
# DeepSeek API 官方价格（deepseek-chat，2025年更新）
# 参考: https://api-docs.deepseek.com/quick_start/pricing
DEEPSEEK_INPUT_PRICE_PER_1K = 0.00027       # $0.27 / 1M input tokens
DEEPSEEK_OUTPUT_PRICE_PER_1K = 0.0011       # $1.10 / 1M output tokens

# ===== 日志系统 =====
import logging
import sys
from pathlib import Path as _Path

_LOG_FILE = _Path(__file__).parent / "data" / "app.log"


def setup_logging(level=logging.INFO):
    """配置全局日志系统（同时输出到控制台和文件）"""
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(str(_LOG_FILE), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # 降低第三方库日志噪音
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger"""
    return logging.getLogger(name)

# ===== 服务器配置 =====
SERVER_PORT = 7860  # HTTP 服务端口（FastAPI / Gradio 通用）
GRADIO_SERVER_PORT = SERVER_PORT  # 向后兼容（旧 app.py 仍可用）
GRADIO_SHARE = False  # 设为 True 可生成外网链接（仅 Gradio）
