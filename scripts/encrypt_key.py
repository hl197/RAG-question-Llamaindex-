#!/usr/bin/env python
"""
加密 API Key 工具。

用法:
    python scripts/encrypt_key.py                          # 交互式输入 Key
    python scripts/encrypt_key.py sk-xxx                   # 直接传 Key

原理：
    使用 Fernet 对称加密，将 API Key 加密后存入 config/keys.enc。
    解密口令由环境变量 KEY_PASSPHRASE 提供（不在仓库中）。
    加密 salt 写入 config/keys.salt（可提交到仓库，本身不泄露信息）。
"""

import os
import sys
import base64
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ── 路径 ──
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SALT_FILE = CONFIG_DIR / "keys.salt"
ENC_FILE = CONFIG_DIR / "keys.enc"


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """使用 PBKDF2 从口令派生 Fernet 密钥"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def encrypt(api_key: str, passphrase: str) -> bytes:
    """加密 API Key"""
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(api_key.encode("utf-8"))
    return salt + encrypted  # 拼接 salt + 密文


def decrypt(encrypted_data: bytes, passphrase: str) -> str:
    """解密 API Key"""
    salt = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    key = _derive_key(passphrase, salt)
    fernet = Fernet(key)
    return fernet.decrypt(ciphertext).decode("utf-8")


def main():
    # 获取 API Key
    if len(sys.argv) >= 2:
        api_key = sys.argv[1]
    else:
        # 尝试从 .env 读取
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=str(env_path))
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if api_key:
                print(f"📖 从 .env 读取到 DEEPSEEK_API_KEY")
        if not api_key:
            api_key = input("请输入要加密的 API Key: ").strip()

    if not api_key:
        print("❌ API Key 不能为空")
        sys.exit(1)

    # 获取口令
    passphrase = os.getenv("KEY_PASSPHRASE", "")
    if not passphrase:
        passphrase = input("请输入加密口令（将用于运行时解密，请牢记）: ").strip()
    if not passphrase:
        print("❌ 口令不能为空")
        sys.exit(1)

    # 加密
    encrypted_data = encrypt(api_key, passphrase)

    # 写入文件（salt 内嵌在 encrypted_data 前16字节）
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ENC_FILE.write_bytes(encrypted_data)

    print(f"✅ 已加密写入: {ENC_FILE}")
    print(f"   解密文件:   {ENC_FILE.name}（可提交到仓库）")
    print()
    print("⚠️ 请确保在运行环境中设置环境变量 KEY_PASSPHRASE，")
    print("   同时在 .gitignore 中排除 `.env` 和 `config/` 外的敏感文件。")
    print()
    print("   在 Git Bash 中:  export KEY_PASSPHRASE='你的口令'")
    print("   在 PowerShell:   $env:KEY_PASSPHRASE='你的口令'")


if __name__ == "__main__":
    main()
