#!/usr/bin/env python3
"""
RAGAS 评估 CLI 入口

用法:
    python evaluation/run_eval.py          # 运行完整评估
    python evaluation/run_eval.py --clean  # 仅清理 eval 数据
"""

# ⚠️ 必须在所有其他 import 之前设置，否则 huggingface_hub 会缓存错误配置
import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from evaluation.evaluator import RAGEvaluator


def main():
    clean_only = "--clean" in sys.argv

    evaluator = RAGEvaluator()

    try:
        if clean_only:
            evaluator.cleanup()
            return

        # 初始化并运行
        evaluator.setup()
        report = evaluator.run()
        evaluator.print_report(report)

    finally:
        # 默认清理（保留数据可加 --keep）
        if "--keep" not in sys.argv:
            evaluator.cleanup()


if __name__ == "__main__":
    main()
