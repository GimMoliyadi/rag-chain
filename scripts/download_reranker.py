"""下载 reranker 权重到本地 HuggingFace 缓存。

为什么单独写一个脚本而不是一行 `python -c`：
    命令行内联脚本在 Windows PowerShell 里会被引号和换行反复改写，
    之前两次尝试都是这样失败的（一次 exit 127，一次起了两个重复进程互锁）。
    写成文件后行为完全可复现。

下载后本仓库的所有脚本都设 HF_HUB_OFFLINE=1 跑，不再联网。
用法：python scripts/download_reranker.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# 本机唯一活着的外网代理端口，见 RETRIEVAL_BASELINE.md 第 8 节的通路排查。
PROXY = "http://127.0.0.1:9674"


def main() -> int:
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[name] = PROXY
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    t0 = time.perf_counter()
    path = snapshot_download(MODEL_NAME, max_workers=1)
    print(f"\n下载完成：{time.perf_counter() - t0:.0f}s")
    print(f"snapshot 路径：{path}")

    # 逐个文件报大小，确认大权重真的落盘了而不是只有 0 字节的占位符。
    for f in sorted(Path(path).iterdir(), key=lambda p: -p.stat().st_size):
        print(f"  {f.stat().st_size / 1024 / 1024:>8.2f} MB  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
