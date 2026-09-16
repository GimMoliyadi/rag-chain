"""命令行入库入口。

用法：
    python ingest.py                        # 导入整个 Vault
    python ingest.py --path 01-精益创业2.0   # 只导入某个子目录
    python ingest.py --limit 20             # 只导入前 20 篇，快速试跑
    python ingest.py --rebuild-vectors      # 只重建 Chroma 向量（换 embedding 模型后用）
"""
from __future__ import annotations

import argparse
import time

from app.config import Settings
from app.embedding import get_embedding_client
from app.ingest import rebuild_vectors, run_ingest
from app.vectorstore import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="把 Obsidian Vault 导入 SQLite + ChromaDB")
    parser.add_argument("--path", help="只导入路径包含该片段的文件，用于按子目录试跑")
    parser.add_argument("--limit", type=int, help="最多导入多少篇，用于小规模试跑")
    parser.add_argument(
        "--rebuild-vectors",
        action="store_true",
        help="只用 SQLite 里的 chunk 重建 Chroma 向量，不重新解析 Vault",
    )
    args = parser.parse_args()

    settings = Settings()
    settings.validate_providers()

    # vault 在下面无条件打印，所以提前取出来。
    # 存在性校验只对完整入库有意义：重建向量索引不需要读 Vault。
    vault = settings.vault_path_resolved
    if not args.rebuild_vectors and not vault.is_dir():
        raise SystemExit(f"VAULT_PATH 不是有效目录：{vault}，请检查 .env 里的 VAULT_PATH")

    embedding = get_embedding_client(settings)

    print(f"Vault       : {vault}")
    print(f"数据库      : {settings.db_path_resolved}")
    print(f"向量库      : {settings.chroma_path_resolved}")
    print(f"Embedding   : {embedding.model_name}")
    print(f"chunk_size  : {settings.chunk_size}  overlap: {settings.chunk_overlap}")
    print("-" * 60)

    if args.rebuild_vectors:
        # 这里故意不先构造 ChromaStore：两阶段重建要删掉并替换正式目录，
        # 手上还握着它的 SQLite handle 时 rename 会失败。
        print(f"模式        : 仅重建向量索引（SQLite 只读，不重新解析 Vault）")
        print(f"batch_size  : {settings.embedding_batch_size}")
        started = time.perf_counter()
        count = rebuild_vectors(settings, embedding)
        elapsed = time.perf_counter() - started
        print("-" * 60)
        print(f"重建向量    : {count}")
        print(f"Embedding   : {embedding.model_name}")
        print(f"维度        : {embedding.dimensions}")
        print(f"设备        : {embedding.device_name}")
        print(f"batch_size  : {settings.embedding_batch_size}")
        print(f"耗时        : {elapsed:.1f} 秒"
              + (f"（平均 {elapsed / count:.2f} 秒/条）" if count else ""))
        print(f"显存峰值    : {_peak_gpu_mb()}")
        # 重建内部自己开过 store，这里重新开一个读指纹
        store = ChromaStore(settings.chroma_path_resolved)
        try:
            print("指纹        : " + str(store.read_index_info()))
        finally:
            store.close()
        return

    store = ChromaStore(settings.chroma_path_resolved)
    stats = run_ingest(settings, embedding, store, path_filter=args.path, limit=args.limit)

    print(f"扫描文件    : {stats.scanned}")
    print(f"新增文档    : {stats.added}")
    print(f"更新文档    : {stats.updated}")
    print(f"未变化跳过  : {stats.unchanged}   ← 增量生效，重复跑不会重复插库")
    print(f"写入 chunk  : {stats.chunks_written}")
    print(f"失败        : {stats.failed}")
    if stats.errors:
        print("\n失败详情：")
        for line in stats.errors[:10]:
            print(f"  - {line}")


def _peak_gpu_mb() -> str:
    """本进程 CUDA 显存峰值（MiB），取不到就如实说没有。

    max_memory_allocated 是本进程内 torch 分配过的历史峰值，重建期间正好是
    我们需要的那个数字。CPU 模式没有这个计数器。
    """
    try:
        import torch
    except ImportError:
        return "n/a（未安装 torch）"
    if not torch.cuda.is_available():
        return "n/a（CUDA 不可用）"
    return f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MiB"


if __name__ == "__main__":
    main()
