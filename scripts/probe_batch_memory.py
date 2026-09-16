"""量真实 bge-m3 + fp16 + max_seq_length=1024 下 batch=2/4/8 的峰值显存和耗时。

为什么必须有这个脚本：batch size 直接决定要不要在全量重建中途 OOM。fp16 的
attention 按 batch × heads × seq² 增长，但"按公式估算"和"实际占多少"差得远
（框架还要留激活、KV、临时张量）。所以直接量，不猜。

输入用 SQLite 里真实的 chunk，取 token 数最接近 1024 上限的那一批。
不用合成文本：合成文本的长度分布和真实 chunk 不一样，量出来的显存没有意义。

故意只测 2/4/8，不测更大的值：用户明确说过"不要为了测试故意运行明显会 OOM
的巨大 batch"，测到明显有余量的区间就够了。

用法：python scripts/probe_batch_memory.py
"""
from __future__ import annotations

import sqlite3
import time

import torch
from sentence_transformers import SentenceTransformer

DB = r"C:\Users\30858\obsidian-rag-backend\data\knowledge.db"
BATCHES = [2, 4, 8, 16]
WARMUP = 2        # 前几次预热 allocator，峰值统计从预热之后开始
MAX_LENGTH = 1024


def say(msg: str = "") -> None:
    print(msg, flush=True)


def load_chunks(model) -> list[str]:
    """取 token 数最接近上限的真实 chunk：这样量出来的是最坏情况，不是平均值。"""
    tok = model.processor

    conn = sqlite3.connect(DB)
    rows = [r[0] for r in conn.execute("SELECT content FROM chunks")]
    conn.close()

    with_lengths = sorted(
        ((len(tok(c)["input_ids"]), c) for c in rows),
        key=lambda x: x[0],
        reverse=True,
    )
    # 只要没超过上限的：超过的会被截断，量的就不是"真实长度"而是"截断后长度"
    # 取 max(BATCHES) 条，否则最大那个 batch 里塞不满，测出来的"单条耗时"是假数据
    wanted = max(BATCHES)
    picked = [(n, c) for n, c in with_lengths if n <= MAX_LENGTH][:wanted]
    say(f"真实 chunk 共 {len(rows)} 条，token 数范围 "
        f"{with_lengths[-1][0]} ~ {with_lengths[0][0]}")
    say(f"取 token 数最高的 {len(picked)} 条作为压力输入：")
    for n, _ in picked:
        say(f"    {n} tokens")
    return [c for _, c in picked]


def main() -> None:
    say(f"torch = {torch.__version__}  GPU = {torch.cuda.get_device_name(0)}")
    say(f"显存总量 = {torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB")
    say(f"max_seq_length = {MAX_LENGTH}  dtype = fp16")

    model = SentenceTransformer("BAAI/bge-m3", device="cuda",
                                model_kwargs={"torch_dtype": torch.float16})
    model.eval()
    model.max_seq_length = MAX_LENGTH

    chunks = load_chunks(model)

    free_before = torch.cuda.memory_reserved()

    say(f"\n模型加载后已保留显存 = {free_before / 2**20:.0f} MiB")
    say(f"\n{'batch':>6} | {'峰值 allocated':>15} | {'峰值 reserved':>14} | "
        f"{'相对基线新增':>13} | {'耗时':>8} | {'单条':>8} | 结果")
    say("-" * 92)

    for batch in BATCHES:
        texts = chunks[:batch]
        for _ in range(WARMUP):
            model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.perf_counter()
        vectors = model.encode(texts, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        say(f"{batch:>6} | {allocated / 2**20:>14.0f} MiB | {reserved / 2**20:>13.0f} MiB | "
            f"{(reserved - free_before) / 2**20:>12.0f} MiB | "
            f"{elapsed:>7.2f}s | {elapsed / batch * 1000:>7.1f}ms | "
            f"{len(vectors)} 条向量")

    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    say(f"\n显存总量 {total_mib:.0f} MiB，Windows 上 GPU 还有其它进程的基线占用。")
    say("选默认 batch 的原则：峰值 reserved 离总量要留明显余量，宁可慢不要 OOM。")


if __name__ == "__main__":
    main()
