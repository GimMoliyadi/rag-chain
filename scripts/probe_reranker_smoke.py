"""Reranker 真实权重 smoke test：先验一遍再写进 rag.py。

这一步存在的原因：`app/reranker.py` 里有两个参数我没法从文档确认，只能实测——
    1. `CrossEncoder.predict(max_length=...)` 这个 kwarg 会不会被接受
       （max_length 是构造参数，predict 走 **kwargs 转发，能不能通是另一回事）；
    2. 6GB 显存够不够同时装 bge-m3（fp16 约 1.18 GB）+ bge-reranker-v2-m3
       （fp16 约 1.12 GB）+ cross-encoder 在 1024 序列长度下的激活内存。

顺便回答真正关心的问题：5 道失败题在精排后期望文档有没有进 Top-5。
这个结果决定 reranker 值不值得进主线。

用法：python scripts/probe_reranker_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.embedding import get_embedding_client
from app.retriever import Retriever
from app.vectorstore import ChromaStore

# 这 5 道就是 baseline 里 recall_at_5 = 0 的题，精排的期望收益全在这里面。
PROBE_QUESTION_IDS = (11, 13, 18, 21, 22)
CANDIDATE_POOL = 50
FINAL_TOP_K = 5
BATCH_SIZE = 8
SEQUENCE_LIMIT = 1024


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _gpu_mib() -> float:
    return torch.cuda.memory_reserved("cuda") / 1024 / 1024


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings()
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(embedding=embedding, store=store, default_top_k=CANDIDATE_POOL)

    questions = json.loads(
        (Path(__file__).resolve().parent.parent / "eval" / "retrieval_questions.json").read_text(
            encoding="utf-8"
        )
    )
    targets = [q for q in questions if q.get("id") in PROBE_QUESTION_IDS]

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_before = _gpu_mib()
    print(f"精排模型加载前显存 reserved: {mem_before:.0f} MiB")

    t0 = time.perf_counter()
    from app.reranker import _load_cross_encoder

    reranker = _load_cross_encoder(settings.reranker_model, "cuda", "fp16")
    load_seconds = time.perf_counter() - t0
    mem_after_load = _gpu_mib()
    print(f"加载耗时 {load_seconds:.1f}s，加载后 reserved: {mem_after_load:.0f} MiB")

    # 第一条：只跑 2 对，验证 predict 的 kwarg 组合能不能被接受。
    warm_pairs = [("焦糖色要写进配料表吗", "焦糖色是食品添加剂，属于着色剂。")] * 2
    w0 = time.perf_counter()
    warm = reranker.predict(
        warm_pairs,
        batch_size=2,
        max_length=SEQUENCE_LIMIT,
        apply_softmax=False,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(f"predict 调用成功，返回类型={type(warm).__name__} 值={warm.tolist()}")

    print()
    print("=" * 100)
    for q in targets:
        tid = q["id"]
        question = q["question"]
        expected = [_norm(p) for p in q.get("expected_documents", [])]

        pool = retriever.search(question, top_k=CANDIDATE_POOL)
        pool = _dedup(pool)

        t0 = time.perf_counter()
        scores = list(reranker.predict(
            [(question, h.content) for h in pool],
            batch_size=BATCH_SIZE,
            max_length=SEQUENCE_LIMIT,
            apply_softmax=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        ))
        rerank_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)
        top = ranked[:FINAL_TOP_K]

        new_ranks = {exp: next((i + 1 for i, (h, _) in enumerate(ranked) if _norm(h.file_path) == exp), None)
                     for exp in expected}
        old_ranks = {exp: next((i + 1 for i, h in enumerate(pool) if _norm(h.file_path) == exp), None)
                     for exp in expected}
        in_top = any(r is not None and r <= FINAL_TOP_K for r in new_ranks.values())

        print(f"[{tid}] {q['category']}  重排耗时 {rerank_ms:.0f} ms（{len(pool)} 对，batch {BATCH_SIZE}）")
        print(f"     Q: {question}")
        for exp in expected:
            tag = "进 Top-5" if (old_ranks[exp] is not None and old_ranks[exp] <= FINAL_TOP_K) else "未进"
            new_tag = "进 Top-5" if (new_ranks[exp] is not None and new_ranks[exp] <= FINAL_TOP_K) else "未进"
            print(
                f"     期望 {exp}\n"
                f"       粗排 #{old_ranks[exp]} -> 精排 #{new_ranks[exp]}   优化前[{tag}] 优化后[{new_tag}]"
            )
        print("     精排后 Top-5:")
        for i, (h, s) in enumerate(top, start=1):
            mark = "  <== 期望文档" if _norm(h.file_path) in expected else ""
            print(f"       #{i} logit={s:>7.3f}  粗排分={h.score:.4f}  {h.title} / {h.heading}{mark}")
        print(f"     本题结论：{'精排救回来了' if in_top and not any(r is not None and r <= FINAL_TOP_K for r in old_ranks.values()) else ('本来就进得来' if in_top else '还是没救回来')}")
        print("-" * 100)

    print(f"\n显存峰值 reserved: {torch.cuda.max_memory_reserved('cuda') / 1024 / 1024:.0f} MiB"
          f"（显卡总量 6141 MiB）")
    return 0


def _dedup(hits):
    seen: set[int] = set()
    out = []
    for h in hits:
        if h.chunk_id in seen:
            continue
        seen.add(h.chunk_id)
        out.append(h)
    return out


if __name__ == "__main__":
    sys.exit(main())
