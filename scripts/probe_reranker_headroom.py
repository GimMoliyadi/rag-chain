"""回答一个决策问题：给这道题加 reranker，到底有没有机会？

reranker 只能在「候选池」里重新排序，它造不出候选池里没有的文档。
所以判断 reranker 值不值得做，先要知道：期望文档此刻在 Top-N 里的第几位。

- 如果期望文档已经在 Top-50 里（哪怕排第 40），reranker 有机会把它顶上来；
- 如果期望文档在 Top-50 之外，reranker 对它一分也救不了——
  问题出在向量检索本身，加 reranker 只是在错的答案池子里精排。

本脚本只跑检索，不调用任何 LLM，不写任何文件，只打印排名。
用法：HF_HUB_OFFLINE=1 python scripts/probe_reranker_headroom.py

注意：离线环境下必须设 HF_HUB_OFFLINE=1，否则模型加载阶段会先花几分钟
在 huggingface.co 的 HEAD 请求超时重试上（见 RETRIEVAL_BASELINE.md 第 8 节）。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import Settings  # noqa: E402
from app.embedding import get_embedding_client  # noqa: E402
from app.retriever import Retriever  # noqa: E402
from app.vectorstore import ChromaStore  # noqa: E402
from eval_retrieval import _norm_path, load_questions  # noqa: E402

FAIL_IDS = [11, 13, 18, 21, 22]
POOL_SIZE = 50


def pool_positions(hits, expected) -> list[tuple[str, int, float, int]]:
    """在候选池里找期望文档：返回 (路径, 最高排名, 该排名对应的分数, 该文档在池内的 chunk 数)。"""
    chunks: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(hits):
        chunks[_norm_path(h.file_path)].append(i + 1)
    out = []
    for path in expected:
        ranks = chunks.get(path)
        if not ranks:
            out.append((path, None, None, 0))
        else:
            top = min(ranks)
            score = next(h.score for i, h in enumerate(hits) if i + 1 == top)
            out.append((path, top, score, len(ranks)))
    return out


def main() -> int:
    settings = Settings()
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    retriever = Retriever(embedding=embedding, store=store, default_top_k=POOL_SIZE)

    questions = {q["id"]: q for q in load_questions(Path("eval/retrieval_questions.json"))}
    retriever.search("预热查询，不计入结果", top_k=POOL_SIZE)

    print(f"候选池深度：Top-{POOL_SIZE}   待查失败题：{FAIL_IDS}")
    print("=" * 100)
    reachable = unreachable = 0
    for qid in FAIL_IDS:
        q = questions[qid]
        hits = retriever.search(q["question"], top_k=POOL_SIZE)
        expected = [_norm_path(p) for p in q["expected_documents"]]
        print(f"[{qid}] {q['category']}  Q: {q['question'][:56]}")
        for path, rank, score, n in pool_positions(hits, expected):
            if rank is None:
                unreachable += 1
                print(f"      [MISS] 不在 Top-{POOL_SIZE} 内  <- reranker 无法触及")
            else:
                reachable += 1
                print(f"      [HIT ] 排名 #{rank:<3} 分数 {score:.4f}  该文档在池内有 {n} 个 chunk")
            print(f"        {path}")
        print("-" * 100)

    print(f"\n期望文档共 {reachable + unreachable} 篇：池内 {reachable} 篇，池外 {unreachable} 篇")
    print("池外的篇数 = reranker 无论如何都救不回的篇数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
