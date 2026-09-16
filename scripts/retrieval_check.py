"""小规模真实检索验证：几百个真实 chunk + 独立临时索引，不碰正式索引。

为什么要单独有这个脚本：
    全量重建 14303 个 chunk 要二十分钟以上。如果 BGE-M3 在这个知识库上根本
    没有检索能力，那二十分钟就是白跑。先用 300 个 chunk 花几十秒验证，
    把「模型行不行」和「构建流程对不对」这两个问题分开：本脚本只回答前者。

和 smoke_real_embedding.py 的分工：
    smoke_real_embedding  验模型本身：维度、归一化、可复现性、语义区分度
    本脚本                验「模型 + 切块 + Chroma」串起来能不能检索到该找的东西

不写指纹（安全前提）：
    ChromaStore 的 _info_path 是 persist_dir.parent / embedding_index.json，
    临时目录 data/chroma_probe/ 的父目录正是 data/，所以它的指纹路径就是
    data/embedding_index.json —— 生产指纹。本脚本全程不调 write_index_info()，
    并在开头和结尾各快照一次正式索引状态，断言没被改动。这样小样本实验绝不
    可能污染全量重建的输入前提。

用法：python scripts/retrieval_check.py
退出码 0 = 每条查询的 Top-5 都命中目标文档，1 = 有未命中或状态被污染。
"""
from __future__ import annotations

import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.database import connect, get_chunks_for_reindex, init_db
from app.embedding import get_embedding_client
from app.ingest import embed_documents, report_truncated
from app.vectorstore import ChromaStore

SAMPLE_TARGET = 300       # 样本 chunk 数：够看出检索能力，构建耗时在几十秒内
PAGE_SIZE = 512           # 一页多少 chunk，和 ingest 保持一致
SEED = 42                 # 补齐用的随机抽样要可复现
TOP_K = 5
SUMMARY_LEN = 90          # content 摘要长度

# (查询, 目标文档标题必须包含的关键词, 这条查询为什么能算语义检索而不是关键词命中)
# 关键词只用于自动核对「该找到的那份文档有没有被检索到」；措辞说明是给人看的，
# 用来确认这次测试确实在验语义。
QUERIES: list[tuple[str, str, str]] = [
    (
        "为什么纳什均衡一定存在？存在性证明的关键工具是什么？",
        "纳什均衡存在性",
        "目标标题含「纳什均衡」「存在性」，查询改成反问 + 问证明工具",
    ),
    (
        "两家厂商同时决定产量而不是价格，这种竞争属于什么模型？",
        "Cournot竞争",
        "查询里只有「决定产量」这个行为描述，不出现模型名",
    ),
    (
        "费米子之间不能占据同一量子态的规律叫什么？",
        "Pauli 不相容原理",
        "查询用「不能占据同一量子态」描述现象，不出现原理名和人名",
    ),
    (
        "预包装食品的营养标签需要标示哪些项目？",
        "GB 28050",
        "查询用日常说法「营养标签」「标示」，不出现标准号",
    ),
    (
        "贝叶斯博弈如何通过类型划分处理不完全信息？",
        "Harsanyi-1967",
        "查询用方法描述，不出现作者名和年份",
    ),
]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def _summary(text: str, limit: int = SUMMARY_LEN) -> str:
    """把 chunk 正文压成一行摘要：去掉换行和多余空白再截断。"""
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _prod_snapshot(settings: Settings, fingerprint_path: Path) -> dict:
    """正式索引的状态快照，用来证明小样本实验没碰到它。"""
    chroma_db = settings.chroma_path_resolved / "chroma.sqlite3"
    return {
        "fingerprint_exists": fingerprint_path.exists(),
        "chroma_db_bytes": chroma_db.stat().st_size if chroma_db.exists() else 0,
    }


def _target_doc_ids(conn, queries) -> tuple[list[int], list[str]]:
    """按关键词找到目标文档的 id，返回 (ids, 没找到的关键词)。"""
    ids: list[int] = []
    missing: list[str] = []
    for _query, keyword, _note in queries:
        row = conn.execute(
            "SELECT id FROM documents WHERE title LIKE ? LIMIT 1", (f"%{keyword}%",)
        ).fetchone()
        if row is None:
            missing.append(keyword)
        else:
            ids.append(row["id"])
    return ids, missing


def _collect_rows(conn, target_ids: list[int], total: int):
    """取目标文档的全部 chunk，再用随机文档的 chunk 补齐到 total。

    目标 chunk 保证答案确实在索引里；补齐的随机 chunk 是干扰项，
    让检索不是「索引里就十几条、随便都能排到」。
    """
    all_rows = get_chunks_for_reindex(conn)
    keep = [r for r in all_rows if r["document_id"] in target_ids]
    target_count = len(keep)

    seen = {r["chunk_id"] for r in keep}
    rng = random.Random(SEED)
    rng.shuffle(all_rows)
    for row in all_rows:
        if len(keep) >= total:
            break
        if row["chunk_id"] not in seen:
            keep.append(row)

    keep.sort(key=lambda r: (r["document_id"], r["chunk_index"]))
    return keep, target_count, len({r["document_id"] for r in keep})


def main() -> int:
    failures: list[str] = []
    settings = Settings()
    settings.validate_providers()

    probe_dir = settings.chroma_path_resolved.parent / (
        settings.chroma_path_resolved.name + "_probe"
    )
    fingerprint_path = settings.chroma_path_resolved.parent / "embedding_index.json"

    before = _prod_snapshot(settings, fingerprint_path)
    say(f"正式索引 {settings.chroma_path_resolved}  : {before['chroma_db_bytes']} 字节")
    say(f"生产指纹 {fingerprint_path} 存在 = {before['fingerprint_exists']}"
        "（小样本实验必须保持这个状态不变）")
    say(f"临时索引 {probe_dir}（用完删除）")

    # ---- 阶段一：选样本。只查 SQLite，不加载模型，失败时一两秒就退 ----
    conn = connect(settings.db_path_resolved)
    init_db(conn)
    try:
        target_ids, missing = _target_doc_ids(conn, QUERIES)
        if missing:
            say("\n样本选取失败：以下目标文档在 SQLite 里找不到，先修正 QUERIES")
            for keyword in missing:
                say(f"  - 关键词 {keyword!r} 没匹配到任何文档标题")
            return 1
        rows, target_count, doc_count = _collect_rows(conn, target_ids, SAMPLE_TARGET)
    finally:
        conn.close()

    say(f"\n样本 = {len(rows)} 个 chunk，来自 {doc_count} 篇文档")
    say(f"        其中目标文档的 chunk {target_count} 个，随机干扰项 {len(rows) - target_count} 个")

    # ---- 阶段二：加载模型并建临时索引 ----
    started = time.perf_counter()
    embedding = get_embedding_client(settings)
    load_seconds = time.perf_counter() - started
    say(f"\n模型加载耗时 = {load_seconds:.1f}s  device={embedding.device_name}"
        f"  dtype={embedding.dtype_name}  max_length={embedding.max_length}")
    say(f"指纹 = {embedding.fingerprint}")

    shutil.rmtree(probe_dir, ignore_errors=True)
    store = ChromaStore(probe_dir)
    # 注意：store 全程不调 write_index_info()。它的 _info_path 落在
    # data/embedding_index.json，正是生产指纹路径——写一次就等于让小样本自称
    # 正式索引，会破坏全量重建的输入前提。

    try:
        say(f"\n开始写入临时索引（batch_size={settings.embedding_batch_size}）")
        started = time.perf_counter()
        truncated = 0
        for i in range(0, len(rows), PAGE_SIZE):
            page = rows[i : i + PAGE_SIZE]
            truncated += report_truncated(embedding, page)
            vectors = embed_documents(embedding, page)
            store.upsert_chunks([{**row, "embedding": vector} for row, vector in zip(page, vectors)])
        build_seconds = time.perf_counter() - started
        say(f"写入完成：{store.count()} / {len(rows)}  耗时 {build_seconds:.1f}s"
            f"（平均 {build_seconds / len(rows) * 1000:.0f} ms/条）")
        if truncated:
            say(f"截断：{truncated} 个 chunk 超过 {embedding.max_length} token，尾部未进入向量")

        # 索引要真的能应答，不只是计数对
        probe = embedding.embed_query(rows[0]["content"])
        top1 = store.query(probe, n_results=1)
        self_hit = bool(top1) and int(top1[0]["chunk_id"]) == int(rows[0]["chunk_id"])
        say(f"自查询校验：拿第 1 条 chunk 的内容去查，top-1 是自己 = {self_hit}"
            "（count 对但 HNSW 坏了时 count 照样对，所以必须发一条真查询）")
        if not self_hit:
            failures.append("临时索引自查询失败")

        # ---- 阶段三：真实 Top-K ----
        say("\n" + "=" * 78)
        say("真实 Top-K 检索")
        say("=" * 78)
        hits: list[bool] = []
        for index, (query, keyword, note) in enumerate(QUERIES, start=1):
            vector = embedding.embed_query(query)
            top = store.query(vector, n_results=TOP_K)
            hit = any(keyword in (r.get("title") or "") for r in top)
            hits.append(hit)

            say(f"\n[查询 {index}] {query}")
            say(f"        目标：标题含「{keyword}」")
            say(f"        措辞：{note}")
            for rank, row in enumerate(top, start=1):
                score = 1.0 - row["distance"]
                marked = "  <== 目标文档" if keyword in (row.get("title") or "") else ""
                say(f"        #{rank}  score={score:.4f}{marked}")
                say(f"            title    = {row.get('title', '')}")
                say(f"            heading  = {row.get('heading', '')}")
                say(f"            chunk_id = {row.get('chunk_id')}")
                say(f"            content  = {_summary(row.get('content', ''))}")
            say(f"        自动核对：{'命中' if hit else '未命中'} 目标文档")

        say("\n" + "-" * 78)
        say("自动核对汇总：" + " / ".join(
            f"查询{i}={'命中' if h else '未命中'}" for i, h in enumerate(hits, start=1)
        ))
        say("注意：这是机械核对，最终以人工判断检索质量为准。")

        # 显存峰值：本进程 torch 分配过的历史峰值，样本只有 300 条，
        # 这个数字的意义是确认 batch_size=8 在真实负载下没有逼近 6 GB 上限
        try:
            import torch

            peak = (
                f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MiB"
                if torch.cuda.is_available() else "n/a（CUDA 不可用）"
            )
        except ImportError:
            peak = "n/a（未安装 torch）"
        say(f"\n显存峰值 = {peak}")

        # ---- 阶段四：确认正式索引没被动过 ----
        after = _prod_snapshot(settings, fingerprint_path)
        if before != after:
            failures.append(f"正式索引状态被改动：{before} -> {after}")
        say(f"\n正式索引状态复核：{after}  {'未改动' if before == after else '被改动'}")

    finally:
        # 先关 handle 再谈删除。第一次写这个脚本时把 rmtree 放在 try 里、close 放在
        # finally，于是 rmtree 先跑——Windows 上 chroma.sqlite3 句柄还开着，目录删不掉，
        # 而 ignore_errors=True 把失败吞了，脚本照样打印「已删除」。和 rebuild_vectors
        # 修掉的是同一个坑，这里改成顺序上保证 close 先发生。
        store.close()

    if failures:
        say(f"\n结果：失败 {len(failures)} 项")
        for item in failures:
            say(f"  - {item}")
        say(f"临时索引保留在 {probe_dir} 供诊断。")
        return 1

    shutil.rmtree(probe_dir, ignore_errors=True)
    if probe_dir.exists():
        say(f"\n结果：检索全部通过，但临时索引 {probe_dir} 删除失败，请手动清理。")
        return 1

    say(f"\n结果：全部通过（{len(hits)}/{len(hits)} 条查询命中）")
    say(f"临时索引 {probe_dir} 已删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
