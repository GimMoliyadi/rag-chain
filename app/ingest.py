"""入库流水线：Vault → SQLite → Chunk → Embedding → ChromaDB。

每一步只做一件事，按顺序串起来：

    遍历 .md 文件
      -> parser.parse_markdown            解析成结构化文档
      -> database.upsert_document         写入 SQLite，并判断内容是否变化
      -> 内容没变就直接跳过（增量）
      -> chunker.chunk_markdown           切块
      -> database.replace_chunks          切块结果存 SQLite（拿到 chunk 的 id）
      -> embedding.embed                  批量算向量
      -> store.delete_by_document         先清掉这个文档的旧向量
      -> store.upsert_chunks              写入新向量
      -> _remove_deleted                  收尾清掉"文件已删、库里还在"的幽灵文档

另有 rebuild_vectors()：换 embedding 模型时只重建 Chroma，
chunk 原文一个字都没变，不需要重新解析 Vault。
"""
from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass, field

from .chunker import chunk_markdown
from .config import Settings
from .database import (
    connect,
    delete_document,
    get_chunks,
    get_document,
    get_chunks_for_reindex,
    init_db,
    list_document_ids_by_path,
    replace_chunks,
    upsert_document,
)
from .embedding import BaseEmbedding
from .parser import ParsedDocument, iter_markdown_files, parse_markdown
from .vectorstore import ChromaStore

# 全量重建时一次在内存里保留多少 chunk 的向量。
# 每个向量是 1024 个 Python float，约 40 KB；全量 14303 条一次装完要 500 MB+，
# 这台机器空闲内存不够，所以分页处理，每页 512 条 ≈ 20 MB。
REBUILD_PAGE_SIZE = 512


@dataclass
class IngestStats:
    """一次 ingest 的结果统计，API 直接把这个返回给调用方。"""

    scanned: int = 0         # 扫到的 md 文件数
    added: int = 0           # 首次入库
    updated: int = 0         # 内容变化被更新
    unchanged: int = 0       # 内容没变被跳过（增量生效的证据）
    removed: int = 0         # 文件已从 Vault 删掉、库里被清掉的文档（删除同步）
    failed: int = 0          # 解析或写入失败
    chunks_written: int = 0
    truncated: int = 0       # 超过 embedding 模型 token 上限、被截尾的 chunk
    errors: list[str] = field(default_factory=list)


def run_ingest(
    settings: Settings,
    embedding: BaseEmbedding,
    store: ChromaStore,
    path_filter: str | None = None,
    limit: int | None = None,
) -> IngestStats:
    """带连接管理的 ingest 入口，CLI 和 API 都用这个。"""
    conn = connect(settings.db_path_resolved)
    init_db(conn)
    try:
        stats = ingest_vault(settings, embedding, store, conn, path_filter=path_filter, limit=limit)
        # 向量全部落地之后再写指纹，半成品的索引不会谎称自己是有效的
        store.write_index_info(embedding.fingerprint)
        return stats
    finally:
        conn.close()


def rebuild_vectors(settings: Settings, embedding: BaseEmbedding) -> int:
    """只用 SQLite 里的 chunk 重建 Chroma 向量索引，不重新解析 Vault。两阶段，正式索引最后才换。

    为什么不走 ingest_vault：换 embedding 模型时 chunk 原文一个字都没变，
    重新解析 911 个 Markdown 文件是纯浪费，还会白算一遍增量哈希。
    SQLite 是唯一的原文存档，从它读出来重算向量就够了。

    两阶段流程，核心纪律是"半成品绝不冒充正式索引"：

        阶段一  向量写进临时目录 data/chroma_building/，正式目录原样不动
        阶段二  数量校验 + 真实查询校验都通过，才用临时目录替换正式目录

    中途任何一步失败，data/chroma/ 保持原样（可能是上次留的半成品，可能带着
    上次的指纹），data/chroma_building/ 留给诊断。Retriever 的指纹检查是第二道
    防线：指纹不匹配就拒绝检索，所以临时目录就算被误当成正式目录，也不会返回
    错误结果，只会安全地报错。

    指纹最后才写：它是"这个索引有效"的凭证。目录换好了但指纹没写，结果是
    "安全地不可用"，而不是"错误地可用"。

    整个过程只读 SQLite，不改任何表。
    """
    conn = connect(settings.db_path_resolved)
    init_db(conn)
    try:
        rows = get_chunks_for_reindex(conn)
    finally:
        conn.close()          # 关掉连接再算向量，释放文件锁和内存

    total = len(rows)
    if total == 0:
        print("SQLite 里没有 chunk，跳过向量重建。")
        return 0

    prod_dir = settings.chroma_path_resolved
    build_dir = prod_dir.parent / (prod_dir.name + "_building")

    shutil.rmtree(build_dir, ignore_errors=True)   # 不继承上次失败留下的残留
    print(f"阶段一：把 {total} 个 chunk 写入临时索引 {build_dir}")
    print(f"        正式索引 {prod_dir} 保持不动，直到全量构建通过")
    build_store = ChromaStore(build_dir)
    # 注意：build_store 全程不调 write_index_info()。它的 _info_path 落在
    # data/embedding_index.json，正是生产指纹的路径——写一次就等于让半成品
    # 自称有效。指纹只由下面换完目录后的正式 store 来写。

    try:
        done = 0
        truncated = 0
        for page in _pages(rows, REBUILD_PAGE_SIZE):
            truncated += report_truncated(embedding, page)
            _write_page(build_store, embedding, page)
            done += len(page)
            print(f"\rEmbedding: {min(done, total)} / {total}", end="", flush=True)
        print()

        # 截断汇总放最后：逐条 warning 已经打过，这里给出总量，方便确认影响范围
        if truncated:
            print(f"截断汇总: {truncated} / {total} 条 chunk 超过 {embedding.max_length} token，"
                  f"这些 chunk 的尾部内容没有进入向量。"
                  f"后续优化项：token-aware chunk splitting（按 token 边界切块而不是按字符数）。")

        if not _verify_build(build_store, embedding, rows, total):
            return 0
    finally:
        # 关掉 SQLite handle：Windows 上文件被占用时目录无法 rename / delete
        build_store.close()

    del build_store
    gc.collect()

    print(f"阶段二：校验通过，用临时目录替换正式目录 {prod_dir}")
    shutil.rmtree(prod_dir, ignore_errors=True)
    build_dir.replace(prod_dir)

    prod_store = ChromaStore(prod_dir)
    try:
        prod_store.write_index_info(embedding.fingerprint)
        print(f"重建完成：{total} 个 chunk，指纹 {embedding.fingerprint}")
    finally:
        prod_store.close()
    return total


def _pages(rows: list[dict], page_size: int) -> list[list[dict]]:
    """把全部 chunk 行切成固定大小的页，逐页嵌入。

    分页只管"一次在内存里保留多少行向量"，和 GPU 显存无关——真正的显存上限是
    LocalSemanticEmbedding 的 batch_size（embed() 内部按它切片喂给模型）。
    """
    return [rows[start : start + page_size] for start in range(0, len(rows), page_size)]


def _write_page(store: ChromaStore, embedding: BaseEmbedding, page: list[dict]) -> None:
    """算一页的向量并写入。整页一次性交给 embed()，让它按 batch_size 内部分批。

    用新 dict 而不是就地写 row["embedding"]：就地写会把整页向量挂回 rows，
    循环结束后 14303 条 embedding 会全部残留在内存里（Python float 32 字节/个，
    约 470 MB）。新 dict 随本页结束就释放。
    """
    vectors = embed_documents(embedding, page)
    store.upsert_chunks([{**row, "embedding": vector} for row, vector in zip(page, vectors)])


def _verify_build(store: ChromaStore, embedding: BaseEmbedding, rows: list[dict], total: int) -> bool:
    """构建完再校验：数量必须等于 SQLite 的 chunk 数，且索引必须真的能应答查询。

    只比 count 不够——count 对但 HNSW 索引坏了时 count 照样是 14303。
    所以再发一条真实查询取回 chunk_id 核对，证明索引可应答而不只是计数对。
    """
    count = store.count()
    print(f"校验 count: {count} / {total}")
    if count != total:
        print("数量不符，临时索引不会替换正式索引。")
        return False

    probe = embed_documents(embedding, [rows[0]])[0]
    top = store.query(probe, n_results=1)
    if not top or int(top[0]["chunk_id"]) != int(rows[0]["chunk_id"]):
        print("自查询没能取回自己，索引不可应答，不替换正式索引。")
        return False

    print("数量校验与自查询校验都通过。")
    return True


def ingest_vault(
    settings: Settings,
    embedding: BaseEmbedding,
    store: ChromaStore,
    conn,
    path_filter: str | None = None,
    limit: int | None = None,
) -> IngestStats:
    """遍历 Vault 文件并逐个入库。单个文件失败不会中断整次 ingest。

    收尾再做一遍删除同步：库里记着、Vault 里已经不存在的路径要整篇清掉，
    否则被删的笔记会变成永远检索不到的"幽灵知识"。
    """
    stats = IngestStats()
    files = iter_markdown_files(settings.vault_path_resolved, path_filter=path_filter, limit=limit)
    stats.scanned = len(files)

    # 记录这次扫描在磁盘上真正看到的相对路径。必须在 try 之外加：
    # 一个文件读失败不代表它被删了，不能因此清掉它已有的知识。
    seen_paths: set[str] = set()
    for path in files:
        seen_paths.add(path.relative_to(settings.vault_path_resolved).as_posix())
        try:
            _ingest_one(settings, embedding, store, conn, path, stats)
        except Exception as exc:
            stats.failed += 1
            stats.errors.append(f"{path.as_posix()}: {exc}")

    _remove_deleted(store, conn, seen_paths, stats, full_scan=path_filter is None and limit is None)
    return stats


def _remove_deleted(store, conn, seen_paths, stats: IngestStats, full_scan: bool) -> None:
    """清掉"库里还在、Vault 里已经不存在"的文档，连同它的 chunk 和向量。

    full_scan=False 时整个跳过：--path / --limit 扫到的只是一部分文件，
    剩下那部分在库里完全正常，却被误判成"已删除"——一次 --path "法律" 就能删空全库。
    """
    if not full_scan:
        return

    for file_path, document_id in list_document_ids_by_path(conn).items():
        if file_path in seen_paths:
            continue
        try:
            # 向量先清：这一步失败时 SQLite 完全没动，下次重跑能原样重试
            store.delete_by_document(document_id)
            if delete_document(conn, document_id):
                stats.removed += 1
        except Exception as exc:
            stats.failed += 1
            stats.errors.append(f"{file_path}: {exc}")


def _ingest_one(settings, embedding, store, conn, path, stats: IngestStats) -> None:
    """处理单个 Markdown 文件的完整入库流程。"""
    doc = parse_markdown(path, settings.vault_path_resolved)
    document_id, changed, created = upsert_document(conn, doc)

    if not changed:
        stats.unchanged += 1
        return

    chunks = chunk_markdown(doc.content, doc.title, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        # 正文被清空、切不出任何块：上面的 upsert_document 已经把 SQLite 里的
        # 全文更新成空内容，但旧 chunk 和旧向量还挂着，照样能被检索命中——
        # 那才是真正的幽灵知识。所以这里必须把两边一起清掉。
        store.delete_by_document(document_id)
        replace_chunks(conn, document_id, [])
        if created:
            stats.added += 1
        else:
            stats.updated += 1
        return

    replace_chunks(conn, document_id, chunks)
    stats.chunks_written += len(chunks)

    rows = _build_vector_rows(conn, document_id)
    stats.truncated += report_truncated(embedding, rows)
    for row, vector in zip(rows, embed_documents(embedding, rows)):
        row["embedding"] = vector
    store.delete_by_document(document_id)
    store.upsert_chunks(rows)

    if created:
        stats.added += 1
    else:
        stats.updated += 1


def _build_vector_rows(conn, document_id: int) -> list[dict]:
    """把 SQLite 里的切块行补上文档元信息，得到 ChromaDB 需要的行。

    元信息从 documents 表读，而不是从 ParsedDocument 拿：重建向量索引时
    手头没有解析结果，两条路径统一从 SQLite 取，产出的行才完全一致。
    """
    doc = get_document(conn, document_id)
    rows = []
    for chunk in get_chunks(conn, document_id):
        rows.append(
            {
                "chunk_id": chunk["id"],
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "title": doc["title"],
                "file_path": doc["file_path"],
                "heading": chunk["heading"],
                "source": doc["source"],
                "folder": doc["folder"],
                "tags": doc["tags"],
            }
        )
    return rows


def embed_documents(embedding: BaseEmbedding, rows: list[dict]) -> list[list[float]]:
    """把 rows 里的 content 变成向量，返回等长的向量列表。

    这里只负责"行 → 文本"的映射，不做分批。分批在 provider 内部：
    LocalSemanticEmbedding.embed() 按 .env 的 EMBEDDING_BATCH_SIZE 自己切批次，
    控制的是显存上限。在这一层再切一遍是重复逻辑，而且省不了内存——
    返回的向量条数不管怎么切都一样多。
    """
    return embedding.embed([row["content"] for row in rows])


def report_truncated(embedding: BaseEmbedding, rows: list[dict]) -> int:
    """报告会被 embedding 模型截尾的 chunk，返回截尾数量。

    截尾是静默丢内容：chunk 的尾部原文留在 SQLite 里，但向量里没有它，
    检索到的答案可能缺最后一段。必须逐条点名（chunk_id / 文档 / 实际 token 数），
    让调用方能判断这次截断是不是可接受的。

    没有序列长度概念的 provider（哈希、远程 API）直接返回 0——它们不做截断。
    在算向量之前调用：先量长度，再算向量，避免把"发现截断"藏在 embedding 的副作用里。
    """
    limit = embedding.max_length
    lengths = embedding.count_tokens([row["content"] for row in rows])
    if limit is None or lengths is None:
        return 0

    count = 0
    for row, length in zip(rows, lengths):
        if length > limit:
            count += 1
            print(
                f"  截断 warning: chunk_id={row['chunk_id']}  "
                f"document={row['title']}  token_length={length} > {limit}",
                flush=True,
            )
    return count
