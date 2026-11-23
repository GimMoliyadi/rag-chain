"""增量 ingest 的同步测试：新增 / 修改 / 删除 / 跳过，以及"幽灵知识"的清除。

这些测试全部在 tmp_path 里造假 Vault + 假数据库 + 假向量库，
embedding 用离线的 local（哈希伪向量），不加载模型、不发网络请求。

为什么单独有一个测试文件：ingest 的正确性不在"能不能写进去"，
而在"磁盘变了之后，库里是不是跟着变"。尤其是删除——早期版本只遍历磁盘上
还存在的文件，删掉一篇 Markdown 会在 SQLite 和 Chroma 里留下删不掉的残留，
检索时照样能命中，那就是"幽灵知识"。这个文件把四条路径都锁住。

local provider 是确定性的哈希向量：同一段文本每次都得到同一个向量，
所以可以用"chunk 原文本身"当查询去确认它还在不在索引里（命中距离接近 0）。
这是本文件里所有向量断言能写得精确的原因。
"""
from __future__ import annotations

import errno
import os
from contextlib import closing
from pathlib import Path

import pytest

from app.database import connect
from app.embedding import HashingEmbedding, get_embedding_client
from app.ingest import VectorCleanupError, run_ingest
from app.retriever import EmbeddingMismatchError, Retriever
from app.vectorstore import ChromaStore

from conftest import CONTRACT_MD, MVP_MD

# 删掉最后一个小节后的版本，用来模拟"文档被改短"
CONTRACT_MD_SHORT = CONTRACT_MD.split("## 免责事由")[0]

# 只剩 frontmatter、正文被清空：这是最容易漏掉旧 chunk 的场景
EMPTY_MD = "---\ntitle: 空掉的文档\ntags: [法律]\n---\n"

# 哈希向量下"同一段文本"的余弦距离是 0，留一点浮点余量
SAME_TEXT_MAX_DISTANCE = 1e-3


def _ingest(settings, embedding, store, **kwargs):
    """跑一次完整 ingest。每次调用都自己开连接、用完就关，和 CLI 的用法一致。"""
    return run_ingest(settings, embedding, store, **kwargs)


def _sqlite(settings):
    """打开一条只用于断言的 SQLite 连接。测试结束要自己关。"""
    return connect(settings.db_path_resolved)


def _chunk_texts(conn, file_path: str) -> list[str]:
    """某个文档当前在 SQLite 里的全部 chunk 原文。文档不在库里就返回空列表。"""
    row = conn.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,)).fetchone()
    if row is None:
        return []
    return [
        r["content"]
        for r in conn.execute(
            "SELECT content FROM chunks WHERE document_id = ? ORDER BY chunk_index", (row["id"],)
        )
    ]


def _still_indexed(store, embedding, texts: list[str]) -> list[str]:
    """用 chunk 原文当查询去问向量库，返回还留在索引里的那些文本。"""
    hits = []
    for text in texts:
        top = store.query(embedding.embed_query(text), n_results=1)
        if top and top[0]["content"] == text and top[0]["distance"] <= SAME_TEXT_MAX_DISTANCE:
            hits.append(text)
    return hits


def test_first_ingest_adds_every_document(settings):
    """新增：两篇文档全部入库，向量条数等于两篇的 chunk 数之和。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        stats = _ingest(settings, embedding, store)
        assert stats.added == 2
        assert stats.removed == 0
        assert stats.failed == 0

        conn = _sqlite(settings)
        try:
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
            total_chunks = (
                len(_chunk_texts(conn, "法律/合同违约.md"))
                + len(_chunk_texts(conn, "创业/MVP.md"))
            )
            assert total_chunks == stats.chunks_written
            assert store.count() == total_chunks
        finally:
            conn.close()
    finally:
        store.close()


def test_second_ingest_skips_unchanged_documents(settings):
    """未变化跳过：不重新切块、不重算向量，也不误报成更新或删除。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        first = _ingest(settings, embedding, store)
        assert first.added == 2
        count_before = store.count()

        second = _ingest(settings, embedding, store)
        assert second.added == 0
        assert second.updated == 0
        assert second.unchanged == 2
        assert second.removed == 0
        assert second.chunks_written == 0
        assert store.count() == count_before
    finally:
        store.close()


def test_metadata_only_change_syncs_sqlite_and_chroma_without_reembedding(
    settings, fake_vault, monkeypatch
):
    """只改 frontmatter 时同步词面元数据，复用原向量。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    target = fake_vault / "法律" / "合同违约.md"
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            document_id = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?", ("法律/合同违约.md",)
            ).fetchone()["id"]
            old_hash = conn.execute(
                "SELECT content_hash FROM documents WHERE id = ?", (document_id,)
            ).fetchone()["content_hash"]
        finally:
            conn.close()
        old_vectors = store.get_document_chunks(document_id)
        old_count = store.count()

        target.write_text(
            CONTRACT_MD.replace("title: 合同违约的认定与救济", "title: 精准合同检索")
            .replace("  - 合同法", "  - 精准标签")
            .replace("source: 民法典笔记", "source: 更新后的民法典笔记"),
            encoding="utf-8",
        )

        embed_calls = 0

        def must_not_embed(texts):
            nonlocal embed_calls
            embed_calls += 1
            raise AssertionError("metadata-only ingest must not recompute embeddings")

        monkeypatch.setattr(embedding, "embed", must_not_embed)
        stats = _ingest(settings, embedding, store)

        assert stats.failed == 0
        assert stats.updated == 0
        assert stats.unchanged == 2
        assert stats.metadata_updated == 1
        assert stats.chunks_written == 0
        assert embed_calls == 0
        assert store.count() == old_count

        conn = _sqlite(settings)
        try:
            row = conn.execute(
                "SELECT title, tags, source, content_hash FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            assert tuple(row) == ("精准合同检索", "法律,精准标签", "更新后的民法典笔记", old_hash)
        finally:
            conn.close()

        new_vectors = store.get_document_chunks(document_id)
        assert [row["content"] for row in new_vectors] == [row["content"] for row in old_vectors]
        assert [row["embedding"] for row in new_vectors] == [row["embedding"] for row in old_vectors]
        assert [row["heading"] for row in new_vectors] == [row["heading"] for row in old_vectors]
        assert {row["title"] for row in new_vectors} == {"精准合同检索"}
        assert {row["tags"] for row in new_vectors} == {"法律,精准标签"}
        assert {row["source"] for row in new_vectors} == {"更新后的民法典笔记"}
    finally:
        store.close()


def test_metadata_only_sync_failure_rolls_back_and_retries_without_reembedding(
    settings, fake_vault, monkeypatch
):
    """Chroma 元数据写失败时两边保持旧版本，下一轮可重试。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    target = fake_vault / "法律" / "合同违约.md"
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            document_id = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?", ("法律/合同违约.md",)
            ).fetchone()["id"]
            old_metadata = tuple(
                conn.execute(
                    "SELECT title, tags, source FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            )
        finally:
            conn.close()
        old_vectors = store.get_document_chunks(document_id)
        target.write_text(
            CONTRACT_MD.replace("title: 合同违约的认定与救济", "title: 可重试的合同标题")
            .replace("source: 民法典笔记", "source: 可重试的新来源"),
            encoding="utf-8",
        )

        original_update = store.update_chunk_metadata
        failed_once = False
        embed_calls = 0

        def fail_once(rows):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("simulated metadata write failure")
            return original_update(rows)

        def count_only(texts):
            nonlocal embed_calls
            embed_calls += 1
            raise AssertionError("metadata-only retry must not recompute embeddings")

        monkeypatch.setattr(store, "update_chunk_metadata", fail_once)
        monkeypatch.setattr(embedding, "embed", count_only)
        failed = _ingest(settings, embedding, store)
        assert failed.failed == 1
        assert failed.metadata_updated == 0
        assert failed.updated == 0
        assert embed_calls == 0

        conn = _sqlite(settings)
        try:
            assert tuple(
                conn.execute(
                    "SELECT title, tags, source FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            ) == old_metadata
        finally:
            conn.close()
        assert store.get_document_chunks(document_id) == old_vectors

        monkeypatch.setattr(store, "update_chunk_metadata", original_update)
        retried = _ingest(settings, embedding, store)
        assert retried.failed == 0
        assert retried.metadata_updated == 1
        assert retried.updated == 0
        assert embed_calls == 0
        assert store.get_document_chunks(document_id)[0]["title"] == "可重试的合同标题"
    finally:
        store.close()


def test_metadata_only_title_change_updates_derived_heading_without_rechunking(
    settings, fake_vault, monkeypatch
):
    """无 Markdown 章节时 heading 来自文档标题，标题改动也要同步它。"""
    target = fake_vault / "无标题.md"
    initial = "---\ntitle: 初始文档标题\ntags: [测试]\n---\n\n这是一篇没有 Markdown 标题的正文。\n第二句仍属于同一个正文块。\n"
    target.write_text(initial, encoding="utf-8")
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            document_id = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?", ("无标题.md",)
            ).fetchone()["id"]
            old_chunk = conn.execute(
                """
                SELECT id, content, heading, start_position, end_position
                  FROM chunks WHERE document_id = ? ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchone()
        finally:
            conn.close()
        assert old_chunk["heading"] == "初始文档标题"
        old_vectors = store.get_document_chunks(document_id)
        assert len(old_vectors) == 1

        target.write_text(initial.replace("title: 初始文档标题", "title: 更新后的文档标题"), encoding="utf-8")
        embed_calls = 0

        def must_not_embed(texts):
            nonlocal embed_calls
            embed_calls += 1
            raise AssertionError("metadata-only title change must not recompute embeddings")

        monkeypatch.setattr(embedding, "embed", must_not_embed)
        stats = _ingest(settings, embedding, store)

        assert stats.failed == 0
        assert stats.metadata_updated == 1
        assert stats.updated == 0
        assert stats.chunks_written == 0
        assert embed_calls == 0

        conn = _sqlite(settings)
        try:
            new_chunk = conn.execute(
                """
                SELECT id, content, heading, start_position, end_position
                  FROM chunks WHERE document_id = ? ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchone()
        finally:
            conn.close()
        assert tuple(new_chunk) == (
            old_chunk["id"],
            old_chunk["content"],
            "更新后的文档标题",
            old_chunk["start_position"],
            old_chunk["end_position"],
        )

        new_vectors = store.get_document_chunks(document_id)
        assert [row["chunk_id"] for row in new_vectors] == [row["chunk_id"] for row in old_vectors]
        assert [row["content"] for row in new_vectors] == [row["content"] for row in old_vectors]
        assert [row["embedding"] for row in new_vectors] == [row["embedding"] for row in old_vectors]
        assert new_vectors[0]["title"] == "更新后的文档标题"
        assert new_vectors[0]["heading"] == "更新后的文档标题"
    finally:
        store.close()


def test_deleted_file_is_synced_out_of_sqlite_and_chroma(settings, fake_vault):
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            old_texts = _chunk_texts(conn, "法律/合同违约.md")
            old_doc_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()
        assert old_texts, "前置条件：被删的文档必须真的有 chunk"

        # 模拟"用户在 Obsidian 里删掉了这篇笔记"
        (fake_vault / "法律" / "合同违约.md").unlink()

        stats = _ingest(settings, embedding, store)
        assert stats.scanned == 1
        assert stats.removed == 1
        assert stats.failed == 0

        conn = _sqlite(settings)
        try:
            # 文档行和外键级联的 chunks 都必须消失
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == old_doc_chunks - len(old_texts)
            assert _chunk_texts(conn, "法律/合同违约.md") == []
        finally:
            conn.close()
        # 向量也必须消失，否则检索还能命中一篇已经不存在的笔记
        assert _still_indexed(store, embedding, old_texts) == []
    finally:
        store.close()


def test_shortened_document_leaves_no_stale_chunks(settings, fake_vault):
    """文档改短：被删掉的那部分内容不能再被检索到。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            before = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()
        assert len(before) > 1

        # 删掉最后一个小节，正文明显变短
        (fake_vault / "法律" / "合同违约.md").write_text(CONTRACT_MD_SHORT, encoding="utf-8")

        stats = _ingest(settings, embedding, store)
        assert stats.updated == 1
        assert stats.added == 0
        assert stats.removed == 0

        conn = _sqlite(settings)
        try:
            after = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()
        assert len(after) < len(before)
        # 旧切块里"不可抗力"那一整段已经不存在了
        assert not any("不可抗力" in t for t in after)
        # 向量库和 SQLite 必须完全一致：改短后没变的 chunk 继续可检索（开头的
        # 引言段内容原样保留，理应还在），被删掉的 chunk 一条都不许留。
        # 比"全部消失"更严，因为它同时锁住"该留的没被误删"。
        assert _still_indexed(store, embedding, before) == after
    finally:
        store.close()


def test_emptied_document_clears_its_chunks_and_vectors(settings, fake_vault):
    """正文被清空：SQLite 的全文已经变成空内容，旧 chunk 和旧向量必须一起清掉。

    这是最隐蔽的一条路径——文件还在，所以不算删除同步，
    但如果不处理，检索命中的是"这篇文档已经没内容"的旧片段。
    """
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            old_texts = _chunk_texts(conn, "创业/MVP.md")
        finally:
            conn.close()
        assert old_texts

        (fake_vault / "创业" / "MVP.md").write_text(EMPTY_MD, encoding="utf-8")

        stats = _ingest(settings, embedding, store)
        assert stats.updated == 1
        assert stats.removed == 0          # 文件还在，不能算成删除
        assert stats.failed == 0

        conn = _sqlite(settings)
        try:
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
            doc = conn.execute(
                "SELECT content FROM documents WHERE file_path = ?", ("创业/MVP.md",)
            ).fetchone()
            assert doc["content"] == ""
            assert _chunk_texts(conn, "创业/MVP.md") == []
        finally:
            conn.close()
        assert _still_indexed(store, embedding, old_texts) == []
    finally:
        store.close()


def test_delete_sync_is_skipped_under_path_filter(settings, fake_vault):
    """--path 只扫一部分文件时绝不能做删除同步，否则半库会被当成"已删除"。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        (fake_vault / "法律" / "合同违约.md").unlink()

        stats = _ingest(settings, embedding, store, path_filter="创业")
        assert stats.scanned == 1
        assert stats.removed == 0
        assert stats.unchanged == 1

        conn = _sqlite(settings)
        try:
            # 没在扫描范围内的文档必须原样保留
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
            assert _chunk_texts(conn, "法律/合同违约.md"), "带 --path 时误删了库里的文档"
        finally:
            conn.close()

        # 不带参数的全量扫描才应该清掉它
        assert _ingest(settings, embedding, store).removed == 1
    finally:
        store.close()


def test_delete_sync_is_skipped_under_limit(settings, fake_vault):
    """--limit 同样让扫描结果变得不完整，删除同步也得跟着关。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        (fake_vault / "法律" / "合同违约.md").unlink()

        stats = _ingest(settings, embedding, store, limit=1)
        assert stats.removed == 0

        conn = _sqlite(settings)
        try:
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        finally:
            conn.close()
    finally:
        store.close()


def test_failed_file_is_not_treated_as_deleted(settings, fake_vault, monkeypatch):
    """读失败的文件不等于被删了：它已有的知识不能因此被清掉。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            old_texts = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()

        import app.ingest as ingest_module
        original_parse = ingest_module.parse_markdown

        def boom(path, vault_path):
            if "合同违约" in path.as_posix():
                raise OSError("模拟读取失败")
            return original_parse(path, vault_path)

        monkeypatch.setattr(ingest_module, "parse_markdown", boom)

        stats = _ingest(settings, embedding, store)
        assert stats.failed == 1
        assert stats.removed == 0
        assert "合同违约" in stats.errors[0]

        conn = _sqlite(settings)
        try:
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        finally:
            conn.close()
        # 向量还在，说明没有误判成删除
        assert _still_indexed(store, embedding, old_texts) == old_texts
    finally:
        store.close()


def test_directory_scan_error_skips_removal_until_a_complete_scan(settings, fake_vault, monkeypatch):
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    inaccessible = fake_vault / "法律"
    existing_path = "法律/合同违约.md"
    deleted_path = "创业/MVP.md"
    scandir = os.scandir

    def fail_on_directory(path):
        if Path(path) == inaccessible:
            raise PermissionError(errno.EACCES, "测试目录无法读取", str(path))
        return scandir(path)

    try:
        _ingest(settings, embedding, store)
        with closing(_sqlite(settings)) as conn:
            existing_chunks = _chunk_texts(conn, existing_path)
            deleted_chunks = _chunk_texts(conn, deleted_path)
        assert existing_chunks and deleted_chunks
        (fake_vault / deleted_path).unlink()
        (fake_vault / "新增.md").write_text("# 新增笔记\n\n正常处理可读文件。", encoding="utf-8")

        with monkeypatch.context() as patch:
            patch.setattr(os, "scandir", fail_on_directory)
            stats = _ingest(settings, embedding, store)

        assert (stats.scan_failures, stats.failed, stats.added, stats.removed) == (1, 1, 1, 0)
        assert str(inaccessible) in stats.errors[0]
        with closing(_sqlite(settings)) as conn:
            assert _chunk_texts(conn, existing_path) == existing_chunks
            assert _chunk_texts(conn, deleted_path) == deleted_chunks
        assert _still_indexed(store, embedding, existing_chunks + deleted_chunks) == existing_chunks + deleted_chunks

        stats = _ingest(settings, embedding, store)

        assert (stats.scan_failures, stats.failed, stats.removed) == (0, 0, 1)
        with closing(_sqlite(settings)) as conn:
            assert _chunk_texts(conn, existing_path) == existing_chunks
            assert _chunk_texts(conn, deleted_path) == []
        assert _still_indexed(store, embedding, existing_chunks) == existing_chunks
        assert _still_indexed(store, embedding, deleted_chunks) == []
    finally:
        store.close()


def test_embedding_failure_rolls_back_document_for_retry(settings, fake_vault, monkeypatch):
    """向量计算失败时保留旧 SQLite/向量，下一次扫描仍能重试该文档。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    target = fake_vault / "法律" / "合同违约.md"
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            old_content = conn.execute(
                "SELECT content FROM documents WHERE file_path = ?", ("法律/合同违约.md",)
            ).fetchone()["content"]
            old_chunks = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()
        old_vector_count = store.count()

        target.write_text(CONTRACT_MD + "\n\nretryable addition.", encoding="utf-8")
        original_embed = embedding.embed

        def fail_embedding(texts):
            return []

        monkeypatch.setattr(embedding, "embed", fail_embedding)
        failed = _ingest(settings, embedding, store)
        assert failed.failed == 1
        assert failed.updated == 0
        assert failed.chunks_written == 0
        assert "Embedding returned 0 vectors" in failed.errors[0]

        conn = _sqlite(settings)
        try:
            row = conn.execute(
                "SELECT content FROM documents WHERE file_path = ?", ("法律/合同违约.md",)
            ).fetchone()
            assert row["content"] == old_content
            assert _chunk_texts(conn, "法律/合同违约.md") == old_chunks
        finally:
            conn.close()
        assert store.count() == old_vector_count

        monkeypatch.setattr(embedding, "embed", original_embed)
        retried = _ingest(settings, embedding, store)
        assert retried.failed == 0
        assert retried.updated == 1
        conn = _sqlite(settings)
        try:
            new_chunks = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()
        assert _still_indexed(store, embedding, new_chunks) == new_chunks
    finally:
        store.close()


def test_partial_vector_write_is_cleaned_and_retryable(settings, fake_vault, monkeypatch):
    """部分写入失败后，旧版本仍可检索，新版本可以重试。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    target = fake_vault / "法律" / "合同违约.md"
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            old_chunks = _chunk_texts(conn, "法律/合同违约.md")
            other_chunks = _chunk_texts(conn, "创业/MVP.md")
        finally:
            conn.close()
        target.write_text(CONTRACT_MD + "\n\npartial write retry.", encoding="utf-8")

        original_upsert = store.upsert_chunks
        failed_once = False

        def partial_upsert(rows):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                original_upsert(rows[:1])
                raise RuntimeError("simulated Chroma write failure")
            original_upsert(rows)

        monkeypatch.setattr(store, "upsert_chunks", partial_upsert)
        failed = _ingest(settings, embedding, store)
        assert failed.failed == 1
        assert _still_indexed(store, embedding, old_chunks) == old_chunks
        assert _still_indexed(store, embedding, other_chunks) == other_chunks

        monkeypatch.setattr(store, "upsert_chunks", original_upsert)
        retried = _ingest(settings, embedding, store)
        assert retried.failed == 0
        assert retried.updated == 1
        conn = _sqlite(settings)
        try:
            new_chunks = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()
        assert _still_indexed(store, embedding, new_chunks) == new_chunks
    finally:
        store.close()


def test_vector_cleanup_failure_aborts_batch_before_next_document(settings, fake_vault, monkeypatch):
    """清理失败时中止整批 ingest，避免回滚后的文档 id 被后续文件复用。"""
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    target = fake_vault / "法律" / "合同违约.md"
    other = fake_vault / "创业" / "MVP.md"
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            old_rows = {
                row["file_path"]: row["content"]
                for row in conn.execute("SELECT file_path, content FROM documents")
            }
        finally:
            conn.close()
        target.write_text(CONTRACT_MD + "\n\ncleanup failure.", encoding="utf-8")
        other.write_text(MVP_MD + "\n\nshould not be processed.", encoding="utf-8")

        original_upsert = store.upsert_chunks
        original_delete = store.delete_by_document
        delete_calls = 0

        def partial_upsert(rows):
            original_upsert(rows[:1])
            raise RuntimeError("simulated Chroma write failure")

        def fail_cleanup(document_id):
            nonlocal delete_calls
            delete_calls += 1
            if delete_calls == 2:
                raise RuntimeError("simulated cleanup failure")
            return original_delete(document_id)

        monkeypatch.setattr(store, "upsert_chunks", partial_upsert)
        monkeypatch.setattr(store, "delete_by_document", fail_cleanup)
        with pytest.raises(VectorCleanupError):
            _ingest(settings, embedding, store)
        assert delete_calls == 2
        assert store.read_index_info() is None
        with pytest.raises(EmbeddingMismatchError):
            Retriever(embedding, store).search("合同违约")

        conn = _sqlite(settings)
        try:
            rows = {
                row["file_path"]: row["content"]
                for row in conn.execute("SELECT file_path, content FROM documents")
            }
        finally:
            conn.close()
        assert rows == old_rows
    finally:
        store.close()


def test_embedding_mismatch_refuses_ingest_without_changes(settings):
    """更换 embedding 模型时，增量 ingest 必须先拒绝，不能覆盖指纹或混入向量。"""
    old_embedding = HashingEmbedding(256)
    new_embedding = HashingEmbedding(128)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, old_embedding, store)
        before_info = store.read_index_info()
        before_count = store.count()
        conn = _sqlite(settings)
        try:
            before_docs = [
                tuple(row)
                for row in conn.execute(
                    "SELECT file_path, content, content_hash FROM documents ORDER BY id"
                )
            ]
            before_chunks = [
                tuple(row)
                for row in conn.execute(
                    "SELECT document_id, chunk_index, content FROM chunks ORDER BY id"
                )
            ]
        finally:
            conn.close()

        with pytest.raises(EmbeddingMismatchError):
            _ingest(settings, new_embedding, store)

        assert store.read_index_info() == before_info
        assert store.count() == before_count
        conn = _sqlite(settings)
        try:
            assert [
                tuple(row)
                for row in conn.execute(
                    "SELECT file_path, content, content_hash FROM documents ORDER BY id"
                )
            ] == before_docs
            assert [
                tuple(row)
                for row in conn.execute(
                    "SELECT document_id, chunk_index, content FROM chunks ORDER BY id"
                )
            ] == before_chunks
        finally:
            conn.close()
    finally:
        store.close()


@pytest.mark.parametrize("change", ["update", "empty", "delete"])
def test_sqlite_commit_failure_restores_vectors(settings, fake_vault, monkeypatch, change):
    import app.ingest as ingest_module

    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    target = fake_vault / "法律" / "合同违约.md"
    try:
        _ingest(settings, embedding, store)
        conn = _sqlite(settings)
        try:
            document_id = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?", ("法律/合同违约.md",)
            ).fetchone()[0]
            old_document = tuple(conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone())
            old_chunks = _chunk_texts(conn, "法律/合同违约.md")
        finally:
            conn.close()
        old_vectors = store.get_document_chunks(document_id)
        old_count = store.count()
        if change == "delete":
            target.unlink()
        else:
            target.write_text(
                EMPTY_MD if change == "empty" else CONTRACT_MD + "\n\n新版本内容。",
                encoding="utf-8",
            )

        class FailOneCommit:
            def __init__(self, connection):
                self.connection = connection
                self.failed = False

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def commit(self):
                if self.connection.in_transaction and not self.failed:
                    self.failed = True
                    raise RuntimeError("simulated SQLite commit failure")
                self.connection.commit()

        original_connect = ingest_module.connect
        monkeypatch.setattr(ingest_module, "connect", lambda path: FailOneCommit(original_connect(path)))
        failed = _ingest(settings, embedding, store)
        assert failed.failed == 1
        assert failed.updated == failed.removed == 0
        assert store.count() == old_count
        assert store.get_document_chunks(document_id) == old_vectors
        assert _still_indexed(store, embedding, old_chunks) == old_chunks
        conn = _sqlite(settings)
        try:
            assert tuple(conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()) == old_document
        finally:
            conn.close()

        monkeypatch.setattr(ingest_module, "connect", original_connect)
        retried = _ingest(settings, embedding, store)
        assert retried.failed == 0
        assert (retried.removed if change == "delete" else retried.updated) == 1
    finally:
        store.close()


def test_failed_new_document_leaves_no_vectors(settings, fake_vault, monkeypatch):
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        old_count = store.count()
        (fake_vault / "new.md").write_text(CONTRACT_MD, encoding="utf-8")
        original_upsert = store.upsert_chunks
        failed_once = False

        def fail_once(rows):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                original_upsert(rows[:1])
                raise RuntimeError("simulated new document failure")
            original_upsert(rows)

        monkeypatch.setattr(store, "upsert_chunks", fail_once)
        failed = _ingest(settings, embedding, store)
        assert failed.failed == 1
        assert store.count() == old_count
        conn = _sqlite(settings)
        try:
            assert conn.execute("SELECT id FROM documents WHERE file_path = 'new.md'").fetchone() is None
        finally:
            conn.close()
        monkeypatch.setattr(store, "upsert_chunks", original_upsert)
        retried = _ingest(settings, embedding, store)
        assert retried.failed == 0
        assert retried.added == 1
    finally:
        store.close()


def test_vector_restore_failure_invalidates_index(settings, fake_vault, monkeypatch):
    embedding = get_embedding_client(settings)
    store = ChromaStore(settings.chroma_path_resolved)
    try:
        _ingest(settings, embedding, store)
        (fake_vault / "法律" / "合同违约.md").write_text(CONTRACT_MD + "\n\n更新。", encoding="utf-8")
        original_upsert = store.upsert_chunks

        def fail_every_write(rows):
            original_upsert(rows[:1])
            raise RuntimeError("simulated persistent write failure")

        monkeypatch.setattr(store, "upsert_chunks", fail_every_write)
        with pytest.raises(VectorCleanupError, match="recovery failed"):
            _ingest(settings, embedding, store)
        assert store.read_index_info() is None
        with pytest.raises(EmbeddingMismatchError):
            Retriever(embedding, store).search("合同违约")
    finally:
        store.close()
