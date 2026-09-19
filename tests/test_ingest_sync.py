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

from app.database import connect
from app.embedding import get_embedding_client
from app.ingest import run_ingest
from app.vectorstore import ChromaStore

from conftest import CONTRACT_MD

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
