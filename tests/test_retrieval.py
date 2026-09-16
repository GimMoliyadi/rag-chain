"""测试五：Search 可以返回相关 Chunk（完整链路：解析→切块→嵌入→向量库→检索）。"""
from __future__ import annotations

from app.ingest import run_ingest
from app.retriever import Retriever, SearchResult
from app.vectorstore import ChromaStore


def test_ingest_stats_look_right(ingested, settings):
    store, _, stats = ingested
    assert stats.scanned == 2
    assert stats.added == 2
    assert stats.unchanged == 0
    assert stats.failed == 0
    assert stats.chunks_written > 0
    # 向量库里的条数必须等于写入的 chunk 数，一个都不能丢
    assert store.count() == stats.chunks_written


def test_ingest_is_incremental(ingested, settings):
    """同一批文件再跑一遍，应该全部跳过，向量库条数不变。"""
    store, embedding, first = ingested
    before = store.count()

    second = run_ingest(settings, embedding, store)
    assert second.added == 0
    assert second.unchanged == 2
    assert second.chunks_written == 0
    assert store.count() == before


def test_stored_offsets_reslice_the_original_exactly(ingested, settings):
    """入库后 SQLite 里的下标必须能精确切出 chunk 原文，引用定位才靠得住。

    这是集成层不变式：单独测 chunker 用合成文本抓不住真实 Markdown 里的
    缩进、空行和换行差异，全库跑过才知道下标是不是真的对得上。
    """
    from app.database import connect

    conn = connect(settings.db_path_resolved)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        bad = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE substr(d.content, c.start_position + 1, c.end_position - c.start_position)
                 != c.content
            """
        ).fetchone()[0]
        assert rows > 0
        assert bad == 0, f"{bad}/{rows} 个 chunk 的下标切不回原文"
    finally:
        conn.close()


def test_search_returns_relevant_chunks(ingested, settings):
    store, embedding, _ = ingested
    retriever = Retriever(embedding=embedding, store=store, default_top_k=5)

    hits = retriever.search("最小可行产品")

    assert hits, "检索必须返回结果"
    for h in hits:
        assert isinstance(h, SearchResult)
        assert 0.0 <= h.score <= 1.0
        assert h.chunk_id > 0
        assert h.document_id > 0
        assert h.title and h.file_path
        assert h.content.strip()
    # 相似度必须从高到低排序
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    # 最好的结果应该真的讲到 MVP
    assert "最小可行产品" in hits[0].content


def test_search_metadata_is_sufficient_for_citation(ingested, settings):
    """引用格式需要 title + file_path + heading，这些字段必须带回来。"""
    store, embedding, _ = ingested
    retriever = Retriever(embedding=embedding, store=store, default_top_k=5)

    hit = retriever.search("合同违约责任")[0]
    assert hit.title
    assert hit.file_path.endswith(".md")
    assert hit.heading
    assert 0.0 <= hit.score <= 1.0


def test_top_k_is_respected(ingested, settings):
    store, embedding, _ = ingested
    retriever = Retriever(embedding=embedding, store=store, default_top_k=10)
    assert len(retriever.search("创业", top_k=2)) <= 2


def test_search_on_empty_store_returns_nothing(tmp_path):
    from app.embedding import HashingEmbedding

    store = ChromaStore(tmp_path / "empty_chroma")
    retriever = Retriever(embedding=HashingEmbedding(32), store=store, default_top_k=5)
    assert retriever.search("任何东西") == []
