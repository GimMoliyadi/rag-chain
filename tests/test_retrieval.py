"""测试五：Search 可以返回相关 Chunk（完整链路：解析→切块→嵌入→向量库→检索）。"""
from __future__ import annotations

from app.ingest import run_ingest
from app.retriever import Retriever, SearchResult
from app.vectorstore import ChromaStore


class _CandidateStore:
    """只返回向量粗排候选的假 store，用来验证混合排序的候选余量。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.requested: list[int] = []

    def count(self) -> int:
        return len(self.rows)

    def read_index_info(self) -> dict:
        return {"model": "hashing-32d", "dimensions": 32}

    def query(self, embedding, n_results: int = 5, where=None) -> list[dict]:
        self.requested.append(n_results)
        return self.rows[:n_results]


class _ScopedCandidateStore(_CandidateStore):
    def document_paths(self):
        return list(dict.fromkeys(row["file_path"] for row in self.rows))

    def query(self, embedding, n_results=5, where=None):
        self.requested.append(where)
        def matches(row, condition):
            if not condition:
                return True
            if "$and" in condition:
                return all(matches(row, part) for part in condition["$and"])
            key, value = next(iter(condition.items()))
            return row.get(key) in value["$in"] if isinstance(value, dict) else row.get(key) == value
        return [row for row in self.rows if matches(row, where)][:n_results]


def test_version_scopes_keep_each_question_and_exclude_other_editions():
    from app.embedding import HashingEmbedding
    from app.reranker import CrossEncoderReranker
    from app.retriever import _source_scopes

    class Model:
        def predict(self, pairs, **kwargs):
            scores = []
            for query, text in pairs:
                if "\n\n" in text and text.endswith("\n\n"):
                    scores.append(100 if "保留期" in text else 0)
                elif "保留" in query:
                    scores.append(next((score for key, score in (
                        ("相关正文甲", 30), ("相关正文乙", 20), ("期限正文", 10),
                    ) if key in text), 0))
                else:
                    scores.append(100 if "温度正文" in text else 0)
            return scores

    rows = [
        dict(chunk_id=i, document_id=i, chunk_index=0, content=body, heading=heading,
             file_path=f"标准/{edition}/条款.md", title="条款", distance=0.25, domain="tests")
        for i, (edition, heading, body) in enumerate([
            ("GB1200-2024", "温度", "温度正文"), ("GB1200-2024", "适用范围", "范围正文"),
            ("GB1300-2023", "其他计算甲", "相关正文甲"),
            ("GB1300-2023", "其他计算乙", "相关正文乙"),
            ("GB1300-2023", "保留期", "期限正文"),
            ("GB1200-2011", "温度", "属于旧版的温度正文"),
        ], 1)
    ]
    retriever = Retriever(HashingEmbedding(32), _ScopedCandidateStore(rows),
                          reranker=CrossEncoderReranker(model=Model(), include_metadata=True))
    assert _source_scopes("只按GB1200-2024答：温度是多少？湿度呢？")[-1] == (
        "edition", "GB1200-2024", "湿度呢")

    hits = retriever.search("GB1200-2024 中，温度是多少？GB1300-2023 中，保留多久？", top_k=4,
                            domains=["tests"])
    assert [h.chunk_id for h in hits] == [1, 3, 2, 5]
    assert hits[-1].score == 0.75
    assert hits[-1].rerank_score == 10
    assert retriever.search("只按 GB1200-9999 查温度") == []


def test_explicit_relative_path_never_substitutes_another_note(ingested):
    store, embedding, _ = ingested
    retriever = Retriever(embedding, store)

    hits = retriever.search("只按《创业/MVP.md》回答：什么是最小可行产品？")
    assert hits and {h.file_path for h in hits} == {"创业/MVP.md"}
    assert retriever.search("只按《创业/不存在.md》回答：什么是最小可行产品？") == []


def test_reranker_recovers_named_document_section_beyond_global_pool():
    from app.embedding import HashingEmbedding
    from app.reranker import CrossEncoderReranker

    rows = [dict(chunk_id=i, document_id=i, chunk_index=0, content="定义", distance=0.3,
                 title=f"其他文档{i}", file_path=f"other/{i}.md", heading="定义", domain="quantum")
            for i in range(1, 52)]
    rows[0].update(title="Shor 算法", file_path="shor.md")
    rows[-1].update(document_id=1, title="Shor 算法", file_path="shor.md", content="实验：35", distance=0.6)
    # Same title in another domain must never be pulled across the scope.
    rows.append(dict(rows[-1], chunk_id=52, domain="unrelated", content="实验：99"))
    class Model:
        def predict(self, pairs, **kwargs):
            return [99.0 if "实验" in text else -3.0 for _, text in pairs]
    reranker = CrossEncoderReranker(model=Model(), include_metadata=True)
    store = _ScopedCandidateStore(rows)
    retriever = Retriever(HashingEmbedding(32), store, reranker=reranker,
                          diversity_enabled=True, max_chunks_per_document=1)

    hits = retriever.search("Shor 算法有哪些实际实验成果？", top_k=5, domains=["quantum"])

    assert hits[0].chunk_id == 51
    assert hits[0].score == 0.4  # not the reranker score of 99
    assert hits[0].rerank_score == 99.0
    assert 52 not in [h.chunk_id for h in hits]
    assert len(hits) == 5
    assert len({h.document_id for h in hits}) == 5
    assert retriever.last_debug["candidates"] == 51
    assert retriever.last_debug["reranker_applied"] is True
    assert store.requested[1] == {"$and": [{"domain": {"$in": ["quantum"]}}, {"document_id": 1}]}


def test_reranker_failure_reports_fallback_and_keeps_cosine():
    from app.embedding import HashingEmbedding
    from app.reranker import CrossEncoderReranker
    class Model:
        def predict(self, *args, **kwargs):
            raise RuntimeError("test failure")
    rows = [dict(chunk_id=1, document_id=1, chunk_index=0, content="正文", distance=0.2,
                 title="某文档", file_path="a.md", heading="正文")]
    retriever = Retriever(HashingEmbedding(32), _CandidateStore(rows),
                          reranker=CrossEncoderReranker(model=Model()))

    hits = retriever.search("未点名的检索")

    assert hits[0].score == 0.8
    assert retriever.last_debug["reranker_fallback"] is True
    assert retriever.last_debug["reranker_applied"] is False


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
    # 返回顺序由 Retriever 的混合相关性决定，不要求原始向量相似度单调；
    # 最好的结果应该真的讲到 MVP。
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


def test_hybrid_search_expands_candidates_and_promotes_exact_section():
    """标题/章节明确点名时，不能被向量 Top-5 的同文档章节压住。

    目标章节故意放在向量粗排第 18 条；只有候选池先扩大到 50，
    再用 metadata 词面信号重排，才能把它送进最终 Top-5。
    """
    from app.embedding import HashingEmbedding

    embedding = HashingEmbedding(32)
    rows = [
        {
            "chunk_id": 1, "document_id": 1, "chunk_index": 0,
            "content": "Shor 算法的定义。", "distance": 0.35,
            "title": "Shor 算法", "file_path": "quantum/shor.md", "heading": "定义",
        }
    ]
    rows.extend(
        {
            "chunk_id": i, "document_id": i, "chunk_index": 0,
            "content": f"无关候选 {i}。", "distance": 0.60,
            "title": f"其他文档 {i}", "file_path": f"other/{i}.md", "heading": "正文",
        }
        for i in range(2, 18)
    )
    rows.append(
        {
            "chunk_id": 18, "document_id": 1, "chunk_index": 1,
            "content": "Shor 算法的实验进展。", "distance": 0.50,
            "title": "Shor 算法", "file_path": "quantum/shor.md", "heading": "实验进展",
        }
    )
    rows.extend(
        {
            "chunk_id": i, "document_id": i, "chunk_index": 0,
            "content": f"无关候选 {i}。", "distance": 0.60,
            "title": f"其他文档 {i}", "file_path": f"other/{i}.md", "heading": "正文",
        }
        for i in range(19, 61)
    )
    store = _CandidateStore(rows)
    retriever = Retriever(embedding, store, default_top_k=5)

    hits = retriever.search("Shor 算法的实验进展有哪些？")

    assert store.requested == [50]
    assert hits[0].heading == "实验进展"
    # 混合排序把较低原始相似度的目标章节排到前面，但 score 仍保持
    # Chroma 原始余弦相似度，不能把内部 _rank_score 泄漏给调用方。
    assert round(hits[0].score, 2) == 0.50
    assert hits[1].heading == "定义"
    assert round(hits[1].score, 2) == 0.65
    assert len(hits) == 5
    assert retriever.last_debug["candidates"] == 50


def test_lexical_match_never_demotes_vector_candidate():
    """精确标题命中只加分，不应因短标题的词面分低于向量分而降权。"""
    from app.embedding import HashingEmbedding

    embedding = HashingEmbedding(32)
    store = _CandidateStore([
        {
            "chunk_id": 1, "document_id": 1, "chunk_index": 0,
            "content": "MVP 的定义。", "distance": 0.10,
            "title": "MVP", "file_path": "product/mvp.md", "heading": "定义",
        },
        {
            "chunk_id": 2, "document_id": 2, "chunk_index": 0,
            "content": "其他文档。", "distance": 0.15,
            "title": "其他文档", "file_path": "other.md", "heading": "正文",
        },
    ])
    retriever = Retriever(embedding, store, default_top_k=1)

    hits = retriever.search("MVP", top_k=1)

    assert hits[0].title == "MVP"
    # 仍然返回原始向量相似度，而不是内部混合排名分数。
    assert round(hits[0].score, 2) == 0.90


def test_diversity_can_return_more_than_twenty_results(tmp_path):
    from app.embedding import HashingEmbedding

    embedding = HashingEmbedding(32)
    store = ChromaStore(tmp_path / "diverse_chroma")
    rows = [
        {
            "chunk_id": i + 1, "document_id": i + 1, "chunk_index": 0,
            "content": f"不同文档 {i}", "embedding": embedding.embed_query(f"不同文档 {i}"),
        }
        for i in range(30)
    ]
    try:
        store.upsert_chunks(rows)
        store.write_index_info(embedding.fingerprint)
        retriever = Retriever(embedding, store, diversity_enabled=True)

        hits = retriever.search("不同文档", top_k=25)

        assert len(hits) == 25
        assert len({hit.document_id for hit in hits}) == 25
    finally:
        store.close()


def test_diversity_debug_counts_candidates_before_selection(ingested):
    store, embedding, _ = ingested
    retriever = Retriever(embedding, store, diversity_enabled=True, max_chunks_per_document=1)

    hits = retriever.search("合同违约", top_k=5)

    assert len(hits) == 2
    assert retriever.last_debug["candidates"] == store.count()
    assert retriever.last_debug["returned"] == 2
