"""LangChain 迁移的兼容性检查：索引、引用 metadata、过滤、Prompt 格式。"""
from __future__ import annotations

from app.embedding import BaseEmbedding, HashingEmbedding
from app.langchain_adapters import LangChainEmbeddingsAdapter
from app.rag import (
    ADVISOR_PARTIAL_EVIDENCE_ADDENDUM,
    ADVISOR_USER_PROMPT_TEMPLATE,
    PARTIAL_EVIDENCE_ADDENDUM,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ADVISOR,
    USER_PROMPT_TEMPLATE,
    _build_advisor_messages,
    _build_messages,
)
from app.vectorstore import ChromaStore


class QueryAwareEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-query-aware"

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len("query: " + text)), 1.0]


def test_langchain_embedding_preserves_query_specific_encoding():
    backend = QueryAwareEmbedding()
    adapter = LangChainEmbeddingsAdapter(backend)
    assert adapter.embed_documents(["abc"]) == [[3.0, 0.0]]
    assert adapter.embed_query("abc") == [10.0, 1.0]


def _row(emb: HashingEmbedding, chunk_id: int, domain: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id,
        "chunk_index": 0,
        "title": "测试文档",
        "file_path": f"/tmp/{chunk_id}.md",
        "heading": "测试标题",
        "source": "obsidian",
        "source_type": "markdown",
        "folder": "",
        "tags": [],
        "domain": domain,
        "content": content,
        "embedding": emb.embed([content])[0],
    }


def test_langchain_chroma_reads_existing_index_and_filters_by_domain(tmp_path):
    emb = HashingEmbedding(32)
    store = ChromaStore(tmp_path / "chroma")
    try:
        store.upsert_chunks([
            _row(emb, 1, "technical", "python programming basics"),
            _row(emb, 2, "fiction", "dragon and sword"),
        ])
        assert store.count() == 2
        vector = emb.embed_query("python programming basics")
        all_rows = store.query(vector, n_results=2)
        assert len(all_rows) == 2
        assert all_rows[0]["chunk_id"] == 1
        assert all_rows[0]["content"] == "python programming basics"
        assert isinstance(all_rows[0]["distance"], float)

        scoped = store.query(vector, n_results=2, where={"domain": {"$in": ["fiction"]}})
        assert len(scoped) == 1
        assert scoped[0]["chunk_id"] == 2

        store.clear()
        assert store.count() == 0
        assert store.query(vector, n_results=2) == []
        store.upsert_chunks([_row(emb, 3, "technical", "another note")])
        assert store.query(emb.embed_query("another note"), n_results=1)[0]["chunk_id"] == 3
    finally:
        store.close()


def test_langchain_prompt_keeps_existing_normal_format():
    messages = _build_messages("什么是 RAG？", "文档片段", "我的库", partial=True)
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT.format(vault_name="我的库") + PARTIAL_EVIDENCE_ADDENDUM},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context="文档片段", question="什么是 RAG？")},
    ]


def test_langchain_prompt_keeps_advisor_isolation():
    messages = _build_advisor_messages(
        "我该怎么做？", "资料一", "我的库",
        personal_context="当前状态一", partial=True,
    )
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT_ADVISOR.format(vault_name="我的库") + ADVISOR_PARTIAL_EVIDENCE_ADDENDUM},
        {"role": "user", "content": ADVISOR_USER_PROMPT_TEMPLATE.format(
            personal_context="当前状态一", context="资料一", question="我该怎么做？"
        )},
    ]
