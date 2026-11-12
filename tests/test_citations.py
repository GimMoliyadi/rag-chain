from __future__ import annotations

import pytest

from app.citations import check_citations, citation_warning
from app.rag import RAGEngine
from app.retriever import Retriever
from fakes import ScriptedLLM


@pytest.mark.parametrize(("answer", "count", "status", "cited", "invalid"), [
    ("结论[1][2]，另一处[1]。", 3, "ok", [1, 2], []),
    ("结论[1, 2]和[2，3]。", 3, "ok", [1, 2, 3], []),
    ("救援结果中的依据[20]。", 20, "ok", [20], []),
    ("结论[0][9]。", 2, "invalid", [0, 9], [0, 9]),
    ("结论没有引用。", 2, "missing", [], []),
    ("```python\nx = [99]\n```\n依据[1]。", 1, "ok", [1], []),
    ("~~~\n[88]\n~~~\n代码 `[77]`，依据[1]。", 1, "ok", [1], []),
    ("代码 ``x = [99]``，依据[1]。", 1, "ok", [1], []),
    ("依据[1]。\n```python\nx = [99]", 1, "ok", [1], []),
    (r"转义 \[99]，[88](https://example.test)，依据[1]。", 1, "ok", [1], []),
])
def test_citation_numbers(answer, count, status, cited, invalid):
    check = check_citations(answer, count)
    assert check["status"] == status
    assert check["cited_indexes"] == cited
    assert check["invalid_indexes"] == invalid
    assert check["unused_indexes"] == sorted(set(range(1, count + 1)) - set(cited))


@pytest.mark.parametrize("answer", ["有依据[1]，错误编号[99]。", "回答未标注来源。"])
def test_sync_and_stream_citation_checks_match(settings, ingested, answer):
    engine = RAGEngine(Retriever(ingested[1], ingested[0]), ScriptedLLM([answer, answer]))
    result = engine.answer("什么是 MVP？")
    events = list(engine.answer_stream("什么是 MVP？"))
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["answer"] == result.answer == answer
    assert events[-1]["data"]["citation_check"] == result.citation_check
    assert citation_warning(result.citation_check)


def test_api_exposes_citation_check(client, monkeypatch):
    llm = ScriptedLLM(["有依据[1]，错误编号[99]。"])
    monkeypatch.setattr(client.app.state.rag, "_llm", llm)
    result = client.post("/api/chat", json={"question": "什么是 MVP？"}).json()
    assert result["citation_check"]["status"] == "invalid"
    assert result["citation_check"]["invalid_indexes"] == [99]


def test_empty_retrieval_skips_citation_check():
    class EmptyRetriever:
        def search(self, *args, **kwargs):
            return []

    engine = RAGEngine(EmptyRetriever(), ScriptedLLM([]))
    assert engine.answer("问题").citation_check is None
    events = list(engine.answer_stream("问题"))
    assert events[-1]["data"]["citation_check"] is None


def test_successful_number_check_has_no_warning():
    assert citation_warning(check_citations("结论[1]。", 1)) is None
    assert citation_warning(None) is None
