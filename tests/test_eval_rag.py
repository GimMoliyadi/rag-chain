import json
from contextlib import closing

import pytest

from app.database import connect
from eval_rag import main
from fakes import ScriptedLLM


@pytest.mark.parametrize(
    ("category", "responses", "outcome", "verdict"),
    [
        ("direct", ["sufficient", "valid_answer", "supported"], "answered", "PASS"),
        ("unknown", ["insufficient"], "insufficient", "PASS"),
        ("unknown", ["malformed"], "gate_error", "FAIL"),
        ("direct", ["sufficient", "invalid_citation"], "verification_blocked", "FAIL"),
    ],
)
def test_production_cli_reports_strict_outcomes(
    settings, ingested, tmp_path, monkeypatch, category, responses, outcome, verdict,
):
    settings.evidence_gate_enabled = True
    settings.strict_evidence_enabled = True
    settings.top_k = 1
    with closing(connect(settings.db_path_resolved)) as conn:
        path = "法律/合同违约.md"
        snippet = conn.execute(
            "SELECT c.content FROM chunks c JOIN documents d ON c.document_id=d.id WHERE d.file_path=?",
            (path,),
        ).fetchone()[0]
    quote = "合同违约"
    assert quote in snippet
    response_for = {
        "sufficient": json.dumps({"status": "sufficient", "confidence": 0.9}),
        "insufficient": json.dumps({"status": "insufficient", "confidence": 0.9}),
        "valid_answer": "依据资料，合同违约的处理需要继续履行或采取补救措施。[1]",
        "invalid_citation": "依据资料，合同违约的处理需要继续履行或采取补救措施。[999]",
        "supported": json.dumps({
            "status": "supported", "confidence": 0.9,
            "claims": [{
                "unit_id": "u1", "claim": "合同违约的处理需要继续履行或采取补救措施",
                "is_claim": True, "supported": True, "citation_indexes": [1],
                "evidence_quotes": [{"citation_index": 1, "quote": quote}],
            }],
        }),
        "malformed": "invalid JSON",
    }
    llm = ScriptedLLM([response_for[response] for response in responses])
    monkeypatch.setattr("eval_rag.Settings", lambda **kwargs: settings)
    monkeypatch.setattr("eval_rag.get_llm_client", lambda _: llm)
    questions = tmp_path / "questions.json"
    report_path = tmp_path / "result.json"
    question = {
        "id": 1, "category": category, "verified": True,
        "question": "合同违约怎么处理？" if category == "direct" else "合同违约笔记里能查到当事人的编号吗？",
    }
    if category == "direct":
        question.update(
            expected_documents=[path], expected_evidence=[{"file_path": path, "quote": quote}],
            expected_answer_terms=["继续履行"],
        )
    questions.write_text(json.dumps([question]), encoding="utf-8")

    code = main(["--questions", str(questions), "--out", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    row, = report["results"]
    assert (row["outcome"], row["verdict"]) == (outcome, verdict)
    assert code == (0 if verdict == "PASS" else 1)
    assert report["empty_retrieval_guard"]["verdict"] == "PASS"
    assert report["empty_retrieval_guard"]["llm_calls"] == 0
    assert report["meta"]["mode"] == "production"
    assert report["meta"]["strict_evidence_enabled"] is True
    assert len(llm.calls) == len(responses)
