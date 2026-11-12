"""MCP 服务器的测试：渲染逻辑 + 工具装配。

工具函数本体只有几行胶水，真正有行为的是它渲染出来的文本——尤其是证据
判定 insufficient 时，有没有把警告带给调用方。所以测试重心放在渲染函数上。
不加载真实模型：渲染函数是纯函数，装配和懒加载的测试把 build_engine
monkeypatch 成假实现——bge-m3 权重 2.17 GB，测试里不能真加载。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp_server
from app.citations import check_citations
from app.evidence import (
    ANSWER_STATUS_UNSUPPORTED,
    AnswerVerification,
    EvidenceVerdict,
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
)
from app.rag import RAGAnswer, Source
from app.retriever import EmbeddingMismatchError
from app.router import RouteDecision
from mcp_server import (
    MAX_TOP_K,
    _clamp_top_k,
    _embedding_mismatch_text,
    _routed_domains,
    render_answer,
    render_hits,
)


def make_answer(**overrides) -> RAGAnswer:
    """造一个 RAGAnswer，默认不带引用和证据信息。"""
    base = dict(answer="合同违约需要继续履行、采取补救措施或赔偿损失。", used_chunks=0)
    base.update(overrides)
    return RAGAnswer(**base)


def make_hit(**overrides):
    """造一条检索命中，字段和 SearchResult 对齐。"""
    base = dict(
        score=0.83,
        chunk_id=1,
        document_id=1,
        chunk_index=0,
        title="合同违约的认定与救济",
        file_path="法律/合同违约.md",
        heading="违约的类型",
        source="民法典笔记",
        content="违约可以分为预期违约与届期违约两种。",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_answer_without_evidence_has_no_warning():
    text = render_answer(make_answer(), elapsed=3.2)
    assert text.startswith("合同违约需要继续履行")
    assert "警告" not in text


def test_insufficient_evidence_warns_and_forbids_fabrication():
    """证据不足时警告必须在最前面，并明确禁止编造。

    这是 evidence gate 的全部意义：调用方（Claude Code）拿到相近资料后
    若没被告知不可信，就会把它们当依据往下编。
    """
    verdict = EvidenceVerdict(status=STATUS_INSUFFICIENT, confidence=0.35, reason="资料未涉及违约责任")
    text = render_answer(make_answer(evidence=verdict), elapsed=3.2)
    assert text.startswith("**警告：本次检索资料不足以回答这个问题**")
    assert "编造" in text
    assert "证据：insufficient (confidence 0.35)" in text
    assert "资料未涉及违约责任" in text


def test_sufficient_evidence_has_no_warning():
    verdict = EvidenceVerdict(status=STATUS_SUFFICIENT, confidence=0.9, reason="直接命中")
    text = render_answer(make_answer(evidence=verdict), elapsed=3.2)
    assert "警告" not in text
    assert "证据：sufficient (confidence 0.90)" in text


def test_judge_error_is_marked_not_silent():
    """judge_error 表示结论是兜底产生的，必须标出来，统计时才能剔除。"""
    verdict = EvidenceVerdict(status=STATUS_SUFFICIENT, confidence=0.5, debug={"judge_error": "timeout"})
    text = render_answer(make_answer(evidence=verdict), elapsed=1.0)
    assert "Judge 失败兜底" in text


def test_strict_judge_error_does_not_claim_full_corpus_absence():
    verdict = EvidenceVerdict(
        status=STATUS_INSUFFICIENT,
        confidence=0.0,
        reason="证据评审失败",
        debug={"judge_error": True, "strict_fail_closed": True},
    )
    text = render_answer(make_answer(evidence=verdict), elapsed=1.0)
    assert text.startswith("**警告：证据评审未完成，未生成答案**")
    assert "不代表知识库全库不存在相关内容" in text
    assert "严格模式已拒绝生成" in text


def test_failed_final_verification_marks_sources_unusable_without_leaking_claims():
    secret_claim = "未核验的个人事实不应出现在 MCP 响应里"
    verification = AnswerVerification(
        status=ANSWER_STATUS_UNSUPPORTED,
        reason=f"不支持：{secret_claim}",
        claims=[{"claim": secret_claim, "supported": False, "basis": "personal_context"}],
        unsupported_claims=[secret_claim],
    )
    text = render_answer(
        make_answer(answer=secret_claim, sources=[Source(
            index=1, title="相近资料", file_path="notes.md", heading="", score=0.8, content="x",
        )], answer_verification=verification),
        elapsed=1.0,
    )

    assert text.startswith("**警告：最终答案未通过逐项证据核对，未输出原回答**")
    assert "不能把检索到的来源当作本次答案的依据" in text
    assert "检索来源（未核验，不可作为答案依据）" in text
    assert secret_claim not in text
    assert "claims" not in text
    assert "unsupported_claims" not in text
    assert "最终答案核验：unsupported" in text


def test_sources_are_numbered_and_keep_paths():
    answer = make_answer(
        sources=[
            Source(index=1, title="合同违约", file_path="法律/合同违约.md", heading="免责事由",
                   score=0.8, content="x"),
            Source(index=2, title="民法总则", file_path="法律/总则.md", heading="", score=0.7, content="y"),
        ],
    )
    text = render_answer(answer, elapsed=2.0)
    assert "**引用来源**" in text
    assert "[1] 相关性分数 0.8000" in text
    assert "法律/合同违约.md" in text
    assert "/ 免责事由" in text
    assert "[2] 相关性分数 0.7000" in text
    assert "法律/总则.md" in text


def test_route_and_mode_are_reported():
    route = RouteDecision(mode="single_domain", domains=["legal"], origin="rules", confidence=0.9)
    text = render_answer(make_answer(route=route, mode="advisor"), elapsed=1.0)
    assert "路由：mode=single_domain" in text
    assert "domains=['legal']" in text
    assert "模式：advisor" in text


def test_elapsed_is_reported():
    assert "耗时：42.3 秒" in render_answer(make_answer(), elapsed=42.3)


def test_no_sources_means_no_source_section():
    assert "引用来源" not in render_answer(make_answer(), elapsed=1.0)


def test_hits_are_numbered_from_one():
    text = render_hits([make_hit(score=0.9), make_hit(score=0.5, chunk_id=2)])
    assert text.startswith("检索到 2 条片段：")
    assert "[1] 原始向量相似度 0.9000" in text
    assert "[2] 原始向量相似度 0.5000" in text
    assert "法律/合同违约.md" in text


def test_empty_hits_says_what_to_run():
    text = render_hits([])
    assert "没有返回任何 chunk" in text
    assert "ingest.py" in text


def test_search_returns_evidence_at_end_of_chunk():
    content = "背景内容。" * 160 + "关键依据位于段落末尾。"
    text = render_hits([make_hit(content=content)])
    assert content in text
    assert "关键依据位于段落末尾。" in text


@pytest.mark.parametrize("answer", ["错误引用[99]。", "没有引用的回答。"])
def test_answer_warns_about_citation_problems(answer):
    result = make_answer(answer=answer, citation_check=check_citations(answer, 1))
    text = render_answer(result, elapsed=1.0)
    assert text.startswith("**警告：")
    assert answer in text


def test_clamp_top_k_uses_default_when_absent():
    assert _clamp_top_k(None, 5) == 5


@pytest.mark.parametrize("k", [0, -3, MAX_TOP_K + 1])
def test_clamp_top_k_rejects_out_of_range(k):
    with pytest.raises(ValueError, match="top_k 必须在"):
        _clamp_top_k(k, 5)


def test_embedding_mismatch_text_tells_how_to_fix():
    text = _embedding_mismatch_text(EmbeddingMismatchError("维度 384 和索引里的 1024 不一致"))
    assert "不一致" in text
    assert "--rebuild-vectors" in text


def test_routed_domains_returns_none_without_router():
    assert _routed_domains(SimpleNamespace(router=None), "海因里希法则") is None


def test_routed_domains_passes_domains_through():
    engine = SimpleNamespace(router=SimpleNamespace(
        route=lambda question: SimpleNamespace(domains=["safety_management"])
    ))
    assert _routed_domains(engine, "海因里希法则") == ["safety_management"]


def test_routed_domains_treats_global_as_unfiltered():
    """global 路由的结果是空列表，必须转成 None 才等于全库检索。

    把空列表原样传给 Retriever 会导致 0 条候选，一次检索都返回不了。
    """
    engine = SimpleNamespace(router=SimpleNamespace(route=lambda question: SimpleNamespace(domains=[])))
    assert _routed_domains(engine, "随便什么问题") is None


class _NoopStore:
    """假的向量库句柄，只提供 atexit 注销需要的 close。"""

    def close(self) -> None:
        pass


def test_build_server_does_not_load_model(monkeypatch):
    """装配必须秒回：MCP 握手有 30 秒超时，而 bge-m3 权重要十几秒。

    模型加载留到第一次工具调用（_engine 懒加载）。这条断言是防回归——
    之前放在 build_server 里时，Claude Code 报
    "connection timed out after 30000ms"，server 完全连不上。
    """
    calls = []
    monkeypatch.setattr(
        mcp_server, "build_engine",
        lambda settings, **kwargs: calls.append(1) or (SimpleNamespace(), _NoopStore()),
    )
    monkeypatch.setattr(mcp_server, "_ENGINE_STATE", None)

    server = mcp_server.build_server()
    assert server.name == "obsidian-rag"
    assert calls == [], "build_server 加载了模型，MCP 握手会超时"

    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"ask", "search"}
    assert calls == [], "list_tools 也不该触发模型加载"


def test_engine_loads_once_and_is_reused(monkeypatch):
    """模型只在第一次工具调用时加载一次：既要秒回握手，又不能每个请求重建。

    不重建是另一条纪律——bge-m3 权重 2.17 GB，每问一句重建一次等于把问答
    变成 20 秒起步。
    """
    calls: list[int] = []

    def fake_build_engine(settings, **kwargs):
        calls.append(1)
        return SimpleNamespace(answer=lambda **kw: None, retriever=None), _NoopStore()

    monkeypatch.setattr(mcp_server, "build_engine", fake_build_engine)
    monkeypatch.setattr(mcp_server, "_ENGINE_STATE", None)

    first, first_settings = mcp_server._engine()
    second, second_settings = mcp_server._engine()
    assert calls == [1]
    assert first is second
    assert first_settings is second_settings


def test_script_entrypoint_configures_utf8_stdio_under_non_utf8_python():
    """The standalone MCP process must emit UTF-8 on a Windows-style stream.

    ``mcp_server`` is also imported by the HTTP app and tests, so the stream
    reconfiguration belongs only to the ``__main__`` entrypoint.  Patching the
    server loop keeps this regression test from loading the real model or
    waiting for MCP input.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "mcp_server.py"
    child_code = f'''
import runpy
import sys
from mcp.server.mcpserver import MCPServer

MCPServer.run = lambda self, *args, **kwargs: None
runpy.run_path({str(script)!r}, run_name="__main__")
print(f"stdout encoding={{sys.stdout.encoding}}")
print(f"stderr encoding={{sys.stderr.encoding}}", file=sys.stderr)
print("中文 stdout")
print("中文 stderr", file=sys.stderr)
'''
    env = os.environ.copy()
    env["PYTHONUTF8"] = "0"
    env.pop("PYTHONIOENCODING", None)

    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    stdout = completed.stdout.decode("utf-8")
    stderr = completed.stderr.decode("utf-8")
    assert "stdout encoding=utf-8" in stdout.lower()
    assert "stderr encoding=utf-8" in stderr.lower()
    assert "中文 stdout" in stdout
    assert "中文 stderr" in stderr
