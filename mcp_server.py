"""MCP 服务器：让 Claude Code 在对话里直接检索 Obsidian 知识库。

stdio 传输——Claude Code 会 spawn 这个进程，通过 stdin/stdout 走 JSON-RPC。
engine 第一次工具调用时构建一次并常驻复用：bge-m3 权重 2.17 GB，如果每个请求
都重建，每次问答都要多等十几秒。这条纪律和 app.main 里的 reranker 加载一致。

只暴露只读入口。ingest 会写 SQLite 和 Chroma，所以不在 MCP 里开——
「我读了什么」和「谁改了知识库」要分开，改知识库必须显式跑 ingest.py。
"""
from __future__ import annotations

import atexit
import time
from typing import Literal

from mcp.server.mcpserver import MCPServer

from app.config import Settings
from app.engine import build_engine
from app.evidence import STATUS_INSUFFICIENT
from app.rag import MODE_NORMAL, RAGAnswer, RAGEngine
from app.retriever import EmbeddingMismatchError, SearchResult

# 和 ask.py 保持一致：引用来源只展示开头这么多字符，剩下的让用户自己打开文件看
SOURCE_PREVIEW_CHARS = 200

# 和 app.main.ChatRequest 的约束保持一致：top_k 上限 20
MAX_TOP_K = 20

INSTRUCTIONS = """\
你的 Obsidian 知识库检索工具（私人笔记，只读，不改任何文件）。

- ask：问自然语言问题，返回带引用来源的答案。默认用它。
- search：只检索不调 LLM，返回语义最相近的原文片段。当 ask 的答案来源不对、
  或者你想直接看笔记原文时用它。

工具返回「证据不足」警告时，说明知识库里确实没有相关内容——请如实说明，
不要基于那些语义相近的资料编造答案。
"""


_ENGINE_STATE: tuple[RAGEngine, Settings] | None = None


def _engine() -> tuple[RAGEngle, Settings]:
    """懒加载引擎单例，返回 (engine, 生效的 settings)。

    MCP 握手必须在 30 秒超时内完成，而加载 bge-m3 权重（2.17 GB、391 个分片）
    要十几秒。把加载放在 build_server 里，Claude Code 会直接连不上这个 server。
    留到第一次工具调用：握手秒回，代价是第一次问答多等十几秒。
    """
    global _ENGINE_STATE
    if _ENGINE_STATE is None:
        settings = Settings()
        engine, store = build_engine(settings)
        atexit.register(store.close)
        _ENGINE_STATE = (engine, settings)
    return _ENGINE_STATE


def build_server() -> MCPServer:
    """装配 MCP server 并注册工具。engine 在此处不加载，见 _engine()。"""
    server = MCPServer("obsidian-rag", instructions=INSTRUCTIONS)

    @server.tool()
    def ask(
        question: str,
        top_k: int | None = None,
        mode: Literal["normal", "advisor"] = MODE_NORMAL,
    ) -> str:
        """在你的 Obsidian 知识库里检索并生成带引用来源的回答。

        返回答案正文 + 引用来源（文档名、文件路径、相似度）+ 检索状态。
        mode="advisor" 时额外注入 personal_context/context.yaml 里的当前情况。
        """
        engine, settings = _engine()
        k = _clamp_top_k(top_k, settings.top_k)
        started = time.perf_counter()
        try:
            result = engine.answer(question, top_k=k, mode=mode)
        except EmbeddingMismatchError as exc:
            return _embedding_mismatch_text(exc)
        return render_answer(result, time.perf_counter() - started)

    @server.tool()
    def search(question: str, top_k: int | None = None) -> str:
        """在你的知识库里按语义检索原文片段，先按知识域过滤再检索。

        返回相似度排名和文件路径，不调 LLM。用来排查检索效果：
        看期望的文档排在第几、被什么挤掉了。
        """
        engine, settings = _engine()
        k = _clamp_top_k(top_k, settings.top_k)
        hits = engine.retriever.search(question, top_k=k, domains=_routed_domains(engine, question))
        return render_hits(hits)

    return server


def _clamp_top_k(top_k: int | None, default: int) -> int:
    """把调用方传的 top_k 收进 [1, MAX_TOP_K]，没传就用配置默认。"""
    if top_k is None:
        return default
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k 必须在 1..{MAX_TOP_K} 之间，当前：{top_k}")
    return top_k


def _routed_domains(engine: RAGEngine, question: str) -> list[str] | None:
    """先用路由器判断知识域，再按域过滤候选。

    路由器不存在、或路由结果是 global 时返回 None，等于走全库检索——
    和 ask 内部的路由分支保持同一行为。这里只走规则路由不带 LLM fallback，
    让检索入口的结果可预测。
    """
    router = engine.router
    if router is None:
        return None
    return router.route(question).domains or None


def _embedding_mismatch_text(exc: EmbeddingMismatchError) -> str:
    """把「向量库和当前 embedding 模型不匹配」翻成调用方能执行的指引。

    不吞异常换成报错文本，是因为 MCP 的错误帧对 LLM 来说不如可读文本有用——
    它需要拿到具体命令才知道怎么修。
    """
    return (
        f"**向量库和当前 embedding 模型不匹配**\n\n{exc}\n\n"
        "解决办法：换过 EMBEDDING_MODEL 之后要重建向量，"
        "在 obsidian-rag-backend 目录运行 `python ingest.py --rebuild-vectors`。"
    )


def render_answer(result: RAGAnswer, elapsed: float) -> str:
    """把一次问答渲染成给 LLM 读的 Markdown。

    证据状态必须跟着答案一起返回：判定 insufficient 时，检索到的只是语义相近
    的资料，不是答案的支撑。不把这条信息带给调用方，Claude Code 就会拿相近
    资料当依据往下编——这正是本项目立 evidence gate 的全部意义。
    """
    lines: list[str] = []
    if result.evidence is not None and result.evidence.status == STATUS_INSUFFICIENT:
        lines += [
            "**警告：知识库不足以回答这个问题**",
            "",
            "检索到的资料语义相近，但证据判定为 insufficient，不能当作答案依据。"
            "请如实说明知识库里没有相关内容，不要基于这些资料编造答案。",
            "",
        ]
    lines += [result.answer, ""]
    lines += _render_sources(result)
    lines += _render_status(result, elapsed)
    return "\n".join(lines).rstrip()


def render_hits(hits: list[SearchResult]) -> str:
    """把检索命中渲染成给 LLM 读的 Markdown，编号从 1 开始。"""
    if not hits:
        return "检索没有返回任何 chunk——检查向量库是否已导入（`python ingest.py`）。"
    lines = [f"检索到 {len(hits)} 条片段：", ""]
    for i, hit in enumerate(hits):
        heading = f" / {hit.heading}" if hit.heading else ""
        preview = _preview(hit.content)
        lines.append(f"[{i + 1}] 相似度 {hit.score:.4f}")
        lines.append(f"    文档：{hit.title}{heading}")
        lines.append(f"    文件：{hit.file_path}")
        lines.append(f"    原文：{preview}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _preview(content: str) -> str:
    """截断超长原文，超出部分留给用户自己打开文件看。"""
    if len(content) <= SOURCE_PREVIEW_CHARS:
        return content
    return content[:SOURCE_PREVIEW_CHARS] + "…"


def _render_sources(result: RAGAnswer) -> list[str]:
    """引用来源，编号和答案正文里的 [1][2] 一一对应。"""
    if not result.sources:
        return []
    lines = ["**引用来源**", ""]
    for source in result.sources:
        heading = f" / {source.heading}" if source.heading else ""
        lines.append(f"[{source.index}] 相似度 {source.score:.4f}")
        lines.append(f"    文档：{source.title}{heading}")
        lines.append(f"    文件：{source.file_path}")
        lines.append("")
    return lines


def _render_status(result: RAGAnswer, elapsed: float) -> list[str]:
    """检索状态：证据判定、路由决策、各阶段耗时。

    打印而不是省略：这些字段是判断「答案可不可信」的依据，藏在日志里等于没有。
    """
    lines = ["**检索状态**", ""]
    if result.evidence is not None:
        ev = result.evidence
        note = "（Judge 失败兜底，非真实评审）" if ev.debug.get("judge_error") else ""
        lines.append(f"- 证据：{ev.status} (confidence {ev.confidence:.2f}){note}")
        if ev.reason:
            lines.append(f"  理由：{ev.reason}")
    if result.route is not None:
        rt = result.route
        lines.append(f"- 路由：mode={rt.mode} domains={rt.domains} origin={rt.origin}")
    if result.mode != MODE_NORMAL:
        lines.append(f"- 模式：{result.mode}")
    lines.append(f"- 耗时：{elapsed:.1f} 秒")
    return lines


def main() -> None:
    """stdio 入口：Claude Code 通过 `python mcp_server.py` 启动。

    故意不在这里加载模型——MCP 握手有 30 秒超时，而 bge-m3 要十几秒，
    先加载模型会让 Claude Code 直接判定连接失败。见 _engine()。
    """
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
