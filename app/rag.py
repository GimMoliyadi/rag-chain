"""RAG 编排：检索 →（精排）→（过滤）→ 拼 Prompt → 调 LLM → 整理引用。

这是把前面所有零件串起来的地方，也是比赛答辩时最该讲的一个文件。

两个可选的中间层都是默认关闭的开关，且顺序固定为 精排 → 过滤：
    精排要在大候选池（默认 50）上重新打分，必须排在过滤前面；
    过滤只从已排序的池子里挑最终 Top-K，排在哪一步都不影响结果，
    放后面可以让它只处理精排后的少量候选。
两个开关全关时 _collect 走的就是原始 Top-K 检索，baseline 可复现。
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .context_filter import prepare_context
from .llm import BaseLLM
from .reranker import CrossEncoderReranker, RerankStats
from .retriever import Retriever, SearchResult

# 引用编号上限：超过这个数的检索结果不再参与编号，避免 Prompt 过长
MAX_CITATIONS = 8

# 检索不到可用 chunk 时的固定回复。answer() 和 answer_stream() 都引用这一份，
# 两处各写一遍的话，改一处漏一处就是调用方看到两套措辞。
EMPTY_KNOWLEDGE_BASE_ANSWER = (
    "知识库中目前没有足够的信息回答这个问题。可能原因：还没有 ingest 任何文档，"
    "你的问题在知识库里找不到对应内容，或者检索到的段落没有实质内容（例如纯链接导航）。"
    "可以先调用 POST /api/ingest 导入文档。"
)

SYSTEM_PROMPT = """你是「{vault_name}」知识库的问答助手。请严格遵守：

1. 只依据【参考资料】回答，不要使用参考资料之外的知识。
2. 每一处结论后面标注它来自哪条资料，用上标编号，例如：公司必须做风险评估[1]。
3. 如果参考资料不足以回答问题，直接说明「知识库中目前没有足够的信息回答这个问题」，
   并简要说明缺的是哪方面的内容。绝不要编造答案。
4. 不要编造引用编号：只能引用下方真实存在的编号。
5. 回答用中文，结构清晰，必要时用列表。
"""

USER_PROMPT_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请基于参考资料回答用户问题，并在结论后标注引用编号。"""

# 单条 chunk 进入 Prompt 前的最大长度，防止一个超长 chunk 吃掉全部上下文
MAX_CONTEXT_CHUNK_CHARS = 1200


@dataclass
class Source:
    """返回给调用方的一条引用来源。"""

    index: int
    title: str
    file_path: str
    heading: str
    score: float
    content: str


@dataclass
class RAGAnswer:
    """一次 RAG 问答的完整结果。"""

    answer: str
    sources: list[Source] = field(default_factory=list)
    used_chunks: int = 0


class RAGEngine:
    """把 Retriever 和 LLM 接成一条问答链路。"""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        top_k: int = 5,
        vault_name: str = "Obsidian",
        context_filter_enabled: bool = False,
        filter_retrieve_top_k: int = 20,
        reranker: CrossEncoderReranker | None = None,
        reranker_candidate_top_k: int = 50,
    ):
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._vault_name = vault_name
        # 两个开关都默认关闭：检索层的原始结果必须随时拿得到，评估才能复现 baseline。
        self._context_filter_enabled = context_filter_enabled
        self._filter_retrieve_top_k = filter_retrieve_top_k
        self._reranker = reranker
        self._reranker_candidate_top_k = reranker_candidate_top_k

    @property
    def rerank_stats(self) -> RerankStats | None:
        """精排的观测数据；未启用精排时为 None。A/B 表里的耗时增量从这里取。"""
        return self._reranker.stats if self._reranker is not None else None

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """完整问答：检索 → 拼 Prompt → 调 LLM → 返回答案和引用。"""
        hits, contents = self._collect(question, top_k or self._top_k)
        numbered = hits[:MAX_CITATIONS]

        if not numbered:
            return RAGAnswer(answer=EMPTY_KNOWLEDGE_BASE_ANSWER, sources=[], used_chunks=0)

        messages = _build_messages(question, _build_context(numbered, contents), self._vault_name)
        return RAGAnswer(
            answer=self._llm.chat(messages),
            sources=_to_sources(numbered),
            used_chunks=len(numbered),
        )

    def answer_stream(self, question: str, top_k: int | None = None) -> Iterator[dict]:
        """流式问答：按阶段吐事件，调用方因此能区分「在检索」还是「在生成」。

        事件顺序：
            retrieval_started → retrieval_completed → sources
            → generation_started → token ×N → done
        任意阶段抛异常时吐一条 error 后结束。检索结果为空时不调用 LLM，
        直接把固定提示当作 answer 放进 done。

        这里只产出结构化 dict，SSE 的 `data: {...}` 包装留在 main.py 里做——
        这样 rag 层不依赖 HTTP，测试直接迭代这个生成器就能断言事件。
        异常也必须在生成器内部自己接：一旦 HTTP 响应头发出去了，
        FastAPI 的 exception_handler 就够不到生成器里抛出的异常了。
        """
        k = top_k or self._top_k
        total_started = time.perf_counter()

        yield {"event": "retrieval_started", "data": {"question": question, "top_k": k}}

        retrieval_started = time.perf_counter()
        try:
            hits, contents = self._collect(question, k)
        except Exception as exc:
            yield {"event": "error", "data": {"message": str(exc)}}
            return
        numbered = hits[:MAX_CITATIONS]

        yield {
            "event": "retrieval_completed",
            "data": {"count": len(numbered), "elapsed_ms": _elapsed_ms(retrieval_started)},
        }
        # sources 紧跟检索完成吐出（而不是按调用方的展示顺序写在 token 之后）：
        # 数据在检索结束时就已经有了，早给可以让调用方在等生成时先展示来源。
        yield {
            "event": "sources",
            "data": {"sources": _source_dicts(numbered), "used_chunks": len(numbered)},
        }

        if not numbered:
            yield {
                "event": "done",
                "data": {
                    "answer": EMPTY_KNOWLEDGE_BASE_ANSWER,
                    "used_chunks": 0,
                    "elapsed_ms": _elapsed_ms(total_started),
                },
            }
            return

        messages = _build_messages(question, _build_context(numbered, contents), self._vault_name)
        yield {"event": "generation_started", "data": {}}

        parts: list[str] = []
        try:
            for chunk in self._llm.chat_stream(messages):
                parts.append(chunk)
                yield {"event": "token", "data": {"text": chunk}}
        except Exception as exc:
            # 已经吐过一部分 token 时上游挂了，保留已生成的内容 + 一条错误，
            # 调用方至少能看到半篇回答，而不是什么都没留下。
            yield {"event": "error", "data": {"message": str(exc)}}
            return

        yield {
            "event": "done",
            "data": {
                "answer": "".join(parts),
                "used_chunks": len(numbered),
                "elapsed_ms": _elapsed_ms(total_started),
            },
        }

    def _collect(self, question: str, k: int) -> tuple[list[SearchResult], dict[int, str] | None]:
        """检索并整理最终进入 context 的 chunk，返回已按 chunk_id 去重的结果。

        返回 (hits, contents)：hits 决定引用编号和 sources（始终用原始 chunk），
        contents 是按 chunk_id 索引的清洗后正文，None 表示未开启清洗、直接用原文。

        两个开关都没开时池子就是 k 条、下面两个分支都被跳过，等价于原始 Top-K 检索，
        所以 baseline 不受这段改动影响。
        """
        limit = min(k, MAX_CITATIONS)

        # 候选池大小由三个开关各自决定，取最大的那个；多取不会多算成本之外的副作用。
        pool_size = k
        if self._context_filter_enabled:
            pool_size = max(pool_size, self._filter_retrieve_top_k)
        if self._reranker is not None:
            pool_size = max(pool_size, self._reranker_candidate_top_k)

        # 去重放在一切加工之前，免得重复 chunk 白占候选名额
        pool = _dedup_by_chunk_id(self._retriever.search(question, top_k=pool_size))

        if self._reranker is not None:
            pool = self._reranker.rerank(question, pool)

        if not self._context_filter_enabled:
            return pool[:limit], None

        # 只挑会真正被编号的那几条，超出的候选不做无谓的清洗
        selection = prepare_context(pool, top_k=limit)
        hits = [chunk.hit for chunk in selection.selected]
        contents = {chunk.hit.chunk_id: chunk.context_content for chunk in selection.selected}
        return hits, contents


def _build_messages(question: str, context: str, vault_name: str) -> list[dict]:
    """拼发给 LLM 的 messages。answer() 和 answer_stream() 共用这一份，
    否则两条链路各自的 Prompt 写法早晚会有出入。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(vault_name=vault_name)},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def _to_sources(hits: list[SearchResult]) -> list[Source]:
    """把检索结果编号成 Source，编号从 1 开始，和 Prompt 里的 [1][2] 一一对应。"""
    return [
        Source(
            index=i + 1,
            title=h.title,
            file_path=h.file_path,
            heading=h.heading,
            score=round(h.score, 4),
            content=h.content,
        )
        for i, h in enumerate(hits)
    ]


def source_to_dict(source: Source) -> dict:
    """Source → API 响应里的 dict。/api/chat 和 /api/chat/stream 共用，
    保证两个端点返回的 sources 字段结构完全一致。"""
    return {
        "index": source.index,
        "title": source.title,
        "file_path": source.file_path,
        "heading": source.heading,
        "score": source.score,
        "chunk": source.content,
    }


def _source_dicts(hits: list[SearchResult]) -> list[dict]:
    return [source_to_dict(s) for s in _to_sources(hits)]


def _elapsed_ms(started: float) -> int:
    """time.perf_counter() 差值转成整数毫秒，调用方直接展示。"""
    return int((time.perf_counter() - started) * 1000)


def _dedup_by_chunk_id(hits: list[SearchResult]) -> list[SearchResult]:
    """去掉重复的 chunk，保留第一次出现的位置（即相似度更高的那次）。"""
    seen: set[int] = set()
    out: list[SearchResult] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        out.append(hit)
    return out


def _build_context(hits: list[SearchResult], contents: dict[int, str] | None = None) -> str:
    """把检索结果拼成带编号的参考资料文本，编号和返回给调用方的 sources 严格对应。

    contents 不为空时用清洗后的正文进 Prompt（contents 缺失的 chunk 回落到原文），
    这样 LLM 看不到 JSON 导出尾巴，而 sources 里仍然是原始 chunk 原文。
    """
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        content = (contents.get(hit.chunk_id, hit.content) if contents else hit.content)
        if len(content) > MAX_CONTEXT_CHUNK_CHARS:
            content = content[:MAX_CONTEXT_CHUNK_CHARS] + "…"
        heading = f" / {hit.heading}" if hit.heading else ""
        blocks.append(
            f"[{i}] 文档：{hit.title}{heading}\n"
            f"     文件：{hit.file_path}\n"
            f"     内容：{content}"
        )
    return "\n\n".join(blocks)
