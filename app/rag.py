"""RAG 编排：路由 → 检索 →（精排）→（过滤）→（证据评审）→（证据救援）→ 拼 Prompt → 调 LLM → 整理引用。

这是把前面所有零件串起来的地方，也是比赛答辩时最该讲的一个文件。

可选中间层全部是默认关闭的开关。开启时的固定顺序：
    Router（规则，可选 LLM fallback）→ Scoped Retrieval → 精排 → 过滤
    → Evidence Gate（LLM 评审证据是否足够）→ Evidence Rescue（只在评审不足时）→ 生成
精排要在大候选池上重新打分，必须排在过滤前面；Evidence Gate 评审的是
「最终进入 Prompt 的那几条资料」，所以必须排在一切检索加工之后。
所有开关全关时，链路就是原始 Top-K 检索 + 原始 Prompt，baseline 可复现。

mode=advisor 是 Personal Advisor 的入口：额外读取用户手写的 Personal Context
（当前状态），换用「个人情况 / 知识库资料 / 问题」三段式 Prompt。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from .context_filter import prepare_context
from .evidence import (
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    EvidenceGate,
    EvidenceVerdict,
)
from .llm import BaseLLM
from .personal_context import PersonalContext, PersonalContextError, render_for_prompt
from .reranker import CrossEncoderReranker, RerankStats
from .retriever import Retriever, SearchResult
from .router import RouteDecision, route_with_fallback

# 引用编号上限：超过这个数的检索结果不再参与编号，避免 Prompt 过长
MAX_CITATIONS = 8

# /api/chat 的 mode 取值。normal 是历史行为，advisor 额外注入 Personal Context。
MODE_NORMAL = "normal"
MODE_ADVISOR = "advisor"
VALID_MODES = frozenset({MODE_NORMAL, MODE_ADVISOR})

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

# 证据评审判定 partial 时附加到 system prompt 的条款：允许回答，
# 但必须把「有依据的」和「缺依据的」分开说，不许用资料外知识补全。
PARTIAL_EVIDENCE_ADDENDUM = """
6. 证据评审判定这些资料只覆盖了问题的部分方面。回答时必须明确区分：
   - 「知识库能够支持的部分」：依据参考资料正常回答并标注引用编号；
   - 「知识库当前缺少依据的部分」：明确说明这部分在知识库中没有找到资料，
     绝不要用参考资料之外的知识补全。
"""

# 证据评审判定 insufficient 时的固定拒答文案。刻意不让 LLM 自由发挥——
# 让模型自己写拒答，它十有八九会顺手用自身知识把答案也写了。
# missing 部分来自 Judge 给出的 unsupported_aspects，让拒答具体而不是一句空话。
INSUFFICIENT_ANSWER_TEMPLATE = (
    "当前知识库中没有足够资料支持这个问题。\n\n"
    "缺少的依据：{missing}\n\n"
    "下面列出的引用只是与问题语义相近的资料，不足以作为答案依据，仅供参考。"
)

# ---- advisor 模式（Personal Context + 知识库）的 Prompt ----
# 与 SYSTEM_PROMPT 分开写而不是复用：两者的约束集合不同，硬拼在一起
# 会让普通问答也背上「区分三类信息」的负担。
SYSTEM_PROMPT_ADVISOR = """你是「{vault_name}」的个人顾问。你会收到三块材料，来源完全不同，必须严格分开使用：

【用户当前情况】= 用户亲手填写的当前状态（方向、项目、目标、约束），是用户亲述的事实，不是知识库内容。
【知识库资料】= 从知识库检索到的资料，可以引用，必须标注引用编号。
你自己 = 基于上面两块做出的分析、推断和建议。

硬性规则：
1. 【用户当前情况】里没有提到的信息，视同不知道。不要推断、不要补全、不要假设用户的处境。
   如果这一块标注为「当前没有个人状态信息」，必须在回答开头明确说明：缺少用户的当前情况，
   只能给出一般性分析，无法针对用户实际处境判断。
2. 【知识库资料】里没有的事实，绝不用你记忆中的知识补全；缺就明说缺。
3. 每一处来自知识库的结论后面用上标标注引用编号，例如：长期合作能降低反复博弈的成本[1]。
   只能引用下面真实存在的编号，不要编造。
4. 分析和建议必须明确是你的判断，不能写成知识库原文，也不能写成用户已确认的事实。
5. 涉及取舍、排序、优先级的问题，给出多个可选行动并说明各自的代价，不要宣称唯一正确答案。
6. 回答用中文。简单问题可以直接简洁回答，不必套结构；只有涉及个人决策或取舍时才按
   「当前情况 → 相关知识 → 分析 → 可选行动」四段组织，且「当前情况」只写
   【用户当前情况】里真实存在的内容。"""

ADVISOR_USER_PROMPT_TEMPLATE = """【用户当前情况】
{personal_context}

【知识库资料】
{context}

【用户问题】
{question}

请结合【用户当前情况】和【知识库资料】回答。知识库有依据的部分标注引用编号；
属于你分析推断的部分明确说明是你的判断，不要写成知识库原文。"""

# Personal Context 拿不到时的占位正文。必须在 Prompt 里显式说出来，
# 空一块【用户当前情况】会让模型顺着自己的记忆把用户情况补全掉。
PERSONAL_CONTEXT_UNAVAILABLE_NOTE = (
    "（当前没有个人状态信息：Personal Context 未开启、文件不存在或内容为空。"
    "请明确说明缺少用户的当前情况，只能给出一般性分析，不要猜测或编造用户的处境。）"
)

# advisor 模式下证据评审判定 partial 的附加条款。和普通模式的 partial 条款
# 语义相同，但措辞必须落在 advisor 的三类信息框架里。
ADVISOR_PARTIAL_EVIDENCE_ADDENDUM = """
7. 证据评审判定【知识库资料】只覆盖了问题的部分方面。回答时必须明确区分：
   - 「知识库能够支持的部分」：依据【知识库资料】正常回答并标注引用编号；
   - 「知识库当前缺少依据的部分」：明确说明知识库没有相应资料，绝不用知识库以外的知识补全。
   你的分析可以继续基于已有材料展开，但必须说明它属于你的判断。"""

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
    # ---- 以下均为可选的过程信息（Advisor 层开启后才有值）----
    # 各阶段耗时与流程标记（毫秒类 key 之外的 rescue_* / *_evidence_status
    # 也是刻意放在这里：调用方按这一个 dict 就能按时间顺序看清发生了什么）。
    timings: dict = field(default_factory=dict)
    # 路由决策；未启用 Router 或调用方手动指定了 domains 时为 None。
    route: RouteDecision | None = None
    # 证据评审结论；未启用 Evidence Gate 时为 None。
    evidence: EvidenceVerdict | None = None
    # 本次请求的 mode；正常调用总是有值。
    mode: str = MODE_NORMAL
    # Personal Context 的读取结果；只有 advisor 模式才有值，normal 模式为 None。
    personal_context: dict | None = None


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
        router: object | None = None,
        router_domains: list[str] | None = None,
        llm_router_fallback_enabled: bool = False,
        evidence_gate: EvidenceGate | None = None,
        evidence_rescue_enabled: bool = False,
        evidence_rescue_k: int = 20,
        rescue_retriever: Retriever | None = None,
        personal_context_loader: Callable[[], PersonalContext] | None = None,
    ):
        if evidence_rescue_enabled and evidence_gate is None:
            raise ValueError("evidence_rescue_enabled 需要 evidence_gate，救援是在第一次评审之后跑的")
        if evidence_rescue_enabled and rescue_retriever is None:
            raise ValueError("evidence_rescue_enabled 需要一个 rescue_retriever 来扩大候选池")
        if evidence_rescue_k < 1:
            raise ValueError(f"evidence_rescue_k 必须是正整数，当前：{evidence_rescue_k}")

        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._vault_name = vault_name
        # 两个开关都默认关闭：检索层的原始结果必须随时拿得到，评估才能复现 baseline。
        self._context_filter_enabled = context_filter_enabled
        self._filter_retrieve_top_k = filter_retrieve_top_k
        self._reranker = reranker
        self._reranker_candidate_top_k = reranker_candidate_top_k
        # ---- Advisor 层（全部默认关闭，关闭时下方分支一行都不会执行）----
        # router 用 object 注入 + route_with_fallback 里才碰它的方法，是为了让
        # 本模块对路由器只依赖 route()/domain_names() 这两个调用点，测试里
        # 用轻量替身就能喂进来，不必背 QueryRouter 的完整构造。
        self._router = router
        self._router_domains = router_domains or []
        self._llm_router_fallback_enabled = llm_router_fallback_enabled
        self._evidence_gate = evidence_gate
        # 证据救援用独立的 Retriever 实例：主检索器可能开着「每文档最多 N chunk」
        # 的 diversity 限额，救援候选池必须不带这个限额（见 _rescue_collect 说明）。
        self._evidence_rescue_enabled = evidence_rescue_enabled
        self._evidence_rescue_k = evidence_rescue_k
        self._rescue_retriever = rescue_retriever
        # Personal Context 通过 loader 注入：默认由 create_app 绑到配置文件，
        # 测试里换成返回固定 PersonalContext 的 lambda 即可，不必碰文件系统。
        self._personal_context_loader = personal_context_loader

    @property
    def rerank_stats(self) -> RerankStats | None:
        """精排的观测数据；未启用精排时为 None。A/B 表里的耗时增量从这里取。"""
        return self._reranker.stats if self._reranker is not None else None

    @property
    def retriever(self) -> Retriever:
        """只读访问内部检索器。

        /api/search 和 MCP 的 search 工具只做检索、不调 LLM——这是本项目排查
        检索问题的固定入口。走这个 property 而不是让调用方自己 new 一个
        Retriever，是为了避免多持有一份 Chroma 连接，也让检索参数和问答
        路径始终一致。
        """
        return self._retriever

    @property
    def router(self) -> object | None:
        """只读访问路由器，和 retriever 的用途一样：让只检索的入口也能按
        知识域过滤候选。不过滤的话「海因里希法则」会被量子力学的相似词
        （变分法、绝热近似）挤到前面，检索到完全无关的领域。
        """
        return self._router

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        domains: list[str] | None = None,
        mode: str = MODE_NORMAL,
    ) -> RAGAnswer:
        """完整问答：路由 → 检索 →（证据评审）→（证据救援）→ 拼 Prompt → 调 LLM → 返回答案和引用。

        domains 手动指定时跳过 Router（与 search.py --domains 语义一致），
        只有 domains=None 且 Router 已注入时才走路由。

        mode=normal 是历史行为；mode=advisor 额外读取 Personal Context 并换用
        「个人情况 / 知识库资料 / 问题」三段式 Prompt。
        """
        mode = _normalize_mode(mode)
        timings: dict = {}
        route = self._route_if_needed(question, domains, timings)
        if route is not None:
            domains = route.domains or None

        # 只有 advisor 模式才碰 Personal Context；normal 模式一个字节都不读
        personal_context_text: str | None = None
        personal_context_info: dict | None = None
        if mode == MODE_ADVISOR:
            started = time.perf_counter()
            personal_context_text, personal_context_info = self._load_personal_context()
            timings["personal_context_ms"] = _elapsed_ms(started)

        k = top_k or self._top_k
        started = time.perf_counter()
        hits, contents = self._collect(question, k, domains)
        timings["retrieval_ms"] = _elapsed_ms(started)

        if not hits:
            return RAGAnswer(
                answer=EMPTY_KNOWLEDGE_BASE_ANSWER, sources=[], used_chunks=0,
                timings=timings, route=route, mode=mode, personal_context=personal_context_info,
            )

        # Evidence Gate 评审的是最终进入 Prompt 的那几条资料，
        # 所以必须在精排 / 过滤 / 去重全部完成之后。
        evidence_hits = hits[:MAX_CITATIONS]
        evidence_contents = contents
        verdict: EvidenceVerdict | None = None
        if self._evidence_gate is not None:
            started = time.perf_counter()
            verdict = self._evidence_gate.judge(
                question, evidence_hits, personal_context=personal_context_text,
            )
            timings["evidence_judge_ms"] = _elapsed_ms(started)
            timings["first_evidence_status"] = verdict.status
            if self._evidence_rescue_enabled:
                # 救援开启时这个标记恒有值：调用方可以区分「没触发救援」和「救援没开」。
                timings["rescue_triggered"] = False

            # Evidence Rescue：只在第一次评审不足时跑一次，sufficient / partial 不多花一步
            if verdict.status == STATUS_INSUFFICIENT and self._evidence_rescue_enabled:
                evidence_hits, verdict = self._rescue(
                    question, domains, evidence_hits, verdict, personal_context_text, timings,
                )
                if timings.get("rescue_triggered"):
                    # 救援池没有经过首轮的 context 清洗，只能退回原文
                    evidence_contents = None

            timings["final_evidence_status"] = verdict.status

            if verdict.status == STATUS_INSUFFICIENT:
                # 拒答不调生成 LLM：Judge 已经判定资料不足，让生成模型再自由发挥
                # 一次只会引入资料外知识。sources 照常返回，调用方据
                # evidence.status 展示「语义相近但不可作为依据」的标记。
                return RAGAnswer(
                    answer=_insufficient_answer(verdict),
                    sources=_to_sources(evidence_hits),
                    used_chunks=len(evidence_hits),
                    timings=timings, route=route, evidence=verdict,
                    mode=mode, personal_context=personal_context_info,
                )

        partial = verdict is not None and verdict.status == STATUS_PARTIAL
        context = _build_context(evidence_hits, evidence_contents)
        if mode == MODE_ADVISOR:
            messages = _build_advisor_messages(
                question, context, self._vault_name,
                personal_context=personal_context_text or PERSONAL_CONTEXT_UNAVAILABLE_NOTE,
                partial=partial,
            )
        else:
            messages = _build_messages(question, context, self._vault_name, partial=partial)
        started = time.perf_counter()
        answer = self._llm.chat(messages)
        timings["answer_generation_ms"] = _elapsed_ms(started)
        return RAGAnswer(
            answer=answer,
            sources=_to_sources(evidence_hits),
            used_chunks=len(evidence_hits),
            timings=timings, route=route, evidence=verdict,
            mode=mode, personal_context=personal_context_info,
        )

    def _route_if_needed(
        self, question: str, domains: list[str] | None, timings: dict,
    ) -> RouteDecision | None:
        """手动指定 domains 或未注入 Router 时返回 None（不路由）；
        否则走「规则优先、零命中才 LLM fallback」的统一入口并计时。"""
        if domains is not None or self._router is None:
            return None
        started = time.perf_counter()
        decision = route_with_fallback(
            self._router, question,
            allowed_domains=self._router_domains,
            llm=self._llm,
            fallback_enabled=self._llm_router_fallback_enabled,
        )
        timings["router_ms"] = _elapsed_ms(started)
        return decision

    def _rescue(
        self,
        question: str,
        domains: list[str] | None,
        first_hits: list[SearchResult],
        first_verdict: EvidenceVerdict,
        personal_context: str | None,
        timings: dict,
    ) -> tuple[list[SearchResult], EvidenceVerdict]:
        """Evidence Rescue：第一次评审不足时的唯一一次救援（扩大候选池 → 再评审一次）。

        返回 (最终证据, 最终结论)。救援池不比首轮多出新 chunk 时原样返回首轮结果——
        同一批证据再评审一次只会得到同一个结论，白白多花一次 LLM 调用。
        """
        started = time.perf_counter()
        rescue_hits = self._rescue_collect(question, domains)
        timings["rescue_retrieval_ms"] = _elapsed_ms(started)

        first_ids = {hit.chunk_id for hit in first_hits}
        if not ({hit.chunk_id for hit in rescue_hits} - first_ids):
            return first_hits, first_verdict

        timings["rescue_triggered"] = True
        started = time.perf_counter()
        second_verdict = self._evidence_gate.judge(
            question, rescue_hits, personal_context=personal_context,
        )
        timings["rescue_evidence_ms"] = _elapsed_ms(started)
        # 留个标记：这条结论是救援产生的，统计时能把「首轮就判出来的」和
        # 「扩大候选池之后才判出来的」分开，前者才是 Judge 本身的能力。
        second_verdict.debug["rescue_used"] = True
        return rescue_hits, second_verdict

    def _rescue_collect(self, question: str, domains: list[str] | None) -> list[SearchResult]:
        """扩大候选池的检索：不带「每文档最多 N chunk」限额，也不走精排。

        不带限额是有实测依据的：凸轮运动规律那道题的 Top-5 是「凸轮文档 chunk0/1
        + 间歇运动机构 ×3」，真正含运动规律表的答案是凸轮文档的第 3 个 chunk、
        真实排名 #7。开 diversity（每文档 2 chunk）时它会被限额直接挤掉，
        救援就救不回这道题；关掉限额后它稳定落在 #7。

        也不按 MAX_CITATIONS 截断：Judge 和生成器看到的是同一批证据，
        截断会造成「Judge 看了 20 条、生成只拿到 8 条」的不一致。
        池子大小由 EVIDENCE_RESCUE_K 显式控制。
        """
        return _dedup_by_chunk_id(
            self._rescue_retriever.search(question, top_k=self._evidence_rescue_k, domains=domains)
        )

    def _load_personal_context(self) -> tuple[str | None, dict]:
        """读取并渲染 Personal Context，返回 (Prompt 正文或 None, 调试信息)。

        正文为 None 的三种情况都会体现在调试信息里：未启用、文件无内容、
        YAML 结构写错。后两种不阻断问答——用户改错一个字段不该让整个系统答不了话，
        但错误信息必须出现在响应里，不能静默。
        """
        if self._personal_context_loader is None:
            return None, {"loaded": False, "reason": "Personal Context 未启用"}

        try:
            context = self._personal_context_loader()
        except PersonalContextError as exc:
            return None, {"loaded": False, "error": str(exc)}

        rendered = render_for_prompt(context)
        if not rendered:
            info = context.to_debug_dict()
            info["reason"] = "Personal Context 没有内容"
            return None, info
        return rendered, context.to_debug_dict()

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
        domains: list[str] | None = None,
        mode: str = MODE_NORMAL,
    ) -> Iterator[dict]:
        """流式问答：按阶段吐事件，调用方因此能区分「在检索」还是「在生成」。

        事件顺序（routing / evidence / rescue 事件仅在对应功能开启时出现）：
            routing_completed → retrieval_started → retrieval_completed → sources
            → evidence_completed →〔evidence_rescue_completed〕
            → generation_started → token ×N → done
        救援发生时 evidence_rescue_completed 会带上新的 sources，取代前面的 sources
        事件——调用方取最后一次看到的 sources 即可。
        任意阶段抛异常时吐一条 error 后结束。检索结果为空、或证据评审判定
        insufficient 时不进入生成阶段，直接把固定文案当作 answer 放进 done。

        这里只产出结构化 dict，SSE 的 `data: {...}` 包装留在 main.py 里做——
        这样 rag 层不依赖 HTTP，测试直接迭代这个生成器就能断言事件。
        异常也必须在生成器内部自己接：一旦 HTTP 响应头发出去了，
        FastAPI 的 exception_handler 就够不到生成器里抛出的异常了。
        """
        mode = _normalize_mode(mode)
        k = top_k or self._top_k
        total_started = time.perf_counter()
        timings: dict = {}

        route = self._route_if_needed(question, domains, timings)
        if route is not None:
            yield {
                "event": "routing_completed",
                "data": {"route": _route_dict(route), "elapsed_ms": timings.get("router_ms", 0)},
            }
        domains_effective = (route.domains or None) if route is not None else domains

        # advisor 模式才读 Personal Context；normal 模式一个字节都不碰
        personal_context_text: str | None = None
        personal_context_info: dict | None = None
        if mode == MODE_ADVISOR:
            started = time.perf_counter()
            personal_context_text, personal_context_info = self._load_personal_context()
            timings["personal_context_ms"] = _elapsed_ms(started)

        yield {"event": "retrieval_started", "data": {"question": question, "top_k": k}}

        retrieval_started = time.perf_counter()
        try:
            hits, contents = self._collect(question, k, domains_effective)
        except Exception as exc:
            yield {"event": "error", "data": {"message": str(exc)}}
            return
        timings["retrieval_ms"] = _elapsed_ms(retrieval_started)
        numbered = hits[:MAX_CITATIONS]

        yield {
            "event": "retrieval_completed",
            "data": {"count": len(numbered), "elapsed_ms": timings["retrieval_ms"]},
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
                    "timings": timings, "route": _route_dict(route) if route else None,
                    "evidence": None,
                    "mode": mode, "personal_context": personal_context_info,
                },
            }
            return

        evidence_hits = numbered
        evidence_contents = contents
        verdict: EvidenceVerdict | None = None
        if self._evidence_gate is not None:
            evidence_started = time.perf_counter()
            verdict = self._evidence_gate.judge(
                question, evidence_hits, personal_context=personal_context_text,
            )
            timings["evidence_judge_ms"] = _elapsed_ms(evidence_started)
            timings["first_evidence_status"] = verdict.status
            if self._evidence_rescue_enabled:
                timings["rescue_triggered"] = False
            yield {
                "event": "evidence_completed",
                "data": {"verdict": verdict.to_dict(), "elapsed_ms": timings["evidence_judge_ms"]},
            }

            # 与非流式 answer() 同一个入口：只在第一次评审不足时救援一次
            if verdict.status == STATUS_INSUFFICIENT and self._evidence_rescue_enabled:
                evidence_hits, verdict = self._rescue(
                    question, domains_effective, evidence_hits, verdict,
                    personal_context_text, timings,
                )
                if timings.get("rescue_triggered"):
                    evidence_contents = None
            timings["final_evidence_status"] = verdict.status

            # 只在救援真正开启时才发这个事件：纯 Gate 路径保持原有的事件序列，
            # 避免给只开了 Evidence Gate 的调用方多塞一条重复结论。
            if self._evidence_rescue_enabled:
                yield {
                    "event": "evidence_rescue_completed",
                    "data": {
                        "rescue_triggered": timings.get("rescue_triggered"),
                        "rescue_retrieval_ms": timings.get("rescue_retrieval_ms"),
                        "rescue_evidence_ms": timings.get("rescue_evidence_ms"),
                        "verdict": verdict.to_dict(),
                        "sources": _source_dicts(evidence_hits),
                        "used_chunks": len(evidence_hits),
                    },
                }

            if verdict.status == STATUS_INSUFFICIENT:
                # 拒答路径与非流式 answer() 完全一致：不调生成 LLM，
                # sources 已经在上面吐给调用方了，由它负责打「仅供参考」标记。
                yield {
                    "event": "done",
                    "data": {
                        "answer": _insufficient_answer(verdict),
                        "used_chunks": len(evidence_hits),
                        "elapsed_ms": _elapsed_ms(total_started),
                        "timings": timings,
                        "route": _route_dict(route) if route else None,
                        "evidence": verdict.to_dict(),
                        "mode": mode, "personal_context": personal_context_info,
                    },
                }
                return

        partial = verdict is not None and verdict.status == STATUS_PARTIAL
        context = _build_context(evidence_hits, evidence_contents)
        if mode == MODE_ADVISOR:
            messages = _build_advisor_messages(
                question, context, self._vault_name,
                personal_context=personal_context_text or PERSONAL_CONTEXT_UNAVAILABLE_NOTE,
                partial=partial,
            )
        else:
            messages = _build_messages(question, context, self._vault_name, partial=partial)
        yield {"event": "generation_started", "data": {}}
        generation_started = time.perf_counter()

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

        timings["answer_generation_ms"] = _elapsed_ms(generation_started)
        yield {
            "event": "done",
            "data": {
                "answer": "".join(parts),
                "used_chunks": len(evidence_hits),
                "elapsed_ms": _elapsed_ms(total_started),
                "timings": timings,
                "route": _route_dict(route) if route else None,
                "evidence": verdict.to_dict() if verdict else None,
                "mode": mode, "personal_context": personal_context_info,
            },
        }

    def _collect(
        self, question: str, k: int, domains: list[str] | None = None,
    ) -> tuple[list[SearchResult], dict[int, str] | None]:
        """检索并整理最终进入 context 的 chunk，返回已按 chunk_id 去重的结果。

        domains=None 时是全库检索（历史行为，回归基线）；
        domains=[...] 时交给 Retriever 做 Chroma metadata 过滤（Scoped Retrieval）。

        返回 (hits, contents)：hits 决定引用编号和 sources（始终用原始 chunk），
        contents 是按 chunk_id 索引的清洗后正文，None 表示未开启清洗、直接用原文。

        没有任何加工开关时池子就是 k 条、下面两个分支都被跳过，等价于原始 Top-K 检索，
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
        pool = _dedup_by_chunk_id(self._retriever.search(question, top_k=pool_size, domains=domains))

        if self._reranker is not None:
            pool = self._reranker.rerank(question, pool)

        if not self._context_filter_enabled:
            return pool[:limit], None

        # 只挑会真正被编号的那几条，超出的候选不做无谓的清洗
        selection = prepare_context(pool, top_k=limit)
        hits = [chunk.hit for chunk in selection.selected]
        contents = {chunk.hit.chunk_id: chunk.context_content for chunk in selection.selected}
        return hits, contents


def _build_messages(
    question: str, context: str, vault_name: str, partial: bool = False,
) -> list[dict]:
    """拼发给 LLM 的 messages。answer() 和 answer_stream() 共用这一份，
    否则两条链路各自的 Prompt 写法早晚会有出入。

    partial=True 表示证据评审判定资料只覆盖了问题的部分方面，
    system prompt 追加「区分支持部分和缺依据部分」的硬性要求。
    """
    system = SYSTEM_PROMPT.format(vault_name=vault_name)
    if partial:
        system += PARTIAL_EVIDENCE_ADDENDUM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def _build_advisor_messages(
    question: str,
    context: str,
    vault_name: str,
    personal_context: str,
    partial: bool = False,
) -> list[dict]:
    """advisor 模式的 messages：三段式 Prompt + 强制区分三类信息来源。

    与 _build_messages 分开写，两者共用同一个 _build_context 拼编号资料，
    这样 [1][2] 和 sources 数组的对应关系不会因模式不同而出偏差。
    """
    system = SYSTEM_PROMPT_ADVISOR.format(vault_name=vault_name)
    if partial:
        system += ADVISOR_PARTIAL_EVIDENCE_ADDENDUM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": ADVISOR_USER_PROMPT_TEMPLATE.format(
            personal_context=personal_context, context=context, question=question,
        )},
    ]


def _normalize_mode(mode: str) -> str:
    """校验 mode 取值。API 层的 pydantic 已拦过一次，这里防的是直接调
    RAGEngine 的脚本和测试写错参数——拼错的字符串不该静默走错分支。"""
    if mode not in VALID_MODES:
        raise ValueError(f"mode 只能是 {'/'.join(sorted(VALID_MODES))}，当前：{mode!r}")
    return mode


def _insufficient_answer(verdict: EvidenceVerdict) -> str:
    """insufficient 的固定拒答文案。缺什么优先用 Judge 列的 unsupported_aspects
    （具体到方面），列不出来时退回 Judge 的一句话理由，保证拒答不是干巴巴一句。"""
    missing = "、".join(verdict.unsupported_aspects) if verdict.unsupported_aspects else verdict.reason
    return INSUFFICIENT_ANSWER_TEMPLATE.format(missing=missing or "知识库中未找到与问题实质相关的内容")


def _route_dict(route: RouteDecision | None) -> dict | None:
    """RouteDecision → API 响应 / 事件流里的普通 dict。"""
    if route is None:
        return None
    return {
        "mode": route.mode,
        "domains": route.domains,
        "confidence": route.confidence,
        "reason": route.reason,
        "origin": route.origin,
    }


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
