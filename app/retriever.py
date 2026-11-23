"""检索器：把一个问题变成 Top-K 相关 chunk。

这条链路是 RAG 的"R"（Retrieval）部分，也是整个系统里最该先验证的一环——
检索错了，LLM 再强也答不对。所以 search.py 可以独立跑它，不用碰 LLM。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import TYPE_CHECKING

from .context_filter import judge_chunk
from .embedding import BaseEmbedding
from .vectorstore import ChromaStore

if TYPE_CHECKING:
    from .reranker import CrossEncoderReranker, RerankStats


@dataclass
class SearchResult:
    """一条检索命中。score 是原始向量余弦相似度，范围 0~1。

    返回顺序由向量相似度与标题/章节词面信号的内部混合排序决定，
    但不把内部排名分数写进这个既有字段，避免调用方误把它当成余弦值。
    """

    score: float
    chunk_id: int
    document_id: int
    chunk_index: int
    title: str
    file_path: str
    heading: str
    source: str
    content: str
    domain: str = ""         # 知识域（阶段二 metadata 回填之前是空串）
    source_type: str = ""
    rerank_score: float | None = None  # 问答来源保留既有精排分数口径；search.score 不变


class EmbeddingMismatchError(RuntimeError):
    """向量库里的向量不是当前 embedding 模型算出来的。

    不同模型的向量不在同一个空间里，混着用算出来的相似度是随机噪声，
    而且不会报任何错——这正是最危险的一种失败方式。所以这里必须显式抛错。
    """


# 混合检索先取一批候选，再按向量相似度和标题/章节词面重排。
# 50 覆盖真实索引中已经观察到的「精确章节排在向量 Top-20 之外」的情况，
# 仍远小于全库 14k chunk，不把每次查询变成全库扫描。
HYBRID_CANDIDATE_K = 50
# 多样性开启时先取多少条候选。候选池太小，文档限额会把结果压到不足 Top-K；
# 20 是"每篇最多 2 条时，Top-5 还有 4 倍冗余"的量级，不是精调值。
DIVERSITY_CANDIDATE_K = 20

# 词面信号不是第二个向量空间；但问题完整点名的标题/章节是比粗排余弦更强
# 的定位证据。略高于半权重，足以让「Shor 算法 / 实验进展」超过同文档的定义章节；
# 混合结果最终取 max(vector, blend)，词面信号不会把向量命中降权。
LEXICAL_BOOST_WEIGHT = 0.6

_NON_WORD = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
_STANDARD_CODE = re.compile(r"(?i)(?<![A-Za-z0-9])GB\s*(\d{3,})\s*[-–]\s*(\d{2,})(?![A-Za-z0-9])")
_STANDARD_REFERENCE = re.compile(r"(?i)(?<![A-Za-z0-9])GB\s*([0-9]{3,})")
_CROSS_STANDARD_QUESTION = re.compile(
    r"哪几部\s*标准|哪(?:些|几[部项个份]?)\s*(?:国家|行业)标准"
    r"|(?:涉及|适用|遵循|经过)\s*哪(?:些|几[部项个份]?)\s*标准"
)
_PATH_REFERENCE = re.compile(r"(?:《|[\"'`])([^《》\"'`\r\n]+\.md)(?:》|[\"'`])", re.IGNORECASE)

_VERSION_LEAD_IN = re.compile(r"^\s*(?:(?:标准|笔记)?\s*(?:中|里|回答)?\s*[:：,，]\s*)+")


class Retriever:
    """用 embedding + 向量库做相似度检索。"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        store: ChromaStore,
        default_top_k: int = 5,
        diversity_enabled: bool = False,
        max_chunks_per_document: int = 2,
        reranker: CrossEncoderReranker | None = None,
        reranker_candidate_top_k: int = 50,
    ):
        self._embedding = embedding
        self._store = store
        self._default_top_k = default_top_k
        self._diversity_enabled = diversity_enabled
        self._max_chunks_per_document = max_chunks_per_document
        self._reranker = reranker
        self._reranker_candidate_top_k = reranker_candidate_top_k
        # 最近一次 search 的调试信息。search 的返回值必须保持 list[SearchResult]
        # 不变（eval / API / 测试都依赖它），domain_empty_fallback 这类过程信息
        # 只能挂在旁边，不能改返回类型。
        self.last_debug: dict = {}

    @property
    def rerank_stats(self) -> RerankStats | None:
        return self._reranker.stats if self._reranker is not None else None

    def search(self, question: str, top_k: int | None = None, domains: list[str] | None = None) -> list[SearchResult]:
        """检索与 question 最相关的 top_k 条 chunk，按混合相关性从高到低排列。

        domains=None 或 [] 时是全库检索，行为与历史版本完全一致（回归基线）；
        domains=[...] 时用 Chroma 的 metadata 过滤只在指定知识域里找。
        指定的域在库里一条 chunk 都没有时（如 fiction），不做空返回，
        回退全库检索，并记录 domain_empty_fallback=True。
        点名标准版本或用引号指定笔记相对路径时，只在指定来源内找，不用其他版本补位。

        返回对象的 ``score`` 始终是原始向量余弦相似度；它不一定随返回顺序单调，
        因为标题/章节词面信号只参与内部排序。
        """
        k = top_k or self._default_top_k
        count = self._store.count()
        if count == 0:
            self.last_debug = {"scoped": False, "domains": [], "domain_empty_fallback": False,
                               "diversity_applied": False, "candidates": 0, "returned": 0}
            return []

        # 库非空时才检查指纹：还没 ingest 过时空库本来就没有可比对的指纹
        _check_index_matches(self._embedding, self._store)

        scoped = bool(domains)
        where = {"domain": {"$in": list(domains)}} if scoped else None
        # 词面重排必须看到向量粗排前 50 条，否则精确章节已经被 Chroma 截掉
        # 就没有机会回来。多取候选只影响一次 Chroma query，不改变返回条数。
        candidate_k = max(k, HYBRID_CANDIDATE_K)
        if self._reranker is not None:
            candidate_k = max(candidate_k, self._reranker_candidate_top_k)
        if self._diversity_enabled:
            candidate_k = max(candidate_k, DIVERSITY_CANDIDATE_K)
        scopes = _source_scopes(question)
        if scopes:
            return self._search_scoped(scopes, k, domains)

        question_vector = self._embedding.embed_query(question)
        rows = self._store.query(question_vector, n_results=min(candidate_k, count), where=where)

        fallback = False
        if scoped and not rows:
            # 指定的域是 0 chunk（比如 fiction 只有 .txt 源、从未入库）。
            # 过滤本身没错，错的是"拿一个空结果当答案"，所以退回全库。
            fallback = True
            rows = self._store.query(question_vector, n_results=min(candidate_k, count))
            where = None

        expanded_documents = []
        if self._reranker is not None:
            # 已观察到：问题点名 Shor 算法时定义在 Top-1，但实验节在
            # 全库 Top-50 以外。只重排全库候选救不回来；补查点名文档。
            normalized = _normalize_for_match(question)
            matches: dict[int, float] = {}
            for row in rows:
                match = _document_match_score(normalized, row)
                if match:
                    doc_id = int(row["document_id"])
                    matches[doc_id] = max(matches.get(doc_id, 0.0), match)
            expanded_documents = sorted(matches, key=matches.get, reverse=True)[:3]
            by_id = {int(row["chunk_id"]): row for row in rows}
            for doc_id in expanded_documents:
                doc_where = {"document_id": doc_id}
                if where:
                    doc_where = {"$and": [where, doc_where]}
                for row in self._store.query(question_vector, n_results=min(candidate_k, count), where=doc_where):
                    by_id.setdefault(int(row["chunk_id"]), row)
            rows = list(by_id.values())

        reference_sources: dict[int, str] = {}
        if _CROSS_STANDARD_QUESTION.search(question):
            reference_rows, reference_sources = self._standard_reference_candidates(
                question_vector, min(candidate_k, count), where,
            )
            existing_ids = {int(row["chunk_id"]) for row in rows}
            rows.extend(row for row in reference_rows if int(row["chunk_id"]) not in existing_ids)

        candidates = len(rows)
        # 先用精确标题/章节信号把候选排好，再做同文档限额。若先限额，
        # 长文档的前两个向量命中可能把真正被问题点名的章节直接丢掉。
        rows = _hybrid_rank(question, rows)
        rerank_fallback = False
        if self._reranker is not None:
            before = self._reranker.stats.fallback
            ranked = self._reranker.rerank(question, [_to_hit(row) for row in rows])
            rerank_fallback = self._reranker.stats.fallback > before
            # CrossEncoderReranker 的 score 是精排分数（供旧 eval 使用）。
            # 公共检索接口只采用其顺序，仍从原始 distance 返回余弦值。
            by_id = {int(row["chunk_id"]): row for row in rows}
            if not rerank_fallback:
                for hit in ranked:
                    by_id[hit.chunk_id]["_rerank_score"] = hit.score
            rows = [by_id[hit.chunk_id] for hit in ranked]
            if not rerank_fallback and len(expanded_documents) == 1:
                _keep_named_document(rows, expanded_documents[0], min(k, self._default_top_k))
        if reference_sources:
            rows = _prioritize_standard_references(rows, reference_sources)
        if self._diversity_enabled:
            rows = _cap_per_document(rows, k, self._max_chunks_per_document)
        else:
            rows = rows[:k]

        hits = [_to_hit(row) for row in rows]
        self.last_debug = {
            "scoped": scoped,
            "domains": list(domains) if domains else [],
            "domain_empty_fallback": fallback,
            "diversity_applied": self._diversity_enabled,
            "candidates": candidates,
            "returned": len(hits),
            "reranker_applied": self._reranker is not None and not rerank_fallback,
            "reranker_fallback": rerank_fallback,
            "expanded_documents": expanded_documents,
        }
        return hits

    def _standard_reference_candidates(
        self, question_vector: list[float], candidate_k: int, where: dict | None,
    ) -> tuple[list[dict], dict[int, str]]:
        standards_by_path = {
            path: standard for path in self._store.document_paths(where=where)
            if (standard := _standard_family(path)) is not None
        }
        known_standards = set(standards_by_path.values())
        if len(known_standards) < 2:
            return [], {}

        file_filter = {"file_path": {"$in": list(standards_by_path)}}
        where = {"$and": [where, file_filter]} if where else file_filter
        rows = self._store.query(
            question_vector, n_results=candidate_k, where=where,
            where_document={"$or": [{"$contains": prefix} for prefix in ("GB", "Gb", "gB", "gb")]},
        )
        linked_rows = []
        reference_sources = {}
        for row in rows:
            source = standards_by_path[row["file_path"]]
            references = {f"GB{number}" for number in _STANDARD_REFERENCE.findall(row["content"])}
            if (references & known_standards) - {source} and not judge_chunk(row["content"]).is_navigation:
                linked_rows.append(row)
                reference_sources[int(row["chunk_id"])] = source

        if len(set(reference_sources.values())) < 2:
            return [], {}
        return linked_rows, reference_sources

    def _search_scoped(
        self,
        scopes: list[tuple[str, str, str]],
        k: int,
        domains: list[str] | None,
    ) -> list[SearchResult]:
        """Keep each requested subquestion inside its exact source and give it a slot."""
        paths = self._store.document_paths()
        groups = []
        candidates = 0
        fallbacks = self._reranker.stats.fallback if self._reranker else 0
        for kind, scope, query in scopes:
            if kind == "path":
                selected = [path for path in paths if _canonical_path(path) == scope]
            else:
                selected = [
                    path for path in paths
                    if any(_standard_id(part) == scope for part in _canonical_path(path).split("/"))
                ]
            if not selected:
                continue

            where = {"file_path": {"$in": selected}}
            if domains:
                where = {"$and": [{"domain": {"$in": domains}}, where]}
            rows = self._store.query(self._embedding.embed_query(query), n_results=100, where=where)
            candidates += len(rows)
            hits = [_to_hit(row) for row in rows]
            if self._reranker and hits:
                hits = self._rank_scoped(query, hits)
            groups.append(hits)

        seen = set()
        merged = []
        for depth in zip_longest(*groups):
            for hit in depth:
                if hit is not None and hit.chunk_id not in seen:
                    seen.add(hit.chunk_id)
                    merged.append(hit)
        if self._diversity_enabled:
            per_document = {}
            limited = []
            for hit in merged:
                count = per_document.get(hit.document_id, 0)
                if count < self._max_chunks_per_document:
                    limited.append(hit)
                    per_document[hit.document_id] = count + 1
            merged = limited
        hits = merged[:k]
        fallback = self._reranker is not None and self._reranker.stats.fallback > fallbacks
        self.last_debug = {
            "scoped": True,
            "domains": list(domains or []),
            "domain_empty_fallback": False,
            "diversity_applied": self._diversity_enabled,
            "candidates": candidates,
            "returned": len(hits),
            "reranker_applied": self._reranker is not None and bool(hits) and not fallback,
            "reranker_fallback": fallback,
            "expanded_documents": [],
            "source_scopes": list(dict.fromkeys(scope for _, scope, _ in scopes)),
        }
        return hits

    def _rank_scoped(self, query: str, hits: list[SearchResult]) -> list[SearchResult]:
        assert self._reranker is not None
        original = {hit.chunk_id: hit for hit in hits}
        before = self._reranker.stats.fallback
        ranked = self._reranker.rerank(query, hits)
        if self._reranker.stats.fallback > before:
            return hits
        # A section heading can answer the question more directly than a long
        # passage. Promote it only if the body also placed it in the first three.
        with_heading = [replace(hit, content="") for hit in hits if hit.heading]
        if len(ranked) >= 3 and with_heading:
            heading = self._reranker.rerank(query, with_heading, top_k=1)
            if self._reranker.stats.fallback == before and heading[0].chunk_id == ranked[2].chunk_id:
                ranked[1], ranked[2] = ranked[2], ranked[1]
        return [replace(original[hit.chunk_id], rerank_score=hit.score) for hit in ranked]


def _canonical_path(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


def _standard_id(value: str) -> str | None:
    match = _STANDARD_CODE.fullmatch(value.removesuffix(".md"))
    return f"GB{match[1]}-{match[2]}" if match else None


def _standard_family(path: str) -> str | None:
    for part in _canonical_path(path).split("/"):
        edition = _standard_id(part)
        if edition:
            return edition.split("-", 1)[0]
    return None


def _prioritize_standard_references(rows: list[dict], sources: dict[int, str]) -> list[dict]:
    first_by_standard: dict[str, dict] = {}
    for row in rows:
        standard = sources.get(int(row["chunk_id"]))
        if standard is not None:
            first_by_standard.setdefault(standard, row)
    primary = list(first_by_standard.values())
    primary_ids = {int(row["chunk_id"]) for row in primary}
    return primary + [row for row in rows if int(row["chunk_id"]) not in primary_ids]


def _source_scopes(question: str) -> list[tuple[str, str, str]]:
    paths = [path for match in _PATH_REFERENCE.finditer(question) if "/" in (path := _canonical_path(match[1]))]
    if paths:
        return [("path", path, question) for path in dict.fromkeys(paths)]

    scopes = []
    editions = list(_STANDARD_CODE.finditer(question))
    for i, match in enumerate(editions):
        end = editions[i + 1].start() if i + 1 < len(editions) else len(question)
        text = _VERSION_LEAD_IN.sub("", question[match.end():end]).strip()
        queries = [clause.strip() for clause in re.split(r"[？?；;。]", text) if clause.strip()]
        scope = f"GB{match[1]}-{match[2]}"
        scopes.extend(("edition", scope, clause) for clause in (queries or [question]))
    return scopes


def _keep_named_document(rows: list[dict], document_id: int, top_k: int) -> None:
    if top_k < 2:
        return
    for position, row in enumerate(rows):
        if int(row["document_id"]) == document_id:
            if position >= top_k:
                # 保留精排首位，让点名笔记得分最高的片段也能进入问答上下文。
                rows.insert(1, rows.pop(position))
            return


def _cap_per_document(rows: list[dict], top_k: int, max_per_document: int) -> list[dict]:
    """同文档 chunk 限额：每个 document_id 最多保留 max_per_document 条，直到凑满 top_k。

    解决的实测问题（baseline #22）：长章节切成多个 chunk 后每段都高分，
    Top-5 被"同一段话的 5 个切片"占满。限额后名额让给次优文档，
    喂给 LLM 的 context 才有多样性。不做 MMR：先用最简单的文档限额验证
    它能不能解决真实失败案例，不够再谈更复杂的。
    """
    kept: list[dict] = []
    per_document: dict[int, int] = {}
    for row in rows:
        document_id = int(row["document_id"])
        if per_document.get(document_id, 0) >= max_per_document:
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        kept.append(row)
        if len(kept) >= top_k:
            break
    return kept


def _to_hit(row: dict) -> SearchResult:
    return SearchResult(
        # _rank_score 只用于候选排序；SearchResult.score 的既有契约是
        # 原始向量余弦相似度，直接检索的 MCP/CLI/API 都会把它展示给调用方。
        score=_to_similarity(row["distance"]),
        chunk_id=int(row["chunk_id"]),
        document_id=int(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        title=str(row.get("title", "")),
        file_path=str(row.get("file_path", "")),
        heading=str(row.get("heading", "")),
        source=str(row.get("source", "")),
        content=row["content"],
        domain=str(row.get("domain", "")),
        source_type=str(row.get("source_type", "")),
        rerank_score=row.get("_rerank_score"),
    )


def _hybrid_rank(question: str, rows: list[dict]) -> list[dict]:
    """用标题/章节的精确词面信号修正向量粗排。

    向量检索擅长找同义表达，但对「问题明确点名某篇文档的某一节」并不
    可靠：同一篇文档的定义、参考资料等章节可能占据前几名，真正的目标
    章节反而落到 Top-20 之外。这里不做全库关键词搜索，只在已取出的
    候选里计算一个保守的 metadata 信号：只有一个完整的标题或章节短语
    出现在问题中才算命中。不能把「问题和标题共享三个常见字」当成证据，
    否则长问题会把一批不相干的标题错误地抬高。

    无词面信号的候选保持原始向量分数；有信号时用略高于半权重的内部混合分数。
    该分数只存在于候选行的 ``_rank_score``，不会改写 SearchResult.score。
    """
    if not rows:
        return rows
    normalized_question = _normalize_for_match(question)
    if not normalized_question:
        for row in rows:
            row.pop("_rank_score", None)
        return rows

    candidates: list[tuple[float, float, int, dict]] = []
    for position, row in enumerate(rows):
        vector_score = _to_similarity(row["distance"])
        title_score = _document_match_score(normalized_question, row)
        heading_score = _field_match_score(normalized_question, row.get("heading", ""))
        # 路径通常包含知识域、编号和通用目录词；它不是用户可见的文档
        # 标题，不能用最长公共子串参与排序，否则会制造大量假命中。
        # 标题缺失时，完整文件名仍可作为弱补充，但只匹配 basename。
        # 标题本身就是定位信号，不能因为章节为空而再乘一次 0.7；
        # 标题和章节都命中时再叠加一点具体度，最多封顶 1。
        lexical_score = max(
            title_score,
            heading_score,
            min(1.0, title_score + 0.3 * heading_score),
        )
        candidates.append((vector_score, lexical_score, position, row))

    # 一个文档通常被切成多个 chunk，标题会在每个 chunk 的 metadata 中重复。
    # 词面命中只给该文档最有说服力的一条 chunk 加权，其余 chunk 保留向量分数，
    # 避免一个精确标题把 Top-K 全部占满，导致同一问题点名的另一篇文档消失。
    best_lexical: dict[int, tuple[float, float, int]] = {}
    for vector_score, lexical_score, position, row in candidates:
        if lexical_score <= 0:
            continue
        document_id = int(row["document_id"])
        previous = best_lexical.get(document_id)
        current = (lexical_score, vector_score, -position)
        if previous is None or current > previous:
            best_lexical[document_id] = current

    ranked: list[tuple[float, int, dict]] = []
    for vector_score, lexical_score, position, row in candidates:
        rank_score = vector_score
        document_id = int(row["document_id"])
        selected = best_lexical.get(document_id)
        if lexical_score > 0 and selected == (lexical_score, vector_score, -position):
            blended_score = (
                (1.0 - LEXICAL_BOOST_WEIGHT) * vector_score
                + LEXICAL_BOOST_WEIGHT * lexical_score
            )
            # 词面信号是 boost，不是惩罚项。短标题的精确命中分数可能
            # 低于很高的向量分数；这种情况下必须保留原始向量排序位置。
            rank_score = max(vector_score, blended_score)
        row["_rank_score"] = rank_score
        ranked.append((rank_score, position, row))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [row for _, _, row in ranked]


def _normalize_for_match(value: object) -> str:
    """折叠大小写、空白和标点，保留中英文和数字供精确匹配。"""
    return _NON_WORD.sub("", str(value).lower())


def _document_match_score(question: str, row: dict) -> float:
    title_score = _field_match_score(question, row.get("title", ""))
    if title_score:
        return title_score
    file_name = str(row.get("file_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
    return _field_match_score(question, file_name.removesuffix(".md"))


def _field_match_score(question: str, field: object) -> float:
    """返回完整字段短语在问题中的精确匹配强度。

    只接受完整字段命中，不接受最长公共子串：问题越长，随机共享几个字的
    机会越大，模糊的局部匹配反而会把错误文档排到前面。中文至少三个字符；
    两字标题/章节必须整段命中，ASCII 实体（Shor、MVP）至少两个字符。
    完整命中仍按短语长度给一点区分度：较长、较具体的标题比其中包含的
    短标题更有定位价值。
    """
    raw_field = str(field)
    variants = [_normalize_for_match(raw_field)]
    # 标题常把中文名和英文释名放在括号里。问题只写中文名时，
    # 括号里的译名不应阻断完整中文标题的匹配。
    without_parenthetical = re.sub(r"\([^)]*\)|（[^）]*）", "", raw_field)
    variants.append(_normalize_for_match(without_parenthetical))

    best = 0.0
    for normalized_field in variants:
        if len(normalized_field) < 2:
            continue
        minimum = 3 if any("\u3400" <= char <= "\u9fff" for char in normalized_field) else 2
        if len(normalized_field) < minimum or normalized_field not in question:
            continue
        # 0.5 是完整命中的最低信号，最多随短语长度升到 1.0；
        # 不让一个两字章节压过更具体的长标题，但也不抹掉短标题的价值。
        best = max(best, 0.5 + 0.5 * min(1.0, len(normalized_field) / 8.0))
    return best


def _to_similarity(distance: float) -> float:
    """把余弦距离转成相似度。

    余弦距离 range 是 [0, 2]，相似度 = 1 - distance，范围 [-1, 1]。
    这里把负值截断到 0，方便展示和做阈值过滤。
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


_REBUILD_HINT = "请运行 `python ingest.py --rebuild-vectors` 重建向量库。"


def _check_index_matches(embedding: BaseEmbedding, store: ChromaStore) -> None:
    """确认向量库是用当前模型建的。对不上就报错，绝不静默继续。

    典型事故：昨天用 bge-small 建的索引，今天把 .env 换成 bge-m3，
    但 Chroma 里的向量没动。这时检索数字还在返回，看着一切正常，
    实际上每一组相似度都是两个无关向量空间的巧合重合。
    """
    info = store.read_index_info()
    if info is None:
        raise EmbeddingMismatchError(
            "Embedding model mismatch. Please rebuild vector index. "
            f"向量库 {store.count()} 条向量没有模型指纹（可能是旧版代码建的），"
            f"当前模型是 {embedding.model_name}。{_REBUILD_HINT}"
        )
    if info != embedding.fingerprint:
        raise EmbeddingMismatchError(
            "Embedding model mismatch. Please rebuild vector index. "
            f"向量库是用 {info.get('model')} (dim={info.get('dimensions')}) 建的，"
            f"当前模型是 {embedding.model_name} (dim={embedding.dimensions})。{_REBUILD_HINT}"
        )
