"""RAG 编排：检索 → 拼 Prompt → 调 LLM → 整理引用。

这是把前面所有零件串起来的地方，也是比赛答辩时最该讲的一个文件。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import BaseLLM
from .retriever import Retriever, SearchResult

# 引用编号上限：超过这个数的检索结果不再参与编号，避免 Prompt 过长
MAX_CITATIONS = 8

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
    """返回给前端的一条引用来源。"""

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

    def __init__(self, retriever: Retriever, llm: BaseLLM, top_k: int = 5, vault_name: str = "Obsidian"):
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._vault_name = vault_name

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """完整问答：检索 → 拼 Prompt → 调 LLM → 返回答案和引用。"""
        hits = self._retriever.search(question, top_k=top_k or self._top_k)

        # 按 chunk_id 去重保序，避免同一 chunk 拿到两个引用编号
        deduped = _dedup_by_chunk_id(hits)
        numbered = deduped[:MAX_CITATIONS]

        if not numbered:
            return RAGAnswer(
                answer="知识库中目前没有足够的信息回答这个问题。可能原因：还没有 ingest 任何文档，"
                "或者你的问题在知识库里找不到对应内容。可以先调用 POST /api/ingest 导入文档。",
                sources=[],
                used_chunks=0,
            )

        context = _build_context(numbered)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(vault_name=self._vault_name)},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
        ]
        answer = self._llm.chat(messages)

        return RAGAnswer(
            answer=answer,
            sources=[
                Source(
                    index=i + 1,
                    title=h.title,
                    file_path=h.file_path,
                    heading=h.heading,
                    score=round(h.score, 4),
                    content=h.content,
                )
                for i, h in enumerate(numbered)
            ],
            used_chunks=len(numbered),
        )


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


def _build_context(hits: list[SearchResult]) -> str:
    """把检索结果拼成带编号的参考资料文本，编号和返回给前端的 sources 严格对应。"""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        content = hit.content
        if len(content) > MAX_CONTEXT_CHUNK_CHARS:
            content = content[:MAX_CONTEXT_CHUNK_CHARS] + "…"
        heading = f" / {hit.heading}" if hit.heading else ""
        blocks.append(
            f"[{i}] 文档：{hit.title}{heading}\n"
            f"     文件：{hit.file_path}\n"
            f"     内容：{content}"
        )
    return "\n\n".join(blocks)
