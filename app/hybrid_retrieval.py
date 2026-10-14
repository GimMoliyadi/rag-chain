"""Optional experimental BM25 + dense retrieval with Reciprocal Rank Fusion.

The lexical index is a read-only in-memory snapshot of rows previously loaded
from SQLite; refresh it by constructing a new instance after ingestion.
RRF output scores are *rank-fusion scores*, not cosine similarities. Keep this
opt-in until thresholds/evidence gates are calibrated for hybrid search.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from .retriever import Retriever, SearchResult

_WORDS = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """English tokens and overlapping Chinese bigrams, no optional packages."""
    out: list[str] = []
    for match in _WORDS.findall(text.lower()):
        if '\u3400' <= match[0] <= '\u9fff' and len(match) > 1:
            out.extend(match[i:i + 2] for i in range(len(match) - 1))
        else:
            out.append(match)
    return out


class BM25Index:
    """Sparse BM25 index over chunks; builds once, query uses postings only."""

    def __init__(self, rows: 'Iterable[Mapping[str, object]]', *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('BM25 expects k1 > 0 and 0 <= b <= 1')
        self.rows = [dict(row) for row in rows]
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, row in enumerate(self.rows):
            text = ' '.join(str(row.get(field) or '') for field in ('title', 'heading', 'content'))
            terms = tokenize(text)
            self.lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((i, count))
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.rows else 0.0

    def search(self, query: str, *, top_k: int = 20, domains: 'Sequence[str] | None' = None) -> list[tuple[dict, float]]:
        if top_k < 1:
            raise ValueError('top_k must be positive')
        n = len(self.rows)
        if not n:
            return []
        allowed = set(domains) if domains else None
        scores: dict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            postings = self.postings.get(token, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                if allowed is not None and self.rows[i].get('domain') not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = sorted(scores, key=lambda i: (-scores[i], int(self.rows[i]['chunk_id'])))[:top_k]
        return [(self.rows[i], scores[i]) for i in best]


def reciprocal_rank_fusion(
    ranked_lists: 'Iterable[Iterable[tuple[int, object]]]', *, top_k: int, rrf_k: int = 60,
) -> list[tuple[int, object, float]]:
    """Fuse (chunk_id, payload) lists using ranks, ignoring duplicates per list."""
    if top_k < 1 or rrf_k < 1:
        raise ValueError('top_k and rrf_k must be positive')
    scores: dict[int, float] = defaultdict(float)
    payloads: dict[int, object] = {}
    order: dict[int, int] = {}
    for ranked in ranked_lists:
        seen: set[int] = set()
        for identifier, payload in ranked:
            if identifier in seen:
                continue
            seen.add(identifier)
            rank = len(seen)
            scores[identifier] += 1.0 / (rrf_k + rank)
            if identifier not in payloads:
                payloads[identifier] = payload
                order[identifier] = len(order)
    ids = sorted(scores, key=lambda i: (-scores[i], order[i], i))[:top_k]
    return [(i, payloads[i], scores[i]) for i in ids]


class HybridRetriever:
    """Drop-in search interface for *opt-in* CLI experiments only.

    The dense retriever keeps its own model fingerprint check and domain
    fallback. Lexical rows should be SQLite's get_chunks_for_reindex output.
    """

    def __init__(self, dense: 'Retriever', rows: 'Iterable[Mapping[str, object]]', *, rrf_k: int = 60,
                 candidate_multiplier: int = 4):
        if rrf_k < 1 or candidate_multiplier < 1:
            raise ValueError('rrf_k and candidate_multiplier must be positive')
        self.dense = dense
        self.lexical = BM25Index(rows)
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.last_debug: dict = {}

    def search(self, question: str, top_k: int = 5, domains: 'Sequence[str] | None' = None) -> list['SearchResult']:
        if top_k < 1:
            raise ValueError('top_k must be positive')
        pool_size = top_k * self.candidate_multiplier
        dense_hits = self.dense.search(question, top_k=pool_size, domains=list(domains) if domains else None)
        # Match dense fallback semantics if the requested domain had no chunks.
        lexical_domains = None if self.dense.last_debug.get('domain_empty_fallback') else domains
        lexical_hits = self.lexical.search(question, top_k=pool_size, domains=lexical_domains)
        fused = reciprocal_rank_fusion(
            ([(hit.chunk_id, hit) for hit in dense_hits],
             [(int(row['chunk_id']), row) for row, _ in lexical_hits]),
            top_k=top_k, rrf_k=self.rrf_k,
        )
        from .retriever import SearchResult  # deferred: pure BM25 tests need no app dependencies
        results = []
        for _, entry, raw_score in fused:
            normalized = min(1.0, raw_score / (2.0 / (self.rrf_k + 1)))
            if isinstance(entry, SearchResult):
                results.append(replace(entry, score=normalized))
            else:
                row = entry
                results.append(SearchResult(
                    score=normalized, chunk_id=int(row['chunk_id']),
                    document_id=int(row['document_id']), chunk_index=int(row['chunk_index']),
                    title=str(row.get('title') or ''), file_path=str(row.get('file_path') or ''),
                    heading=str(row.get('heading') or ''), source=str(row.get('source') or ''),
                    content=str(row['content']), domain=str(row.get('domain') or ''),
                    source_type=str(row.get('source_type') or ''),
                ))
        self.last_debug = {
            'mode': 'hybrid_rrf', 'candidates': len(dense_hits) + len(lexical_hits),
            'dense_candidates': len(dense_hits), 'lexical_candidates': len(lexical_hits),
            'returned': len(results), 'rrf_k': self.rrf_k,
            'domain_empty_fallback': self.dense.last_debug.get('domain_empty_fallback', False),
            'diversity_applied': False,
        }
        return results
