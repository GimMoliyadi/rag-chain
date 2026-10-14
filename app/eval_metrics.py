"""Dependency-free, document-level ranking metrics for retrieval experiments.

Expected/retrieved document IDs must use the same canonical representation.
Repeated chunks of one document count once: retrieval is evaluated at the
*document* level, not by treating the same source as several independent hits.

These functions never judge the quality of annotations. Only use verified qrels
for externally reported measurements.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def document_ranking_metrics(
    relevance: Mapping[str, float],
    ranked_document_ids: Iterable[str],
    cutoffs: Iterable[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute hit rate, recall, precision and nDCG at each document cutoff.

    ``relevance`` maps document IDs to nonnegative relevance grades (0 =
    irrelevant); binary annotations work as grades of 1. IDs duplicated by
    multiple retrieved chunks retain their *first* occurrence. Recall divides
    by all relevant documents; precision divides by K, even if fewer than K
    results are returned. nDCG uses gain ``2**grade - 1`` and log2 discount.

    Queries with no positive relevance judgement are not scoreable, and must
    be reported separately as coverage/abstention probes.
    """
    grades = {doc: float(grade) for doc, grade in relevance.items()}
    if any(not math.isfinite(grade) or grade < 0 for grade in grades.values()):
        raise ValueError("Relevance grades must be finite and nonnegative")
    positive = {doc: grade for doc, grade in grades.items() if grade > 0}
    if not positive:
        raise ValueError("At least one positively relevant document is required")

    ks = tuple(cutoffs)
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in ks):
        raise ValueError("Cutoffs must be positive integers")
    ranked = list(dict.fromkeys(ranked_document_ids))
    ideal_grades = sorted(positive.values(), reverse=True)

    def dcg(scores: Iterable[float]) -> float:
        return sum((2.0 ** grade - 1.0) / math.log2(rank + 1)
                   for rank, grade in enumerate(scores, start=1))

    scores: dict[str, float] = {}
    for k in ks:
        observed = [positive.get(doc, 0.0) for doc in ranked[:k]]
        n_relevant = sum(grade > 0 for grade in observed)
        ideal = dcg(ideal_grades[:k])
        scores[f"hit_at_{k}"] = float(n_relevant > 0)
        scores[f"doc_recall_at_{k}"] = n_relevant / len(positive)
        scores[f"precision_at_{k}"] = n_relevant / k
        scores[f"ndcg_at_{k}"] = dcg(observed) / ideal if ideal else 0.0
    return scores
