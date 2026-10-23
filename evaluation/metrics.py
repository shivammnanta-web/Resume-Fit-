"""Information Retrieval & Ranking Evaluation Metrics (Task P6.7).

Implemented from mathematical first principles with verified bounds:
- Precision@K (P@K)
- Recall@K (R@K)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)
- F1@K
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(
    relevance_grades: Sequence[int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """Compute Precision@K.

    Fraction of top-K retrieved items that are relevant (grade >= threshold).
    """
    if k <= 0:
        return 0.0
    top_k = relevance_grades[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for g in top_k if g >= relevance_threshold)
    return relevant_count / k


def recall_at_k(
    relevance_grades: Sequence[int],
    total_relevant: int,
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """Compute Recall@K.

    Fraction of all relevant items that are retrieved in the top-K.
    """
    if total_relevant <= 0:
        return 1.0 if not any(g >= relevance_threshold for g in relevance_grades) else 0.0
    if k <= 0:
        return 0.0
    top_k = relevance_grades[:k]
    retrieved_relevant = sum(1 for g in top_k if g >= relevance_threshold)
    return min(1.0, retrieved_relevant / total_relevant)


def reciprocal_rank(
    relevance_grades: Sequence[int],
    relevance_threshold: int = 1,
) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant item)."""
    for idx, grade in enumerate(relevance_grades, start=1):
        if grade >= relevance_threshold:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(
    query_rankings: Sequence[Sequence[int]],
    relevance_threshold: int = 1,
) -> float:
    """Compute Mean Reciprocal Rank across multiple queries."""
    if not query_rankings:
        return 0.0
    rr_scores = [reciprocal_rank(q, relevance_threshold=relevance_threshold) for q in query_rankings]
    return sum(rr_scores) / len(rr_scores)


def dcg_at_k(relevance_grades: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain at rank K using standard exponential gain."""
    top_k = relevance_grades[:k]
    score = 0.0
    for idx, rel in enumerate(top_k, start=1):
        gain = (2.0**rel) - 1.0
        discount = math.log2(idx + 1)
        score += gain / discount
    return score


def ndcg_at_k(relevance_grades: Sequence[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if k <= 0 or not relevance_grades:
        return 0.0

    actual_dcg = dcg_at_k(relevance_grades, k)
    ideal_relevances = sorted(relevance_grades, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0.0:
        return 1.0 if actual_dcg == 0.0 else 0.0

    return min(1.0, actual_dcg / ideal_dcg)


def f1_at_k(precision: float, recall: float) -> float:
    """Compute harmonic mean of Precision and Recall."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)
