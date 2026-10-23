"""Unit tests for information retrieval metrics and bootstrap statistics (Tasks P6.7, P6.8)."""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    dcg_at_k,
    f1_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.stats import paired_bootstrap_test


def test_precision_at_k_hand_calculated() -> None:
    # 2 relevant (grade 2 and 1), 2 non-relevant (0, 0)
    grades = [2, 1, 0, 0]
    assert precision_at_k(grades, k=1) == 1.0
    assert precision_at_k(grades, k=2) == 1.0
    assert precision_at_k(grades, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(grades, k=4) == 0.5
    assert precision_at_k([], k=3) == 0.0


def test_recall_at_k_hand_calculated() -> None:
    grades = [2, 0, 1, 0]
    total_relevant = 4
    assert recall_at_k(grades, total_relevant=total_relevant, k=1) == 0.25
    assert recall_at_k(grades, total_relevant=total_relevant, k=2) == 0.25
    assert recall_at_k(grades, total_relevant=total_relevant, k=3) == 0.50
    assert recall_at_k(grades, total_relevant=total_relevant, k=4) == 0.50
    assert recall_at_k([], total_relevant=0, k=2) == 1.0


def test_reciprocal_rank_and_mrr() -> None:
    assert reciprocal_rank([2, 0, 0]) == 1.0
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert reciprocal_rank([0, 0, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0, 0]) == 0.0

    mrr = mean_reciprocal_rank([[2, 0], [0, 1], [0, 0]])
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert mrr == pytest.approx(0.5)


def test_ndcg_at_k_bounds_and_ideal_ordering() -> None:
    # Ideal ordering gives perfect nDCG = 1.0
    ideal = [2, 2, 1, 0, 0]
    assert ndcg_at_k(ideal, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(ideal, k=3) == pytest.approx(1.0)

    # Inverted ranking gives lower score
    suboptimal = [0, 1, 2, 0, 2]
    ndcg_sub = ndcg_at_k(suboptimal, k=5)
    assert 0.0 < ndcg_sub < 1.0

    # All zeros
    assert ndcg_at_k([0, 0, 0], k=3) == pytest.approx(1.0)
    assert dcg_at_k([0, 0, 0], k=3) == 0.0


def test_f1_at_k() -> None:
    assert f1_at_k(1.0, 1.0) == 1.0
    assert f1_at_k(0.5, 0.5) == 0.5
    assert f1_at_k(0.0, 0.0) == 0.0
    assert f1_at_k(0.8, 0.4) == pytest.approx(2 * 0.8 * 0.4 / 1.2)


def test_paired_bootstrap_test() -> None:
    # Identical models
    a = [0.8, 0.85, 0.9, 0.75, 0.82]
    res_identical = paired_bootstrap_test(a, a, n_bootstraps=200)
    assert res_identical.mean_diff == 0.0
    assert not res_identical.is_significant

    # Superior model
    b = [0.4, 0.45, 0.5, 0.35, 0.42]
    res_superior = paired_bootstrap_test(a, b, n_bootstraps=500)
    assert res_superior.mean_diff > 0.35
    assert res_superior.ci_lower > 0.0
    assert res_superior.is_significant
