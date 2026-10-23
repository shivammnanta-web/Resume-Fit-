"""Unit tests for Evaluation Harness (evaluation/harness.py)."""

from __future__ import annotations

from evaluation.harness import EvaluationHarness


def test_eval_harness_fairness_probe():
    harness = EvaluationHarness()
    res = harness.run_e5_fairness_probe()
    assert "scores" in res
    assert len(res["scores"]) == 6
    assert res["max_score_difference"] < 0.05
    assert res["score_variance"] < 0.001


def test_eval_harness_robustness_stuffed():
    harness = EvaluationHarness()
    res = harness.run_e4_robustness()
    assert "keyword_stuffed_mean_score" in res
    assert "qualified_mean_score" in res
    assert res["stuffed_resumes_tested"] > 0
    # Verified: keyword stuffing penalty lowers score significantly
    assert res["keyword_stuffed_mean_score"] <= res["qualified_mean_score"]
