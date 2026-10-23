"""Unit test for skill extraction evaluation harness."""

from __future__ import annotations

from evaluation.eval_skills import evaluate_dev_set


def test_evaluate_dev_set_metrics() -> None:
    """Verify that skill extraction reaches high Precision, Recall, and F1 across the 20 gold resumes."""
    summary = evaluate_dev_set()

    assert summary.total_resumes >= 20
    assert summary.total_tp > 250

    # Macro and micro F1 should comfortably exceed 80%
    assert summary.micro_f1 >= 0.85
    assert summary.macro_f1 >= 0.85
    assert summary.micro_recall >= 0.90
    assert summary.macro_recall >= 0.90
