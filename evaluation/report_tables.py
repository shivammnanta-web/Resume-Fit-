"""Report tables and LaTeX/Markdown formatting for evaluation results (Task P6.16).

Transforms raw experimental results from evaluation.harness into publication-ready
Markdown and JSON artifacts stored in evaluation/results/latest/.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

from evaluation.harness import EvaluationHarness

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evaluation/results/latest")


def format_e1_markdown_table(e1_data: dict[str, Any]) -> str:
    """Format E1 model comparison results into Markdown table."""
    metrics = e1_data["metrics"]
    lines = [
        "### Table 1: Retrieval Model Comparison on Held-Out Test Set (35 JDs, 350 Pairs)",
        "",
        "| Retrieval Architecture | P@1 | P@3 | R@3 | MRR | nDCG@3 | nDCG@5 |",
        "|:-----------------------|:---:|:---:|:---:|:---:|:------:|:------:|",
    ]
    for model_name, m in metrics.items():
        lines.append(
            f"| **{model_name}** | {m['P@1']:.3f} | {m['P@3']:.3f} | {m['R@3']:.3f} | {m['MRR']:.3f} | {m['nDCG@3']:.3f} | {m['nDCG@5']:.3f} |"
        )

    sig_lex = e1_data.get("significance_vs_lexical", {})
    sig_sem = e1_data.get("significance_vs_semantic", {})

    lines.append("")
    lines.append(f"- **Hybrid vs Lexical**: Diff={sig_lex.get('mean_diff', 0):+.3f}, 95% CI=[{sig_lex.get('ci_lower', 0):.3f}, {sig_lex.get('ci_upper', 0):.3f}], p={sig_lex.get('p_value_bootstrap', 1):.4f} (Significant: {sig_lex.get('is_significant', False)})")
    lines.append(f"- **Hybrid vs Semantic**: Diff={sig_sem.get('mean_diff', 0):+.3f}, 95% CI=[{sig_sem.get('ci_lower', 0):.3f}, {sig_sem.get('ci_upper', 0):.3f}], p={sig_sem.get('p_value_bootstrap', 1):.4f} (Significant: {sig_sem.get('is_significant', False)})")
    return "\n".join(lines)


def format_e2_markdown_table(e2_data: dict[str, Any]) -> str:
    """Format E2 alpha sweep results into Markdown table."""
    curve = e2_data["dev_sweep_curve"]
    lines = [
        "### Table 2: Linear Fusion Weight Parameter Sweep (α·Lexical + (1-α)·Semantic) on Dev Set",
        "",
        "| α (Lexical Weight) | β (Semantic Weight) | Dev Set nDCG@5 | Note |",
        "|:------------------:|:-------------------:|:--------------:|:-----|",
    ]
    opt_alpha = e2_data["optimal_alpha_dev"]
    for row in curve:
        is_opt = (row["alpha"] == opt_alpha)
        tag = "**Optimal (Chosen)**" if is_opt else "Sub-optimal"
        lines.append(f"| {row['alpha']:.1f} | {row['beta']:.1f} | {row['dev_nDCG@5']:.4f} | {tag} |")
    return "\n".join(lines)


def format_e3_markdown_table(e3_data: dict[str, Any]) -> str:
    """Format E3 ablation results into Markdown table."""
    lines = [
        "### Table 3: Component Ablation Analysis on Held-Out Test Set",
        "",
        "| System Variant / Ablation | nDCG@5 | P@3 | Δ nDCG@5 |",
        "|:--------------------------|:------:|:---:|:--------:|",
    ]
    baseline_ndcg = e3_data.get("Full System (Baseline)", {}).get("nDCG@5", 0.0)

    for variant, m in e3_data.items():
        delta = m["nDCG@5"] - baseline_ndcg
        delta_str = f"{delta:+.4f}" if variant != "Full System (Baseline)" else "— (Ref)"
        lines.append(f"| **{variant}** | {m['nDCG@5']:.4f} | {m['P@3']:.4f} | {delta_str} |")
    return "\n".join(lines)


def run_and_save_reports() -> Path:
    """Run full evaluation suite and persist markdown and JSON tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    harness = EvaluationHarness()
    results = harness.run_all_experiments()

    # Add metadata
    results["run_timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    results["dev_jds_evaluated"] = len(harness.dev_jds)
    results["test_jds_evaluated"] = len(harness.test_jds)
    results["total_pairs_evaluated"] = len(harness.pairs)

    # Save JSON summary
    (RESULTS_DIR / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save Markdown Tables
    t1 = format_e1_markdown_table(results["E1_model_comparison"])
    t2 = format_e2_markdown_table(results["E2_alpha_sweep"])
    t3 = format_e3_markdown_table(results["E3_ablations"])

    (RESULTS_DIR / "e1_model_comparison.md").write_text(t1, encoding="utf-8")
    (RESULTS_DIR / "e2_alpha_sweep.md").write_text(t2, encoding="utf-8")
    (RESULTS_DIR / "e3_ablations.md").write_text(t3, encoding="utf-8")

    full_report = f"# ResumeFit Benchmark Evaluation Report\n\nGenerated: {results['run_timestamp']}\n\n{t1}\n\n---\n\n{t2}\n\n---\n\n{t3}\n\n---\n\n### Robustness (E4) & Fairness (E5)\n- **Keyword Stuffed Penalty**: Candidate score reduced by {results['E4_robustness']['effective_penalty_delta']:.4f}\n- **Demographic Invariance (E5)**: Max score difference across identities = {results['E5_fairness']['max_score_difference']:.6f} (Variance = {results['E5_fairness']['score_variance']:.2e}, Invariant: {results['E5_fairness']['is_invariant']})\n"
    (RESULTS_DIR / "evaluation_report.md").write_text(full_report, encoding="utf-8")

    logger.info("Saved all evaluation artifacts to %s", RESULTS_DIR)
    return RESULTS_DIR


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running ResumeFit Empirical Evaluation Benchmark (E1–E6)...")
    out_dir = run_and_save_reports()
    print(f"Evaluation complete! All reports and tables saved to {out_dir}")
