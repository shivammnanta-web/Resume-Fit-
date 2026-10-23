"""Evaluation harness for skill extraction Precision, Recall, and F1 (Task P2.11).

Evaluates the ESCO + Alias SkillExtractor against the 20 hand-labeled gold standard
resumes in `data/gold/skill_labels/`.
Computes both micro-averaged and macro-averaged Precision, Recall, and F1 metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.ingestion.sections import detect_sections
from core.skills.extractor import SkillExtractor

DEFAULT_GOLD_DIR = Path("data/gold/skill_labels")


@dataclass
class ResumeEvalResult:
    """Evaluation result for an individual benchmark resume."""

    resume_id: str
    domain: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    expected_count: int
    predicted_count: int
    matched_skills: list[str]
    missing_skills: list[str]
    spurious_skills: list[str]


@dataclass
class EvaluationSummary:
    """Aggregated micro and macro performance summary across all benchmark resumes."""

    total_resumes: int
    total_tp: int
    total_fp: int
    total_fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    results: list[ResumeEvalResult]

    def print_report(self) -> None:
        """Print a formatted ASCII evaluation report."""
        print("\n" + "=" * 90)
        print(f" RESUMEFIT SKILL EXTRACTION EVALUATION REPORT (N={self.total_resumes})")
        print("=" * 90)
        header = f"{'Resume ID':<12} | {'Domain':<22} | {'Exp':<4} | {'Pred':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}"
        print(header)
        print("-" * 90)

        for r in self.results:
            row = (
                f"{r.resume_id:<12} | {r.domain:<22} | {r.expected_count:<4} | {r.predicted_count:<4} | "
                f"{r.true_positives:<4} | {r.false_positives:<4} | {r.false_negatives:<4} | "
                f"{r.precision:.1%} | {r.recall:.1%} | {r.f1:.1%}"
            )
            print(row)

        print("-" * 90)
        print("SUMMARY METRICS:")
        print(f"  • Total Ground-Truth Skills : {self.total_tp + self.total_fn}")
        print(f"  • Total Extracted Skills    : {self.total_tp + self.total_fp}")
        print(f"  • True Positives (TP)       : {self.total_tp}")
        print(f"  • False Positives (FP)      : {self.total_fp}")
        print(f"  • False Negatives (FN)      : {self.total_fn}")
        print("  " + "-" * 40)
        print(f"  • Micro-Averaged Precision  : {self.micro_precision:.1%}")
        print(f"  • Micro-Averaged Recall     : {self.micro_recall:.1%}")
        print(f"  • Micro-Averaged F1 Score   : {self.micro_f1:.1%}")
        print("  " + "-" * 40)
        print(f"  • Macro-Averaged Precision  : {self.macro_precision:.1%}")
        print(f"  • Macro-Averaged Recall     : {self.macro_recall:.1%}")
        print(f"  • Macro-Averaged F1 Score   : {self.macro_f1:.1%}")
        print("=" * 90 + "\n")


def normalize_skill_string(s: str) -> str:
    """Normalize skill name for matching evaluation."""
    return (
        s.strip()
        .lower()
        .replace(" (programming language)", "")
        .replace(" (software)", "")
        .replace(" (computer programming)", "")
    )


def evaluate_dev_set(gold_dir: Path | str | None = None) -> EvaluationSummary:
    """Evaluate skill extraction across the gold-standard benchmark set."""
    path = Path(gold_dir) if gold_dir else DEFAULT_GOLD_DIR
    json_files = sorted(path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No gold standard files found in {path.resolve()}")

    extractor = SkillExtractor()
    per_resume_results: list[ResumeEvalResult] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for fpath in json_files:
        with open(fpath, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        rid = data.get("id", fpath.stem)
        domain = data.get("domain", "general")
        resume_text = data.get("text", "")
        expected_raw = data.get("expected_skills", [])

        # Ingestion: Detect sections and extract skills
        sections = detect_sections(resume_text)
        res = extractor.extract_from_sections(sections)
        predicted_raw = res.canonical_names

        # Normalization for set comparison
        expected_norm_map = {normalize_skill_string(s): s for s in expected_raw}
        predicted_norm_map = {normalize_skill_string(s): s for s in predicted_raw}

        expected_set = set(expected_norm_map.keys())
        predicted_set = set(predicted_norm_map.keys())

        # Exact and canonical overlap
        tp_keys = expected_set.intersection(predicted_set)
        fp_keys = predicted_set.difference(expected_set)
        fn_keys = expected_set.difference(predicted_set)

        tp = len(tp_keys)
        fp = len(fp_keys)
        fn = len(fn_keys)

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_resume_results.append(
            ResumeEvalResult(
                resume_id=rid,
                domain=domain,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=prec,
                recall=rec,
                f1=f1,
                expected_count=len(expected_raw),
                predicted_count=len(predicted_raw),
                matched_skills=[expected_norm_map[k] for k in tp_keys],
                missing_skills=[expected_norm_map[k] for k in fn_keys],
                spurious_skills=[predicted_norm_map[k] for k in fp_keys],
            )
        )

    # Micro metrics
    micro_p = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_r = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # Macro metrics
    n = len(per_resume_results)
    macro_p = sum(r.precision for r in per_resume_results) / n if n > 0 else 0.0
    macro_r = sum(r.recall for r in per_resume_results) / n if n > 0 else 0.0
    macro_f1 = sum(r.f1 for r in per_resume_results) / n if n > 0 else 0.0

    summary = EvaluationSummary(
        total_resumes=n,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        results=per_resume_results,
    )
    return summary


def main() -> None:
    summary = evaluate_dev_set()
    summary.print_report()


if __name__ == "__main__":
    main()
