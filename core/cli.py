"""ResumeFit command line interface (Tasks P3.11 & P3.13).

Provides:
- analyze: End-to-end matching of a resume against a job description.
- compare: Side-by-side comparison of TF-IDF, Semantic, and Hybrid architectures.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

from core.explain.report import ReportGenerator
from core.ingestion.pipeline import process_resume
from core.ingestion.types import UnsupportedDocument
from core.jd.parser import JDParser
from core.matching.constraints import ConstraintEvaluator, detect_candidate_highest_degree
from core.matching.hybrid import HybridFusionEngine
from core.matching.lexical import LexicalScorer
from core.matching.semantic import SemanticScorer
from core.skills.extractor import SkillExtractor
from resumefit.config import get_config
from resumefit.logging import setup_logging

logger = logging.getLogger(__name__)


class MatchPipeline:
    """Full end-to-end ResumeFit matching pipeline."""

    def __init__(self) -> None:
        self.config = get_config()
        self.skill_extractor = SkillExtractor()
        self.jd_parser = JDParser(skill_extractor=self.skill_extractor)
        self.lexical_scorer = LexicalScorer()
        self.semantic_scorer = SemanticScorer()
        self.hybrid_engine = HybridFusionEngine()
        self.constraint_evaluator = ConstraintEvaluator()
        self.report_generator = ReportGenerator()

    def run(self, resume_path: Path | str, jd_text: str) -> Any:
        resume_path = Path(resume_path)

        # 1. Ingestion (validates scan/encryption/size, cleans, sections, parses dates, redacts PII)
        ingestion = process_resume(resume_path)
        sections = ingestion.sections
        clean_text = ingestion.unredacted_full_text
        total_years = ingestion.total_experience_years
        candidate_degree = detect_candidate_highest_degree(clean_text)

        # 2. Skill extraction & Evidence
        skill_res = self.skill_extractor.extract_from_sections(sections)
        extracted_skills = skill_res.skills
        is_stuffed = skill_res.stuffing_report.is_stuffed
        stuffing_ratio = skill_res.stuffing_report.ratio

        # 3. JD Parsing
        parsed_jd = self.jd_parser.parse(jd_text)

        # 4. Lexical Scoring (TF-IDF with frozen reference vocabulary)
        lexical_result = self.lexical_scorer.score(clean_text, parsed_jd.raw_text)

        # 5. Semantic Scoring (Dual Architecture)
        semantic_result = self.semantic_scorer.score(
            resume_sections=sections,
            parsed_jd=parsed_jd,
            resume_raw_text=ingestion.redacted_full_text,  # Pre-embedding PII redaction
        )

        # 6. Hybrid Fusion
        hybrid_score = self.hybrid_engine.fuse(lexical_result, semantic_result)

        # 7. Constraint Evaluation
        skill_names = [s.canonical_label for s in extracted_skills]
        constraint_result = self.constraint_evaluator.evaluate(
            base_score=hybrid_score.hybrid_score,
            parsed_jd=parsed_jd,
            candidate_years_exp=total_years,
            candidate_degree=candidate_degree,
            candidate_resume_text=clean_text,
            candidate_extracted_skills=skill_names,
        )

        # 8. Report Generation
        report = self.report_generator.build_report(
            parsed_jd=parsed_jd,
            resume_sections=sections,
            candidate_extracted_skills=extracted_skills,
            lexical_result=lexical_result,
            semantic_result=semantic_result,
            hybrid_score=hybrid_score,
            constraint_result=constraint_result,
            candidate_name=ingestion.detected_candidate_name or "Candidate",
            candidate_years_exp=total_years,
            candidate_degree=candidate_degree,
            keyword_stuffing_flag=is_stuffed,
            keyword_stuffing_ratio=stuffing_ratio,
        )
        return report


def format_terminal_report(report: Any) -> str:
    """Format FitReport into clean, auditable terminal output."""
    d = report.to_dict()
    scores = d["scores"]
    skills = d["skills_analysis"]
    constraints = d["constraints"]

    lines = [
        "",
        "=" * 70,
        "  RESUMEFIT EXPLAINABLE MATCH REPORT",
        "=" * 70,
        f"  Candidate: {d.get('candidate_name', 'Unknown')}   |   Role: {d.get('role_title', 'Not specified')}",
        f"  FIT INDEX: {d['fit_index']:.1f} / 100   [{d['verdict_headline']}]",
        "-" * 70,
        f"  {d['narrative_summary']}",
        "",
        "  RETRIEVAL SCORES BREAKDOWN:",
        f"    * Lexical (TF-IDF Cosine):      Raw: {scores['lexical_raw']:.4f}  |  Normalized: {scores['lexical_normalized']:.4f}  (weight: {scores['alpha']})",
        f"    * Semantic (Sentence-Transf):   Raw: {scores['semantic_raw']:.4f}  |  Normalized: {scores['semantic_normalized']:.4f}  (weight: {scores['beta']})",
        f"    * Hybrid Score:                 {scores['hybrid_raw']:.4f} (Pre-constraint Fit: {scores['pre_constraint_fit_index']:.1f})",
        f"    * Final Adjusted Score:         {scores['final_score']:.4f} -> Fit Index: {scores['final_fit_index']:.1f}",
        "",
        "  SKILL COVERAGE AUDIT:",
        f"    * Required Skills Coverage:     {skills['required_coverage_pct']:.1f}% ({skills['matched_required_count']}/{skills['required_skills_count']})",
        f"    * Keyword Stuffing Detected:    {'YES (Ratio: ' + str(skills['keyword_stuffing_ratio']) + ')' if skills['keyword_stuffing_detected'] else 'No'}",
    ]

    coverage_table = skills.get("skill_coverage_table", [])
    if coverage_table:
        lines.append("    * Key Matched / Missing Skills:")
        for item in coverage_table[:8]:
            status = "[+] MATCHED" if item["is_matched"] else "[-] MISSING"
            req_str = "[REQ]" if item["is_required"] else "[PREF]"
            tier = f"({item['evidence_tier']})" if item.get("evidence_tier") else ""
            lines.append(f"      - {req_str} {item['skill_name']:<22} {status:<11} {tier}")

    deductions = constraints.get("deductions", [])
    lines.append("")
    lines.append("  CONSTRAINTS & DEDUCTIONS:")
    if deductions:
        for ded in deductions:
            lines.append(f"    * [PENALTY] {ded['constraint_name']}: {ded['explanation']}")
    else:
        lines.append("    * No penalties applied; all minimum tenure and degree requirements met.")

    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)


def format_comparison_table(report: Any) -> str:
    """Format comparative table between pure lexical, pure semantic, and hybrid."""
    d = report.to_dict()
    scores = d["scores"]

    lex_raw = scores["lexical_raw"]
    lex_fit = scores["lexical_normalized"] * 100.0
    sem_raw = scores["semantic_raw"]
    sem_fit = scores["semantic_normalized"] * 100.0
    hyb_fit = scores["pre_constraint_fit_index"]
    final_fit = d["fit_index"]

    lines = [
        "",
        "=" * 72,
        "  ARCHITECTURE COMPARISON: LEXICAL vs SEMANTIC vs HYBRID",
        "=" * 72,
        f"{'Architecture':<22} | {'Raw Score':<12} | {'Normalized Fit':<16} | {'Status'}",
        "-" * 72,
        f"{'1. Lexical (TF-IDF)':<22} | {lex_raw:<12.4f} | {lex_fit:<16.1f} | Exact keyword overlap",
        f"{'2. Semantic (MiniLM)':<22} | {sem_raw:<12.4f} | {sem_fit:<16.1f} | Conceptual alignment",
        f"{'3. Hybrid (ResumeFit)':<22} | {scores['hybrid_raw']:<12.4f} | {hyb_fit:<16.1f} | Balanced (α=0.4, β=0.6)",
        "-" * 72,
        f"{'Final (Post-Constraints)':<22} | {scores['final_score']:<12.4f} | {final_fit:<16.1f} | Audited & Viva-Ready",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    with contextlib.suppress(Exception):
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    setup_logging(log_format="text")
    parser = argparse.ArgumentParser(description="ResumeFit Match & Explain CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze resume against job description")
    analyze_parser.add_argument("--resume", required=True, help="Path to resume PDF")
    analyze_parser.add_argument("--jd", required=True, help="Path to JD text file or raw JD string")
    analyze_parser.add_argument("--output", help="Optional path to save report JSON")

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare Lexical vs Semantic vs Hybrid retrieval"
    )
    compare_parser.add_argument("--resume", required=True, help="Path to resume PDF")
    compare_parser.add_argument("--jd", required=True, help="Path to JD text file or raw JD string")

    args = parser.parse_args()

    # Read JD
    jd_arg = args.jd
    if Path(jd_arg).is_file():
        jd_text = Path(jd_arg).read_text(encoding="utf-8")
    else:
        jd_text = jd_arg.replace("\\n", "\n")

    pipeline = MatchPipeline()
    try:
        report = pipeline.run(args.resume, jd_text)
    except UnsupportedDocument as e:
        print(f"\n[UNSUPPORTED DOCUMENT]: {e.reason.value} — {e.message}\n", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        logger.exception("Error analyzing match: %s", e)
        sys.exit(1)

    if args.command == "analyze":
        print(format_terminal_report(report))
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report.to_json(), encoding="utf-8")
            print(f"Report JSON written to {out_path}\n")

    elif args.command == "compare":
        print(format_comparison_table(report))


if __name__ == "__main__":
    main()
