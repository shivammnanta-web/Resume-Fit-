"""Constraint checking and transparent penalty module (Task P3.8).

Enforces Non-Negotiable Principle 4:
"Soft Constraints by Default. Years-of-experience and degree requirements must
default to soft multiplicative penalties (e.g., 0.85^Δy for missing years,
0.70 for missing degree) with itemized deductions shown in the report.
Hard-filter mode must be an explicit, opt-in config toggle."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from core.jd.parser import ParsedJD
from resumefit.config import get_config

logger = logging.getLogger(__name__)

DEGREE_HIERARCHY = {
    "phd": 3,
    "doctorate": 3,
    "master": 2,
    "bachelor": 1,
    "none": 0,
}


@dataclass
class Deduction:
    """An individual itemized penalty applied to the candidate's score."""

    constraint_name: str
    expected: str
    actual: str
    multiplier: float
    deduction_pct: float  # (1.0 - multiplier) * 100
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "expected": self.expected,
            "actual": self.actual,
            "multiplier": round(self.multiplier, 4),
            "deduction_pct": round(self.deduction_pct, 1),
            "explanation": self.explanation,
        }


@dataclass
class ConstraintEvaluationResult:
    """Outcome of constraint evaluation (soft penalties or hard pass/fail)."""

    passed_filter: bool  # True if candidate passes hard filter (always True in soft mode)
    final_score: float  # Score after applying cumulative multiplicative penalties [0.0, 1.0]
    final_fit_index: float  # [0.0, 100.0]
    total_multiplier: float  # Product of all penalty multipliers [0.0, 1.0]
    deductions: list[Deduction] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed_filter": self.passed_filter,
            "final_score": round(self.final_score, 4),
            "final_fit_index": round(self.final_fit_index, 1),
            "total_multiplier": round(self.total_multiplier, 4),
            "deductions": [d.to_dict() for d in self.deductions],
            "rejection_reasons": self.rejection_reasons,
        }


def detect_candidate_highest_degree(text: str) -> str:
    """Detect candidate's highest educational degree from resume text."""
    lower = text.lower()
    if re.search(r"\b(?:ph\.?d\.?|doctorate|doctoral)\b", lower):
        return "phd"
    if re.search(r"\b(?:master'?s?|m\.?s\.?|m\.?tech|m\.?sc|mba|postgraduate)\b", lower):
        return "master"
    if re.search(r"\b(?:bachelor'?s?|b\.?s\.?|b\.?tech|b\.?e\.?|b\.?sc|bba|undergraduate)\b", lower):
        return "bachelor"
    return "none"


class ConstraintEvaluator:
    """Evaluates experience, degree, and required skill constraints."""

    def __init__(
        self,
        mode: Literal["soft", "hard"] | None = None,
        min_years_penalty: float | None = None,
        missing_degree_penalty: float | None = None,
        missing_skill_penalty: float | None = None,
    ) -> None:
        cfg = get_config().constraints
        self.mode = mode or cfg.mode
        self.min_years_penalty = (
            min_years_penalty if min_years_penalty is not None else cfg.min_years_penalty_per_year
        )
        self.missing_degree_penalty = (
            missing_degree_penalty
            if missing_degree_penalty is not None
            else cfg.missing_degree_penalty
        )
        self.missing_skill_penalty = (
            missing_skill_penalty
            if missing_skill_penalty is not None
            else cfg.missing_must_have_skill_penalty
        )

    def evaluate(
        self,
        base_score: float,
        parsed_jd: ParsedJD,
        candidate_years_exp: float,
        candidate_degree: str | None = None,
        candidate_resume_text: str = "",
        candidate_extracted_skills: list[str] | None = None,
    ) -> ConstraintEvaluationResult:
        """Evaluate constraints against JD and compute itemized deductions."""
        deductions: list[Deduction] = []
        rejection_reasons: list[str] = []
        total_multiplier = 1.0

        # 1. Experience Years Constraint
        if parsed_jd.min_years_experience > 0 and candidate_years_exp < parsed_jd.min_years_experience:
            delta_years = parsed_jd.min_years_experience - candidate_years_exp
            mult = float(self.min_years_penalty**delta_years)
            deduction_pct = (1.0 - mult) * 100.0
            deductions.append(
                Deduction(
                    constraint_name="Experience Duration",
                    expected=f">={parsed_jd.min_years_experience:.1f} years",
                    actual=f"{candidate_years_exp:.1f} years",
                    multiplier=mult,
                    deduction_pct=deduction_pct,
                    explanation=(
                        f"Missing {delta_years:.1f} years required experience. "
                        f"Multiplicative deduction applied: {self.min_years_penalty:.2f}^{delta_years:.1f}."
                    ),
                )
            )
            total_multiplier *= mult
            if self.mode == "hard":
                rejection_reasons.append(
                    f"Requires {parsed_jd.min_years_experience} years; candidate has {candidate_years_exp:.1f} years."
                )

        # 2. Minimum Degree Constraint
        if parsed_jd.min_degree:
            req_degree_level = DEGREE_HIERARCHY.get(parsed_jd.min_degree.lower(), 1)
            actual_degree = candidate_degree or detect_candidate_highest_degree(
                candidate_resume_text
            )
            actual_degree_level = DEGREE_HIERARCHY.get(actual_degree.lower(), 0)

            if actual_degree_level < req_degree_level:
                mult = self.missing_degree_penalty
                deduction_pct = (1.0 - mult) * 100.0
                deductions.append(
                    Deduction(
                        constraint_name="Education Level",
                        expected=parsed_jd.min_degree.capitalize(),
                        actual=actual_degree.capitalize() if actual_degree != "none" else "Not detected",
                        multiplier=mult,
                        deduction_pct=deduction_pct,
                        explanation=(
                            f"Role requires {parsed_jd.min_degree.capitalize()} degree; "
                            f"candidate highest detected credential is {actual_degree.capitalize()}."
                        ),
                    )
                )
                total_multiplier *= mult
                if self.mode == "hard":
                    rejection_reasons.append(
                        f"Requires {parsed_jd.min_degree.capitalize()} degree; candidate has {actual_degree}."
                    )

        # 3. Missing Must-Have Skills Constraint
        if parsed_jd.required_skills:
            candidate_skills_lower = {
                s.lower() for s in (candidate_extracted_skills or [])
            }
            # Check against full text as well
            resume_lower = candidate_resume_text.lower()
            missing_skills: list[str] = []
            for req_skill in parsed_jd.required_skills:
                req_lower = req_skill.lower()
                if (
                    req_lower not in candidate_skills_lower
                    and not re.search(rf"\b{re.escape(req_lower)}\b", resume_lower)
                ):
                    missing_skills.append(req_skill)

            if missing_skills:
                missing_ratio = len(missing_skills) / len(parsed_jd.required_skills)
                mult = float(self.missing_skill_penalty**missing_ratio)
                deduction_pct = (1.0 - mult) * 100.0
                deductions.append(
                    Deduction(
                        constraint_name="Required Skills Coverage",
                        expected=f"All {len(parsed_jd.required_skills)} required skills",
                        actual=f"Missing {len(missing_skills)}: {', '.join(missing_skills[:3])}{'...' if len(missing_skills) > 3 else ''}",
                        multiplier=mult,
                        deduction_pct=deduction_pct,
                        explanation=(
                            f"Candidate lacks {len(missing_skills)} out of {len(parsed_jd.required_skills)} "
                            f"required skills: {', '.join(missing_skills[:5])}."
                        ),
                    )
                )
                total_multiplier *= mult
                if self.mode == "hard":
                    rejection_reasons.append(
                        f"Missing required skills: {', '.join(missing_skills)}"
                    )

        passed = True
        if self.mode == "hard" and rejection_reasons:
            passed = False
            final_score = 0.0
            final_fit_idx = 0.0
        else:
            final_score = float(max(0.0, min(1.0, base_score * total_multiplier)))
            final_fit_idx = round(final_score * 100.0, 1)

        return ConstraintEvaluationResult(
            passed_filter=passed,
            final_score=final_score,
            final_fit_index=final_fit_idx,
            total_multiplier=total_multiplier,
            deductions=deductions,
            rejection_reasons=rejection_reasons,
        )
