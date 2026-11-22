"""Explainable Fit Report generator and serializer (Task P3.10).

Enforces Non-Negotiable Principle 1:
"Explainability First. Every match score must decompose into human-auditable components:
- Fit Index (0–100, never framed as 'probability')
- Lexical match score (TF-IDF cosine) with top contributing n-grams
- Semantic match score (Sentence-Transformers cosine) with qualitative rating (Strong / Moderate / Weak)
- Skill coverage table (required skills vs. candidate skills, evidence tier: Demonstrated / Listed-only / Tentative, text span + offset)
- Constraint checks (years of experience delta, degree check, itemized deductions)
- Keyword-stuffing flag (if uncorroborated skills exceed threshold)
Deterministic template wording. No opaque generative LLM summaries in the core pipeline."
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from core.ingestion.types import Section
from core.jd.parser import ParsedJD
from core.matching.constraints import ConstraintEvaluationResult
from core.matching.hybrid import HybridMatchScore
from core.matching.lexical import LexicalMatchResult
from core.matching.semantic import SemanticMatchResult
from core.skills.extractor import ExtractedSkill
from resumefit.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SkillCoverageItem:
    """Detailed audit entry for an individual JD skill."""

    skill_name: str
    is_required: bool
    is_matched: bool
    evidence_tier: str | None  # "Demonstrated", "Listed-only", "Tentative", or None
    matched_section: str | None
    matched_span: str | None


@dataclass
class FitReport:
    """Comprehensive, human-auditable fit report for candidate-JD matching."""

    candidate_name: str | None
    role_title: str | None
    fit_index: float  # [0.0, 100.0]
    verdict_headline: str
    narrative_summary: str
    scores: dict[str, Any]
    skills_analysis: dict[str, Any]
    lexical_attribution: dict[str, Any]
    semantic_attribution: dict[str, Any]
    constraints: dict[str, Any]
    audit_metadata: dict[str, Any]
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ReportGenerator:
    """Deterministic report builder combining evidence from all pipeline stages."""

    def __init__(self) -> None:
        self.config = get_config()

    def generate_narrative(
        self,
        fit_index: float,
        role_title: str | None,
        coverage_pct: float,
        matched_required: int,
        total_required: int,
        constraint_result: ConstraintEvaluationResult,
        keyword_stuffing: bool,
    ) -> tuple[str, str]:
        """Generate deterministic verdict headline and narrative paragraph."""
        role_str = role_title or "Target Position"

        # 1. Headline
        if fit_index >= 80.0:
            headline = f"Strong Match for {role_str}"
        elif fit_index >= 65.0:
            headline = f"Competitive Candidate for {role_str}"
        elif fit_index >= 50.0:
            headline = f"Moderate Match with Gaps for {role_str}"
        else:
            headline = f"Low Alignment for {role_str}"

        # 2. Multi-point narrative sentences
        sentences: list[str] = []

        # Fit overview
        if fit_index >= 80.0:
            sentences.append(
                f"Candidate demonstrates high overall alignment (Fit Index: {fit_index:.1f}/100) with solid verified competencies."
            )
        elif fit_index >= 65.0:
            sentences.append(
                f"Candidate demonstrates competitive baseline qualifications (Fit Index: {fit_index:.1f}/100) across key domain areas."
            )
        elif fit_index >= 50.0:
            sentences.append(
                f"Candidate demonstrates moderate technical relevance (Fit Index: {fit_index:.1f}/100), but displays notable qualification deficits."
            )
        else:
            sentences.append(
                f"Candidate shows limited overall match (Fit Index: {fit_index:.1f}/100) with the published position requirements."
            )

        # Skill coverage breakdown
        if total_required > 0:
            sentences.append(
                f"Verified coverage of required core skills is {coverage_pct:.1f}% ({matched_required}/{total_required} mandatory skills present)."
            )

        # Constraint deduction narrative
        if constraint_result.deductions:
            ded_strs = [
                f"{d.constraint_name} (-{d.deduction_pct:.0f}%)"
                for d in constraint_result.deductions
            ]
            sentences.append(
                f"Overall score was adjusted by soft constraint penalties: {'; '.join(ded_strs)}."
            )
        else:
            sentences.append(
                "Candidate satisfies all stated tenure duration and educational credential baselines without penalty."
            )

        # Keyword stuffing notice
        if keyword_stuffing:
            sentences.append(
                "WARNING: Skill evidence audit detected an abnormal ratio of uncorroborated keyword listings without contextual demonstration in experience or project sections."
            )

        narrative = " ".join(sentences)
        return headline, narrative

    def build_report(
        self,
        parsed_jd: ParsedJD,
        resume_sections: list[Section],
        candidate_extracted_skills: list[ExtractedSkill],
        lexical_result: LexicalMatchResult,
        semantic_result: SemanticMatchResult,
        hybrid_score: HybridMatchScore,
        constraint_result: ConstraintEvaluationResult,
        candidate_name: str | None = None,
        candidate_years_exp: float = 0.0,
        candidate_degree: str | None = None,
        keyword_stuffing_flag: bool = False,
        keyword_stuffing_ratio: float = 0.0,
    ) -> FitReport:
        """Assemble full FitReport with deterministic components."""
        _ = resume_sections
        # Index candidate skills by lowercase canonical label
        skill_by_name: dict[str, ExtractedSkill] = {}
        for s in candidate_extracted_skills:
            skill_by_name[s.canonical_label.lower()] = s

        coverage_items: list[SkillCoverageItem] = []
        matched_required_count = 0
        total_required_count = len(parsed_jd.required_skills)

        # Process required skills
        for r_skill in parsed_jd.required_skills:
            sk = skill_by_name.get(r_skill.lower())
            is_matched = sk is not None
            ev = sk.evidence_records[0] if (sk and sk.evidence_records) else None
            if is_matched:
                matched_required_count += 1
            coverage_items.append(
                SkillCoverageItem(
                    skill_name=r_skill,
                    is_required=True,
                    is_matched=is_matched,
                    evidence_tier=sk.evidence_level.value if sk else None,
                    matched_section=ev.section_type.value if ev else None,
                    matched_span=ev.context_span[:120] if ev else None,
                )
            )

        # Process preferred skills
        for p_skill in parsed_jd.preferred_skills:
            sk = skill_by_name.get(p_skill.lower())
            ev = sk.evidence_records[0] if (sk and sk.evidence_records) else None
            coverage_items.append(
                SkillCoverageItem(
                    skill_name=p_skill,
                    is_required=False,
                    is_matched=sk is not None,
                    evidence_tier=sk.evidence_level.value if sk else None,
                    matched_section=ev.section_type.value if ev else None,
                    matched_span=ev.context_span[:120] if ev else None,
                )
            )

        req_coverage_pct = (
            (matched_required_count / total_required_count * 100.0)
            if total_required_count > 0
            else 100.0
        )

        # Semantic rating classification
        qualitative_rating = semantic_result.requirement_matches[0].rating if semantic_result.requirement_matches else "Moderate"

        # Final fit index after constraints
        final_fit_idx = constraint_result.final_fit_index

        headline, narrative = self.generate_narrative(
            fit_index=final_fit_idx,
            role_title=parsed_jd.role_title,
            coverage_pct=req_coverage_pct,
            matched_required=matched_required_count,
            total_required=total_required_count,
            constraint_result=constraint_result,
            keyword_stuffing=keyword_stuffing_flag,
        )

        return FitReport(
            candidate_name=candidate_name,
            role_title=parsed_jd.role_title,
            fit_index=final_fit_idx,
            verdict_headline=headline,
            narrative_summary=narrative,
            scores={
                "hybrid_raw": hybrid_score.hybrid_score,
                "pre_constraint_fit_index": hybrid_score.fit_index,
                "final_score": constraint_result.final_score,
                "final_fit_index": final_fit_idx,
                "fit_index": final_fit_idx,
                "lexical_raw": hybrid_score.lexical_raw,
                "lexical_normalized": hybrid_score.lexical_normalized,
                "lexical_score": hybrid_score.lexical_normalized,
                "semantic_raw": hybrid_score.semantic_raw,
                "semantic_normalized": hybrid_score.semantic_normalized,
                "semantic_score": hybrid_score.semantic_normalized,
                "semantic_rating": qualitative_rating,
                "alpha": hybrid_score.alpha,
                "beta": hybrid_score.beta,
                "fusion_method": hybrid_score.fusion_method,
            },
            skills_analysis={
                "total_jd_skills": len(parsed_jd.all_skills),
                "required_skills_count": total_required_count,
                "matched_required_count": matched_required_count,
                "required_coverage_pct": round(req_coverage_pct, 1),
                "matched_required_skills": matched_required_count,
                "total_required_skills": total_required_count,
                "skill_coverage_percentage": round(req_coverage_pct, 1),
                "skill_coverage_table": [asdict(item) for item in coverage_items],
                "keyword_stuffing_detected": keyword_stuffing_flag,
                "keyword_stuffing_ratio": round(keyword_stuffing_ratio, 2),
            },
            lexical_attribution={
                **lexical_result.to_dict(),
                "top_contributing_terms": lexical_result.to_dict().get("matched_terms", []),
            },
            semantic_attribution={
                "mean_pooled_score": round(semantic_result.mean_pooled_score, 4),
                "requirement_to_chunk_score": round(
                    semantic_result.requirement_to_chunk_score, 4
                ),
                "qualitative_rating": qualitative_rating,
                "requirement_evidence": [
                    rm.to_dict() for rm in semantic_result.requirement_matches
                ],
            },
            constraints={
                "passed_filter": constraint_result.passed_filter,
                "candidate_experience_years": round(candidate_years_exp, 1),
                "candidate_years_exp": round(candidate_years_exp, 1),
                "required_experience_years": parsed_jd.min_years_experience,
                "required_years_exp": parsed_jd.min_years_experience,
                "candidate_degree": candidate_degree,
                "required_degree": parsed_jd.min_degree,
                "years_penalty": next((d.multiplier for d in constraint_result.deductions if "Experience" in d.constraint_name), 1.0),
                "degree_penalty": next((d.multiplier for d in constraint_result.deductions if "Education" in d.constraint_name), 1.0),
                "must_have_penalty": next((d.multiplier for d in constraint_result.deductions if "Skill" in d.constraint_name), 1.0),
                "total_multiplier": round(constraint_result.total_multiplier, 4),
                "total_penalty_multiplier": round(constraint_result.total_multiplier, 4),
                "deductions": [d.to_dict() for d in constraint_result.deductions],
                "rejection_reasons": constraint_result.rejection_reasons,
            },
            audit_metadata={
                "esco_version": self.config.skills.esco_version,
                "embedding_model": self.config.matching.embedding_model,
                "pipeline_version": self.config.app.version,
                "alpha": hybrid_score.alpha,
                "beta": hybrid_score.beta,
                "timestamp_iso": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )
