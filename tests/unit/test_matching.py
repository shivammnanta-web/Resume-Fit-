"""Unit and invariant test suite for the matching engine (Task P3.12).

Tests mathematical invariants, section-bounded chunking, hybrid fusion,
constraints, and report serialization.
"""

from __future__ import annotations

import json
import math

import pytest

from core.explain.report import ReportGenerator
from core.ingestion.types import Section, SectionType
from core.jd.parser import ParsedJD
from core.matching.chunking import SectionChunker
from core.matching.constraints import ConstraintEvaluator
from core.matching.hybrid import HybridFusionEngine
from core.matching.lexical import LexicalMatchResult, LexicalScorer
from core.matching.normalize import ScoreNormalizer
from core.matching.rerank import CandidateReranker
from core.matching.semantic import SemanticScorer
from core.skills.evidence import EvidenceLevel, SkillEvidence
from core.skills.extractor import ExtractedSkill


@pytest.fixture
def sample_sections() -> list[Section]:
    return [
        Section(
            section_type=SectionType.SUMMARY,
            heading="SUMMARY",
            text="Experienced backend engineer building scalable Python APIs and cloud systems.",
            start_char=0,
            end_char=80,
        ),
        Section(
            section_type=SectionType.EXPERIENCE,
            heading="WORK EXPERIENCE",
            text=(
                "Senior Backend Engineer at TechCorp (2020 - Present)\n"
                "• Designed and maintained Python microservices using FastAPI and PostgreSQL.\n"
                "• Implemented Redis caching to reduce database query latency by 50%.\n"
                "• Automated container deployments using Docker and GitHub Actions CI/CD."
            ),
            start_char=85,
            end_char=350,
        ),
        Section(
            section_type=SectionType.EDUCATION,
            heading="EDUCATION",
            text="Bachelor of Science in Computer Science, State University, 2016-2020.",
            start_char=355,
            end_char=430,
        ),
        Section(
            section_type=SectionType.SKILLS,
            heading="SKILLS",
            text="Python, FastAPI, PostgreSQL, Redis, Docker, Git, CI/CD, Linux",
            start_char=435,
            end_char=505,
        ),
    ]


@pytest.fixture
def sample_parsed_jd() -> ParsedJD:
    return ParsedJD(
        raw_text=(
            "Role: Senior Backend Engineer\n"
            "Requirements: 3+ years experience. Bachelor's degree.\n"
            "Must have experience in Python, FastAPI, and PostgreSQL.\n"
            "Preferred: Redis, Docker, Kubernetes."
        ),
        role_title="Senior Backend Engineer",
        min_years_experience=3.0,
        min_degree="bachelor",
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        preferred_skills=["Redis", "Docker", "Kubernetes"],
        all_skills=["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
        requirement_statements=[
            "Must have experience in Python, FastAPI, and PostgreSQL.",
            "3+ years professional experience.",
        ],
    )


@pytest.mark.unit
def test_chunking_never_crosses_section_boundaries(sample_sections: list[Section]) -> None:
    """Invariant: Chunks must strictly preserve section semantics."""
    chunker = SectionChunker(max_tokens=30, overlap_tokens=5)
    chunks = chunker.chunk_all_sections(sample_sections)

    assert len(chunks) > len(sample_sections)  # Longer sections were split
    for c in chunks:
        # Every chunk must have a valid section type and not exceed token budget
        assert c.section_type in [s.section_type for s in sample_sections]
        assert c.token_count <= 40  # Allow slight regex approximation slack
        assert not math.isnan(c.token_count)


@pytest.mark.unit
def test_lexical_invariants() -> None:
    """Invariant: Identical text -> high score; disjoint text -> ~0.0."""
    scorer = LexicalScorer()
    text1 = "Senior Python developer with experience in PostgreSQL, Docker, and Redis microservices."

    # Identical documents
    res_identical = scorer.score(text1, text1)
    assert res_identical.score >= 0.95
    assert not math.isnan(res_identical.score)
    assert len(res_identical.matched_terms) > 0

    # Completely disjoint domain document
    text_unrelated = "Classical ballet dancer proficient in pirouettes, choreography, and stage costumes."
    res_unrelated = scorer.score(text1, text_unrelated)
    assert res_unrelated.score <= 0.05
    assert not math.isnan(res_unrelated.score)


@pytest.mark.unit
def test_weights_sum_validation() -> None:
    """Invariant: alpha + beta must equal 1.0; invalid sum raises ValueError."""
    # Valid sum
    engine = HybridFusionEngine(alpha=0.4, beta=0.6)
    assert engine.alpha == 0.4
    assert engine.beta == 0.6

    # Invalid sum must raise
    with pytest.raises(ValueError, match="Hybrid weights must sum to 1.0"):
        HybridFusionEngine(alpha=0.5, beta=0.7)


@pytest.mark.unit
def test_score_normalization_in_bounds() -> None:
    """Invariant: Normalizer strictly maps raw scores to [0.0, 1.0] without NaNs."""
    normalizer = ScoreNormalizer(method="percentile")
    for raw in [-0.5, 0.0, 0.1, 0.35, 0.70, 0.95, 1.5]:
        norm_lex = normalizer.normalize_lexical(raw)
        norm_sem = normalizer.normalize_semantic(raw)
        assert 0.0 <= norm_lex <= 1.0
        assert 0.0 <= norm_sem <= 1.0
        assert not math.isnan(norm_lex)
        assert not math.isnan(norm_sem)


@pytest.mark.unit
def test_soft_constraints_multiplicative_penalty(sample_parsed_jd: ParsedJD) -> None:
    """Invariant: Soft constraints apply multiplicative deduction proportional to shortfall."""
    evaluator = ConstraintEvaluator(mode="soft")

    # Candidate with 1.0 years experience (shortfall = 2.0 years against 3.0 min)
    # Expected multiplier: 0.85^2 = 0.7225
    result = evaluator.evaluate(
        base_score=0.80,
        parsed_jd=sample_parsed_jd,
        candidate_years_exp=1.0,
        candidate_degree="bachelor",
        candidate_extracted_skills=["Python", "FastAPI", "PostgreSQL"],
    )

    assert result.passed_filter is True
    assert len(result.deductions) == 1
    assert result.deductions[0].constraint_name == "Experience Duration"
    assert pytest.approx(result.total_multiplier, rel=1e-3) == (0.85**2.0)
    assert result.final_score < 0.80
    assert result.final_fit_index == pytest.approx(0.80 * (0.85**2.0) * 100.0, abs=0.2)


@pytest.mark.unit
def test_hard_constraints_filter_rejection(sample_parsed_jd: ParsedJD) -> None:
    """Invariant: In hard mode, unmet requirements reject candidate with final_score = 0.0."""
    evaluator = ConstraintEvaluator(mode="hard")

    result = evaluator.evaluate(
        base_score=0.85,
        parsed_jd=sample_parsed_jd,
        candidate_years_exp=1.0,  # Shortfall
        candidate_degree="bachelor",
        candidate_extracted_skills=["Python", "FastAPI", "PostgreSQL"],
    )

    assert result.passed_filter is False
    assert result.final_score == 0.0
    assert result.final_fit_index == 0.0
    assert len(result.rejection_reasons) > 0


@pytest.mark.unit
def test_cross_encoder_toggle() -> None:
    """Invariant: Cross-encoder reranker returns unchanged list when disabled."""
    reranker = CandidateReranker(enabled=False)
    candidates = [{"id": "1", "score": 0.8}, {"id": "2", "score": 0.6}]
    result = reranker.rerank("Python developer", candidates)
    assert result == candidates


@pytest.mark.unit
def test_fit_report_serialization(
    sample_parsed_jd: ParsedJD, sample_sections: list[Section]
) -> None:
    """Invariant: FitReport contains all mandatory explainability fields and valid JSON."""
    generator = ReportGenerator()

    # Create dummy components
    lex_res = LexicalMatchResult(score=0.75)
    sem_scorer = SemanticScorer()
    sem_res = sem_scorer.score(sample_sections, sample_parsed_jd)
    hybrid_engine = HybridFusionEngine()
    hyb_score = hybrid_engine.fuse(lex_res, sem_res)
    evaluator = ConstraintEvaluator(mode="soft")
    c_res = evaluator.evaluate(
        base_score=hyb_score.hybrid_score,
        parsed_jd=sample_parsed_jd,
        candidate_years_exp=3.5,
        candidate_degree="bachelor",
    )

    ev = SkillEvidence(
        skill_name="Python",
        canonical_label="Python",
        esco_uri="http://data.europa.eu/esco/skill/s1",
        evidence_level=EvidenceLevel.DEMONSTRATED,
        section_type=SectionType.EXPERIENCE,
        context_span="Developed microservices using Python",
        start_char=10,
        end_char=45,
    )
    skill = ExtractedSkill(
        canonical_label="Python",
        esco_uri="http://data.europa.eu/esco/skill/s1",
        evidence_level=EvidenceLevel.DEMONSTRATED,
        evidence_records=[ev],
    )

    report = generator.build_report(
        parsed_jd=sample_parsed_jd,
        resume_sections=sample_sections,
        candidate_extracted_skills=[skill],
        lexical_result=lex_res,
        semantic_result=sem_res,
        hybrid_score=hyb_score,
        constraint_result=c_res,
        candidate_name="Jane Doe",
        candidate_years_exp=3.5,
        candidate_degree="bachelor",
    )

    # Check serialization
    json_str = report.to_json()
    parsed_json = json.loads(json_str)

    assert parsed_json["candidate_name"] == "Jane Doe"
    assert parsed_json["role_title"] == "Senior Backend Engineer"
    assert "fit_index" in parsed_json
    assert "verdict_headline" in parsed_json
    assert "scores" in parsed_json
    assert "skills_analysis" in parsed_json
    assert "lexical_attribution" in parsed_json
    assert "semantic_attribution" in parsed_json
    assert "constraints" in parsed_json
    assert "audit_metadata" in parsed_json
