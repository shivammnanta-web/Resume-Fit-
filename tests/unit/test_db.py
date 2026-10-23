"""Unit tests for database models, repositories, and right-to-erasure cascade (Task P4.10, P4.11).

Tests CRUD, vector retrieval, and cascading deletion using SQLite in-memory engine.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.db.models import (
    ResumeChunkModel,
    ResumeModel,
    ResumeSectionModel,
    ResumeSkillModel,
)
from core.db.repo import (
    DatabaseSessionManager,
    JobRepository,
    MatchReportRepository,
    ResumeRepository,
)


@pytest.fixture
def db_session():
    """Create fresh in-memory SQLite database session for unit testing."""
    manager = DatabaseSessionManager(db_url="sqlite:///:memory:")
    manager.create_tables()
    session = manager.get_session()
    yield session
    session.close()


@pytest.fixture
def sample_resume_payload() -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Sample resume data for database repository tests."""
    res_id = uuid.uuid4()
    resume_data = {
        "id": res_id,
        "filename": "candidate_test.pdf",
        "total_pages": 2,
        "total_experience_years": 4.5,
        "highest_degree": "master",
        "candidate_name": "Aarav Sharma",
        "candidate_email": "aarav@example.com",
        "candidate_phone": "+1-555-0199",
        "redacted_text": "Experienced software engineer...",
        "is_stuffed": False,
        "metadata_json": {"source": "unit_test"},
    }

    sections = [
        {
            "id": uuid.uuid4(),
            "section_type": "experience",
            "heading": "WORK EXPERIENCE",
            "text": "Senior Developer at TechCorp. Built scalable microservices in Python.",
            "start_char": 0,
            "end_char": 70,
        },
        {
            "id": uuid.uuid4(),
            "section_type": "skills",
            "heading": "TECHNICAL SKILLS",
            "text": "Python, FastAPI, PostgreSQL, Docker",
            "start_char": 75,
            "end_char": 115,
        },
    ]

    # Dummy 384-dimensional normalized embedding
    emb1 = [0.05] * 384
    emb2 = [0.08] * 384

    chunks = [
        {
            "id": uuid.uuid4(),
            "section_type": "experience",
            "chunk_index": 0,
            "text": "Senior Developer at TechCorp. Built scalable microservices in Python.",
            "token_count": 10,
            "embedding": emb1,
        },
        {
            "id": uuid.uuid4(),
            "section_type": "skills",
            "chunk_index": 1,
            "text": "Python, FastAPI, PostgreSQL, Docker",
            "token_count": 6,
            "embedding": emb2,
        },
    ]

    skills = [
        {
            "id": uuid.uuid4(),
            "canonical_label": "Python",
            "esco_uri": "http://data.europa.eu/esco/skill/s1",
            "evidence_level": "demonstrated",
            "section_type": "experience",
            "context_span": "Built scalable microservices in Python",
            "match_type": "exact",
        },
        {
            "id": uuid.uuid4(),
            "canonical_label": "FastAPI",
            "esco_uri": None,
            "evidence_level": "listed_only",
            "section_type": "skills",
            "context_span": "FastAPI",
            "match_type": "alias",
        },
    ]

    return resume_data, sections, chunks, skills


@pytest.mark.unit
def test_resume_repository_create_and_get(
    db_session, sample_resume_payload
) -> None:
    """Test full resume persistence and graph traversal."""
    resume_data, sections, chunks, skills = sample_resume_payload
    repo = ResumeRepository(db_session)

    created = repo.create_resume(resume_data, sections, chunks, skills)
    assert created.id == resume_data["id"]

    fetched = repo.get_resume(created.id)
    assert fetched is not None
    assert fetched.candidate_name == "Aarav Sharma"
    assert len(fetched.sections) == 2
    assert len(fetched.chunks) == 2
    assert len(fetched.skills) == 2


@pytest.mark.unit
def test_right_to_erasure_cascading_deletion(
    db_session, sample_resume_payload
) -> None:
    """Task P4.11: Deleting a resume must cascade and remove all chunks, sections, and skills."""
    resume_data, sections, chunks, skills = sample_resume_payload
    repo = ResumeRepository(db_session)

    created = repo.create_resume(resume_data, sections, chunks, skills)
    res_id = created.id

    # Verify rows exist
    assert db_session.scalar(select(ResumeModel).where(ResumeModel.id == res_id)) is not None
    assert len(db_session.scalars(select(ResumeSectionModel).where(ResumeSectionModel.resume_id == res_id)).all()) == 2
    assert len(db_session.scalars(select(ResumeChunkModel).where(ResumeChunkModel.resume_id == res_id)).all()) == 2
    assert len(db_session.scalars(select(ResumeSkillModel).where(ResumeSkillModel.resume_id == res_id)).all()) == 2

    # Execute Right-to-Erasure delete
    deleted = repo.delete_resume(res_id)
    assert deleted is True

    # Confirm all parent and child rows were completely expunged (0 orphan rows)
    assert db_session.scalar(select(ResumeModel).where(ResumeModel.id == res_id)) is None
    assert len(db_session.scalars(select(ResumeSectionModel).where(ResumeSectionModel.resume_id == res_id)).all()) == 0
    assert len(db_session.scalars(select(ResumeChunkModel).where(ResumeChunkModel.resume_id == res_id)).all()) == 0
    assert len(db_session.scalars(select(ResumeSkillModel).where(ResumeSkillModel.resume_id == res_id)).all()) == 0


@pytest.mark.unit
def test_vector_search_top_k_and_section_restricted(
    db_session, sample_resume_payload
) -> None:
    """Test top-k vector cosine similarity retrieval and section filtering."""
    resume_data, sections, chunks, skills = sample_resume_payload
    repo = ResumeRepository(db_session)
    repo.create_resume(resume_data, sections, chunks, skills)

    query_emb = [0.05] * 384
    top_chunks = repo.search_top_k_chunks(query_emb, limit=5)
    assert len(top_chunks) == 2
    assert top_chunks[0]["cosine_similarity"] > 0.99  # Identical vector

    # Section-restricted search: only 'skills' section
    restricted = repo.search_section_restricted(query_emb, allowed_sections=["skills"], limit=5)
    assert len(restricted) == 1
    assert restricted[0]["section_type"] == "skills"


@pytest.mark.unit
def test_job_repository_create_and_reverse_search(db_session) -> None:
    """Test job creation and reverse job suggestion matching."""
    repo = JobRepository(db_session)
    job_emb = [0.1] * 384

    job_data = {
        "id": uuid.uuid4(),
        "role_title": "Senior Python Backend Engineer",
        "raw_text": "We are seeking a senior backend engineer proficient in Python and PostgreSQL.",
        "min_years_experience": 3.0,
        "min_degree": "bachelor",
        "required_skills": ["Python", "PostgreSQL"],
        "preferred_skills": ["Docker"],
        "embedding": job_emb,
    }

    job = repo.create_job(job_data)
    assert job.role_title == "Senior Python Backend Engineer"

    # Reverse match using candidate embedding
    matches = repo.search_top_k_jobs(job_emb, limit=3)
    assert len(matches) == 1
    assert matches[0]["role_title"] == "Senior Python Backend Engineer"
    assert matches[0]["cosine_similarity"] > 0.99


@pytest.mark.unit
def test_match_report_repository_save_and_get(
    db_session, sample_resume_payload
) -> None:
    """Test persisting and retrieving an explainable FitReport."""
    resume_data, sections, chunks, skills = sample_resume_payload
    r_repo = ResumeRepository(db_session)
    resume = r_repo.create_resume(resume_data, sections, chunks, skills)

    j_repo = JobRepository(db_session)
    job = j_repo.create_job(
        {
            "id": uuid.uuid4(),
            "role_title": "Backend Engineer",
            "raw_text": "Requirements: Python, Flask",
            "embedding": [0.05] * 384,
        }
    )

    report_repo = MatchReportRepository(db_session)
    rep_id = uuid.uuid4()
    report_data = {
        "id": rep_id,
        "resume_id": resume.id,
        "job_id": job.id,
        "fit_index": 82.5,
        "verdict_headline": "Strong Match for Backend Engineer",
        "narrative_summary": "Candidate exhibits high technical competency.",
        "scores": {"hybrid_score": 0.825},
        "skills_analysis": {"coverage_pct": 100.0},
        "constraints": {"passed_filter": True},
    }

    saved = report_repo.save_report(report_data)
    assert saved.id == rep_id
    assert saved.fit_index == 82.5

    fetched = report_repo.get_report(rep_id)
    assert fetched is not None
    assert fetched.verdict_headline == "Strong Match for Backend Engineer"
