"""SQLAlchemy database models for ResumeFit (Task P4.3).

Schema covers:
- resumes: Candidate metadata, page counts, total experience years, PII redactions
- resume_sections: Detected sections with start/end character offsets
- resume_chunks: Bounded text chunks with 384-d pgvector embeddings and full-text TSVector
- resume_skills: Extracted ESCO skills with evidence levels and contextual spans
- jobs: Job descriptions, role requirements, degree constraints, and embeddings
- match_reports: Persisted auditable match reports and JSON explainability payloads

Enforces cascading deletions (ON DELETE CASCADE) for complete Right-to-Erasure compliance.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy database entities."""

    pass


class ResumeModel(Base):
    """Candidate resume entity."""

    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )
    total_pages: Mapped[int] = mapped_column(Integer, default=1)
    total_experience_years: Mapped[float] = mapped_column(Float, default=0.0)
    highest_degree: Mapped[str | None] = mapped_column(String(50), nullable=True)
    candidate_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    redacted_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # Opt-in persistence
    is_stuffed: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict
    )

    # Cascading relationships for Right-to-Erasure
    sections: Mapped[list[ResumeSectionModel]] = relationship(
        "ResumeSectionModel",
        back_populates="resume",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list[ResumeChunkModel]] = relationship(
        "ResumeChunkModel",
        back_populates="resume",
        cascade="all, delete-orphan",
    )
    skills: Mapped[list[ResumeSkillModel]] = relationship(
        "ResumeSkillModel",
        back_populates="resume",
        cascade="all, delete-orphan",
    )
    match_reports: Mapped[list[MatchReportModel]] = relationship(
        "MatchReportModel",
        back_populates="resume",
        cascade="all, delete-orphan",
    )


class ResumeSectionModel(Base):
    """A detected resume section (e.g., Work Experience, Education)."""

    __tablename__ = "resume_sections"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    heading: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, default=0)
    end_char: Mapped[int] = mapped_column(Integer, default=0)

    resume: Mapped[ResumeModel] = relationship("ResumeModel", back_populates="sections")
    chunks: Mapped[list[ResumeChunkModel]] = relationship(
        "ResumeChunkModel",
        back_populates="section",
        cascade="all, delete-orphan",
    )


class ResumeChunkModel(Base):
    """Bounded semantic chunk with 384-dimensional dense embedding."""

    __tablename__ = "resume_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("resume_sections.id", ondelete="CASCADE"), nullable=True
    )
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    resume: Mapped[ResumeModel] = relationship("ResumeModel", back_populates="chunks")
    section: Mapped[ResumeSectionModel | None] = relationship(
        "ResumeSectionModel", back_populates="chunks"
    )


class ResumeSkillModel(Base):
    """ESCO-grounded skill occurrence linked to a resume."""

    __tablename__ = "resume_skills"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    canonical_label: Mapped[str] = mapped_column(String(255), nullable=False)
    esco_uri: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence_level: Mapped[str] = mapped_column(String(50), nullable=False)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    context_span: Mapped[str] = mapped_column(Text, nullable=False)
    match_type: Mapped[str] = mapped_column(String(50), default="exact")

    resume: Mapped[ResumeModel] = relationship("ResumeModel", back_populates="skills")


class JobModel(Base):
    """Job description entity with requirements, constraints, and embedding."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    role_title: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    min_years_experience: Mapped[float] = mapped_column(Float, default=0.0)
    min_degree: Mapped[str | None] = mapped_column(String(50), nullable=True)
    required_skills: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list
    )
    preferred_skills: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )

    match_reports: Mapped[list[MatchReportModel]] = relationship(
        "MatchReportModel",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class MatchReportModel(Base):
    """Persisted match assessment report linking a candidate resume and job."""

    __tablename__ = "match_reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), nullable=True
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True
    )
    fit_index: Mapped[float] = mapped_column(Float, nullable=False)
    verdict_headline: Mapped[str] = mapped_column(String(255), nullable=False)
    narrative_summary: Mapped[str] = mapped_column(Text, nullable=False)
    scores_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict
    )
    skills_analysis_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict
    )
    constraints_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict
    )
    audit_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.UTC)
    )

    resume: Mapped[ResumeModel | None] = relationship("ResumeModel", back_populates="match_reports")
    job: Mapped[JobModel | None] = relationship("JobModel", back_populates="match_reports")
