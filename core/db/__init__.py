"""Database layer for ResumeFit (PostgreSQL 15 + pgvector + HNSW)."""

from core.db.models import (
    Base,
    JobModel,
    MatchReportModel,
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

__all__ = [
    "Base",
    "DatabaseSessionManager",
    "JobModel",
    "JobRepository",
    "MatchReportModel",
    "MatchReportRepository",
    "ResumeChunkModel",
    "ResumeModel",
    "ResumeRepository",
    "ResumeSectionModel",
    "ResumeSkillModel",
]
