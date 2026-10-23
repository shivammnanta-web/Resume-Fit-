"""Database repository layer with typed CRUD operations and vector search (Tasks P4.10 & P4.11).

Provides:
- ResumeRepository: CRUD, Right-to-Erasure cascading deletions
- JobRepository: CRUD and reverse job suggestions
- MatchReportRepository: Persistence of explainable fit reports
- Vector Search: Executes optimized pgvector SQL queries from core/db/sql/
- Dual-Engine Fallback: Fallback memory vector comparison when testing without active Postgres
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from core.db.models import (
    Base,
    JobModel,
    MatchReportModel,
    ResumeChunkModel,
    ResumeModel,
    ResumeSectionModel,
    ResumeSkillModel,
)
from resumefit.config import get_config

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"


def load_sql_query(filename: str) -> str:
    """Load SQL query text from core/db/sql/."""
    path = SQL_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"SQL file not found: {path}")


def _is_postgres(session: Session) -> bool:
    """Safely check if the active database session is connected to PostgreSQL."""
    bind = session.get_bind()
    url = getattr(bind, "url", None)
    if url is None and hasattr(bind, "engine"):
        url = getattr(bind.engine, "url", None)
    return "postgresql" in str(url) if url else False



class DatabaseSessionManager:
    """Manages SQLAlchemy engine and session lifecycle."""

    def __init__(self, db_url: str | None = None) -> None:
        self.url = db_url or get_config().database.url
        # If testing with SQLite, set check_same_thread=False
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        self.engine = create_engine(self.url, pool_pre_ping=True, connect_args=connect_args)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """Create tables directly (useful for SQLite / test environments)."""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Yield a new database session."""
        return self.session_factory()


class ResumeRepository:
    """Repository handling resume persistence, vector search, and right-to-erasure."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.config = get_config()

    def create_resume(
        self,
        resume_data: dict[str, Any],
        sections: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        skills: list[dict[str, Any]],
    ) -> ResumeModel:
        """Persist a complete resume with its sections, embedded chunks, and skills."""
        res_id = resume_data.get("id") or uuid.uuid4()
        resume = ResumeModel(
            id=res_id,
            filename=resume_data.get("filename", "resume.pdf"),
            total_pages=resume_data.get("total_pages", 1),
            total_experience_years=resume_data.get("total_experience_years", 0.0),
            highest_degree=resume_data.get("highest_degree"),
            candidate_name=resume_data.get("candidate_name"),
            candidate_email=resume_data.get("candidate_email"),
            candidate_phone=resume_data.get("candidate_phone"),
            redacted_text=resume_data.get("redacted_text", ""),
            raw_text=resume_data.get("raw_text"),
            is_stuffed=resume_data.get("is_stuffed", False),
            metadata_json=resume_data.get("metadata_json", {}),
        )
        self.session.add(resume)
        self.session.flush()

        # Add sections
        section_id_map: dict[str, uuid.UUID] = {}
        for s in sections:
            sec_id = s.get("id") or uuid.uuid4()
            section = ResumeSectionModel(
                id=sec_id,
                resume_id=res_id,
                section_type=s["section_type"],
                heading=s["heading"],
                text=s["text"],
                start_char=s.get("start_char", 0),
                end_char=s.get("end_char", 0),
            )
            self.session.add(section)
            section_id_map[s["section_type"]] = sec_id

        # Add embedded chunks
        for idx, c in enumerate(chunks):
            sec_id = c.get("section_id") or section_id_map.get(c["section_type"])
            chunk = ResumeChunkModel(
                id=c.get("id") or uuid.uuid4(),
                resume_id=res_id,
                section_id=sec_id,
                section_type=c["section_type"],
                chunk_index=c.get("chunk_index", idx),
                text=c["text"],
                token_count=c.get("token_count", 0),
                embedding=c["embedding"],
            )
            self.session.add(chunk)

        # Add skills
        for sk in skills:
            skill = ResumeSkillModel(
                id=sk.get("id") or uuid.uuid4(),
                resume_id=res_id,
                canonical_label=sk["canonical_label"],
                esco_uri=sk.get("esco_uri"),
                evidence_level=sk["evidence_level"],
                section_type=sk["section_type"],
                context_span=sk["context_span"],
                match_type=sk.get("match_type", "exact"),
            )
            self.session.add(skill)

        self.session.commit()
        return resume

    def get_resume(self, resume_id: uuid.UUID | str) -> ResumeModel | None:
        """Fetch resume by ID with all relationships."""
        uid = uuid.UUID(str(resume_id))
        stmt = select(ResumeModel).where(ResumeModel.id == uid)
        return self.session.scalars(stmt).first()

    def list_resumes(self, limit: int = 50, offset: int = 0) -> list[ResumeModel]:
        """List resumes paginated."""
        stmt = select(ResumeModel).order_by(ResumeModel.created_at.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt).all())

    def delete_resume(self, resume_id: uuid.UUID | str) -> bool:
        """Right-to-Erasure: Cascade delete resume and all associated chunks, skills, reports."""
        uid = uuid.UUID(str(resume_id))
        resume = self.get_resume(uid)
        if not resume:
            return False

        self.session.delete(resume)
        self.session.commit()
        return True

    def search_top_k_chunks(
        self,
        query_embedding: list[float],
        limit: int = 10,
        ef_search: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find top-K resume chunks using pgvector HNSW cosine distance."""
        ef = ef_search or self.config.database.hnsw_ef_search
        if _is_postgres(self.session):
            sql_text = load_sql_query("top_k_resumes.sql")
            # Replace SET LOCAL statement if needed or execute in transaction
            results = self.session.execute(
                text(sql_text),
                {"ef_search": ef, "query_embedding": str(query_embedding), "limit": limit},
            )
            rows = results.mappings().all()
            return [dict(r) for r in rows]

        # In-memory numpy fallback for SQLite test environments
        return self._search_chunks_in_memory(query_embedding, limit=limit)

    def search_section_restricted(
        self,
        query_embedding: list[float],
        allowed_sections: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search chunks restricted to specified section types."""
        if _is_postgres(self.session):
            sql_text = load_sql_query("section_restricted.sql")
            results = self.session.execute(
                text(sql_text),
                {
                    "ef_search": self.config.database.hnsw_ef_search,
                    "query_embedding": str(query_embedding),
                    "allowed_sections": allowed_sections,
                    "limit": limit,
                },
            )
            return [dict(r) for r in results.mappings().all()]

        # Fallback
        return self._search_chunks_in_memory(
            query_embedding, allowed_sections=allowed_sections, limit=limit
        )

    def _search_chunks_in_memory(
        self,
        query_embedding: list[float],
        allowed_sections: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """In-memory cosine similarity search for testing without active PostgreSQL."""
        stmt = select(ResumeChunkModel, ResumeModel.candidate_name).join(
            ResumeModel, ResumeModel.id == ResumeChunkModel.resume_id
        )
        if allowed_sections:
            stmt = stmt.where(ResumeChunkModel.section_type.in_(allowed_sections))

        results = self.session.execute(stmt).all()
        if not results:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        norm_val = float(np.linalg.norm(q_vec))
        q_norm: float = norm_val if norm_val > 0.0 else 1.0

        scored: list[dict[str, Any]] = []
        for chunk, cand_name in results:
            c_vec = np.array(chunk.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            sim = float(np.dot(q_vec, c_vec) / (q_norm * (c_norm or 1.0)))
            scored.append(
                {
                    "chunk_id": str(chunk.id),
                    "resume_id": str(chunk.resume_id),
                    "candidate_name": cand_name,
                    "section_type": chunk.section_type,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "cosine_similarity": max(0.0, min(1.0, sim)),
                }
            )

        scored.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return scored[:limit]


class JobRepository:
    """Repository handling job description entities and reverse matching."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.config = get_config()

    def create_job(self, job_data: dict[str, Any]) -> JobModel:
        """Persist a new job posting."""
        job = JobModel(
            id=job_data.get("id") or uuid.uuid4(),
            role_title=job_data["role_title"],
            raw_text=job_data["raw_text"],
            min_years_experience=job_data.get("min_years_experience", 0.0),
            min_degree=job_data.get("min_degree"),
            required_skills=job_data.get("required_skills", []),
            preferred_skills=job_data.get("preferred_skills", []),
            embedding=job_data["embedding"],
        )
        self.session.add(job)
        self.session.commit()
        return job

    def get_job(self, job_id: uuid.UUID | str) -> JobModel | None:
        """Fetch job posting by ID."""
        uid = uuid.UUID(str(job_id))
        stmt = select(JobModel).where(JobModel.id == uid)
        return self.session.scalars(stmt).first()

    def list_jobs(self, limit: int = 50) -> list[JobModel]:
        """List active jobs."""
        stmt = select(JobModel).order_by(JobModel.created_at.desc()).limit(limit)
        return list(self.session.scalars(stmt).all())

    def search_top_k_jobs(
        self,
        candidate_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find top matching job opportunities for a candidate embedding."""
        if _is_postgres(self.session):
            sql_text = load_sql_query("top_k_jobs.sql")
            results = self.session.execute(
                text(sql_text),
                {
                    "ef_search": self.config.database.hnsw_ef_search,
                    "candidate_embedding": str(candidate_embedding),
                    "limit": limit,
                },
            )
            return [dict(r) for r in results.mappings().all()]

        # Fallback in-memory
        jobs = self.list_jobs(limit=100)
        q_vec = np.array(candidate_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) or 1.0

        scored: list[dict[str, Any]] = []
        for j in jobs:
            j_vec = np.array(j.embedding, dtype=np.float32)
            j_norm = np.linalg.norm(j_vec) or 1.0
            sim = float(np.dot(q_vec, j_vec) / (q_norm * j_norm))
            scored.append(
                {
                    "job_id": str(j.id),
                    "role_title": j.role_title,
                    "min_years_experience": j.min_years_experience,
                    "min_degree": j.min_degree,
                    "required_skills": j.required_skills,
                    "preferred_skills": j.preferred_skills,
                    "cosine_similarity": max(0.0, min(1.0, sim)),
                }
            )

        scored.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return scored[:limit]


class MatchReportRepository:
    """Repository handling auditable match reports."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save_report(self, report_data: dict[str, Any]) -> MatchReportModel:
        """Persist a generated FitReport."""
        r_id = uuid.UUID(str(report_data["resume_id"])) if report_data.get("resume_id") else None
        j_id = uuid.UUID(str(report_data["job_id"])) if report_data.get("job_id") else None

        report = MatchReportModel(
            id=report_data.get("id") or uuid.uuid4(),
            resume_id=r_id,
            job_id=j_id,
            fit_index=report_data["fit_index"],
            verdict_headline=report_data["verdict_headline"],
            narrative_summary=report_data["narrative_summary"],
            scores_json=report_data.get("scores", {}),
            skills_analysis_json=report_data.get("skills_analysis", {}),
            constraints_json=report_data.get("constraints", {}),
            audit_metadata_json=report_data.get("audit_metadata", {}),
        )
        self.session.add(report)
        self.session.commit()
        return report

    def get_report(self, report_id: uuid.UUID | str) -> MatchReportModel | None:
        """Retrieve match report by ID."""
        uid = uuid.UUID(str(report_id))
        stmt = select(MatchReportModel).where(MatchReportModel.id == uid)
        return self.session.scalars(stmt).first()
