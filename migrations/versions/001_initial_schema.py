"""Initial database schema with pgvector HNSW and GIN full-text search indexes (Tasks P4.2, P4.4, P4.5).

Revision ID: 001_initial_schema
Revises: None
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create 'resumes' table
    op.create_table(
        "resumes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("total_pages", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("total_experience_years", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("highest_degree", sa.String(50), nullable=True),
        sa.Column("candidate_name", sa.String(255), nullable=True),
        sa.Column("candidate_email", sa.String(255), nullable=True),
        sa.Column("candidate_phone", sa.String(50), nullable=True),
        sa.Column("redacted_text", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("is_stuffed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
    )

    # 3. Create 'resume_sections' table
    op.create_table(
        "resume_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_type", sa.String(50), nullable=False),
        sa.Column("heading", sa.String(100), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("end_char", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_resume_sections_resume_id", "resume_sections", ["resume_id"])

    # 4. Create 'resume_chunks' table with 384-dimensional vector embedding
    op.create_table(
        "resume_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("resume_sections.id", ondelete="CASCADE"), nullable=True),
        sa.Column("section_type", sa.String(50), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", Vector(384), nullable=False),
    )
    op.create_index("ix_resume_chunks_resume_id", "resume_chunks", ["resume_id"])

    # Task P4.4: HNSW Vector Index with vector_cosine_ops, m=16, ef_construction=64
    op.execute(
        """
        CREATE INDEX ix_resume_chunks_embedding
        ON resume_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # Task P4.5: GIN index for PostgreSQL full-text search
    op.execute(
        """
        CREATE INDEX ix_resume_chunks_tsv
        ON resume_chunks
        USING gin (to_tsvector('english', text));
        """
    )

    # 5. Create 'resume_skills' table
    op.create_table(
        "resume_skills",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("canonical_label", sa.String(255), nullable=False),
        sa.Column("esco_uri", sa.String(255), nullable=True),
        sa.Column("evidence_level", sa.String(50), nullable=False),
        sa.Column("section_type", sa.String(50), nullable=False),
        sa.Column("context_span", sa.Text(), nullable=False),
        sa.Column("match_type", sa.String(50), nullable=False, server_default="exact"),
    )
    op.create_index("ix_resume_skills_resume_id", "resume_skills", ["resume_id"])
    op.create_index("ix_resume_skills_canonical", "resume_skills", ["canonical_label"])

    # 6. Create 'jobs' table
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("role_title", sa.String(255), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("min_years_experience", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("min_degree", sa.String(50), nullable=True),
        sa.Column("required_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("preferred_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )

    # HNSW and GIN indexes on jobs table
    op.execute(
        """
        CREATE INDEX ix_jobs_embedding
        ON jobs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )
    op.execute(
        """
        CREATE INDEX ix_jobs_tsv
        ON jobs
        USING gin (to_tsvector('english', raw_text));
        """
    )

    # 7. Create 'match_reports' table
    op.create_table(
        "match_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("resumes.id", ondelete="CASCADE"), nullable=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("fit_index", sa.Float(), nullable=False),
        sa.Column("verdict_headline", sa.String(255), nullable=False),
        sa.Column("narrative_summary", sa.Text(), nullable=False),
        sa.Column("scores_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("skills_analysis_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("constraints_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("audit_metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_match_reports_resume_id", "match_reports", ["resume_id"])
    op.create_index("ix_match_reports_job_id", "match_reports", ["job_id"])


def downgrade() -> None:
    # Task P4.12: Clean downgrade dropping all tables, indexes, and vector extension
    op.drop_table("match_reports")
    op.drop_table("jobs")
    op.drop_table("resume_skills")
    op.drop_table("resume_chunks")
    op.drop_table("resume_sections")
    op.drop_table("resumes")
    op.execute("DROP EXTENSION IF EXISTS vector;")
