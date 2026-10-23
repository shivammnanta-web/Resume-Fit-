"""Pydantic schemas and error envelope definitions for ResumeFit API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error object contained inside standard error envelopes."""

    code: str
    message: str
    details: Any = None


class ErrorEnvelope(BaseModel):
    """Standardized API error response envelope."""

    error: ErrorDetail


class AnalyzeTextRequest(BaseModel):
    """Request payload for text-based resume and JD matching."""

    resume_text: str = Field(..., min_length=20, description="Full raw resume text")
    jd_text: str = Field(..., min_length=20, description="Full job description text")
    candidate_name: str | None = Field(default=None, description="Optional candidate display name")
    filename: str = Field(default="pasted_resume.txt", description="Virtual file identifier")


class RankCandidateItem(BaseModel):
    """Individual candidate record for batch ranking."""

    name: str = Field(..., min_length=1, description="Candidate name")
    text: str = Field(..., min_length=20, description="Resume text")
    filename: str = Field(default="resume.txt", description="Source filename")


class RankRequest(BaseModel):
    """Batch ranking request matching multiple candidates against a target JD."""

    jd_text: str = Field(..., min_length=20, description="Target job description")
    candidates: list[RankCandidateItem] = Field(
        ..., min_length=1, max_length=50, description="List of candidate resumes to rank"
    )


class SuggestJobsRequest(BaseModel):
    """Request to recommend job opportunities for a given resume."""

    resume_text: str = Field(..., min_length=20, description="Resume text to find jobs for")
    candidate_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class DeleteResponse(BaseModel):
    """Response returned upon Right-to-Erasure execution."""

    deleted: bool
    id: str
    message: str


class HealthResponse(BaseModel):
    """System health check payload."""

    status: str = "ok"
    app: str = "ResumeFit"
    version: str = "0.1.0"
    environment: str = "development"
    database_connected: bool = True
