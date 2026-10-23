"""Typed data structures and enumerations for the ingestion pipeline.

Ensures every module is independently testable with a clear typed interface
(Non-Negotiable Principle 8).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SectionType(StrEnum):
    """Normalized section classifications according to ESCO & standard resume semantics."""

    SUMMARY = "summary"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    SKILLS = "skills"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"
    PERSONAL_DETAILS = "personal_details"
    OTHER = "other"


class UnsupportedReason(StrEnum):
    """Actionable rejection categories for unsupported input documents."""

    SCANNED_OR_IMAGE_ONLY = "scanned/image-only"
    NOT_A_PDF = "not-a-pdf"
    ENCRYPTED_PDF = "encrypted-pdf"
    OVERSIZED_FILE = "file-too-large"
    EXCEEDS_PAGE_LIMIT = "too-many-pages"
    NON_ENGLISH = "non-english"
    CORRUPTED_PDF = "corrupted-pdf"


class UnsupportedDocument(Exception):
    """Typed exception raised when a document cannot be processed by the text pipeline.

    Carries an actionable reason and user-facing diagnostic explanation.
    """

    def __init__(
        self, reason: UnsupportedReason, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "unsupported_document",
            "reason": self.reason.value,
            "message": self.message,
            "details": self.details,
        }


class PageExtractionResult(BaseModel):
    """Extraction metrics and chosen text for an individual PDF page."""

    page_number: int
    extractor_chosen: str  # "pdfplumber" or "pypdf"
    char_count: int
    garbled_token_ratio: float
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Full extraction result across all pages of a PDF."""

    total_pages: int
    pages: list[PageExtractionResult]
    combined_raw_text: str
    is_scanned: bool = False
    average_garbled_ratio: float = 0.0


class Section(BaseModel):
    """A detected resume section with text and character offsets in the cleaned text."""

    section_type: SectionType
    heading: str
    text: str
    start_char: int
    end_char: int


class DateRange(BaseModel):
    """Parsed date range for an experience or project entry."""

    start_str: str
    end_str: str
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False
    duration_months: float = 0.0


class ExperienceEntry(BaseModel):
    """Structured experience item extracted from the Work Experience section."""

    title: str | None = None
    company: str | None = None
    date_range: DateRange | None = None
    text: str
    years: float = 0.0


class PIIRedactionResult(BaseModel):
    """Result of PII detection and sanitization."""

    original_text: str
    redacted_text: str
    redacted_spans: list[dict[str, Any]] = Field(default_factory=list)
    detected_email: str | None = None
    detected_phone: str | None = None
    detected_name: str | None = None
