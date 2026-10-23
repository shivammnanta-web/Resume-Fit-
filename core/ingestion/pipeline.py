"""High-level ingestion pipeline orchestrator.

Coordinates:
1. check_pdf_preconditions & extract_pdf_text
2. preprocess_text (unicode repair, de-hyphenation, bullet normalization)
3. detect_sections & compute_total_experience_years
4. redact_pii
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.ingestion.extract import extract_pdf_text
from core.ingestion.pii import redact_pii
from core.ingestion.preprocess import preprocess_text
from core.ingestion.sections import (
    compute_total_experience_years,
    detect_sections,
    parse_experience_entries,
)
from core.ingestion.types import DateRange, ExperienceEntry, Section, SectionType


class IngestionOutput(BaseModel):
    """Complete structured output from the resume ingestion phase."""

    filename: str
    total_pages: int
    raw_character_count: int
    cleaned_character_count: int
    sections: list[Section]
    experience_entries: list[ExperienceEntry] = Field(default_factory=list)
    total_experience_years: float = 0.0
    detected_candidate_name: str | None = None
    detected_email: str | None = None
    detected_phone: str | None = None
    redacted_full_text: str
    unredacted_full_text: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def process_resume(
    file_source: str | Path | bytes, filename: str = "resume.pdf"
) -> IngestionOutput:
    """Execute the end-to-end ingestion pipeline on a resume PDF.

    Raises:
        UnsupportedDocument if the file is scanned, encrypted, corrupted, non-PDF, or non-English.
    """
    fname = Path(file_source).name if isinstance(file_source, (str, Path)) else filename

    # Step 1: Extraction with dual-extractor & scan detection
    extraction = extract_pdf_text(file_source)
    page_texts = [p.raw_text for p in extraction.pages]

    # Step 2: Preprocessing
    cleaned_text = preprocess_text(extraction.combined_raw_text, page_texts=page_texts)

    # Step 3: Section segmentation
    sections = detect_sections(cleaned_text)

    # Step 4: Parse experience dates & calculate total experience
    exp_sections = [s for s in sections if s.section_type == SectionType.EXPERIENCE]
    all_date_ranges: list[DateRange] = []
    exp_entries: list[ExperienceEntry] = []

    for s in exp_sections:
        entries = parse_experience_entries(s.text)
        exp_entries.extend(entries)
        for e in entries:
            if e.date_range:
                all_date_ranges.append(e.date_range)

    total_years = compute_total_experience_years(all_date_ranges)

    # Step 5: PII detection and redaction
    pii_result = redact_pii(cleaned_text)

    return IngestionOutput(
        filename=fname,
        total_pages=extraction.total_pages,
        raw_character_count=len(extraction.combined_raw_text),
        cleaned_character_count=len(cleaned_text),
        sections=sections,
        experience_entries=exp_entries,
        total_experience_years=total_years,
        detected_candidate_name=pii_result.detected_name,
        detected_email=pii_result.detected_email,
        detected_phone=pii_result.detected_phone,
        redacted_full_text=pii_result.redacted_text,
        unredacted_full_text=cleaned_text,
    )


def process_resume_text(
    text: str, filename: str = "pasted_resume.txt", candidate_name: str | None = None
) -> IngestionOutput:
    """Execute the ingestion pipeline directly on raw/pasted resume text."""
    cleaned_text = preprocess_text(text)
    sections = detect_sections(cleaned_text)

    exp_sections = [s for s in sections if s.section_type == SectionType.EXPERIENCE]
    all_date_ranges: list[DateRange] = []
    exp_entries: list[ExperienceEntry] = []

    for s in exp_sections:
        entries = parse_experience_entries(s.text)
        exp_entries.extend(entries)
        for e in entries:
            if e.date_range:
                all_date_ranges.append(e.date_range)

    total_years = compute_total_experience_years(all_date_ranges)
    pii_result = redact_pii(cleaned_text)

    return IngestionOutput(
        filename=filename,
        total_pages=1,
        raw_character_count=len(text),
        cleaned_character_count=len(cleaned_text),
        sections=sections,
        experience_entries=exp_entries,
        total_experience_years=total_years,
        detected_candidate_name=candidate_name or pii_result.detected_name,
        detected_email=pii_result.detected_email,
        detected_phone=pii_result.detected_phone,
        redacted_full_text=pii_result.redacted_text,
        unredacted_full_text=cleaned_text,
    )

