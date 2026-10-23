"""Unit tests for section detection and date parsing."""

from core.ingestion.sections import (
    compute_total_experience_years,
    detect_sections,
    parse_date_range,
    parse_experience_entries,
)
from core.ingestion.types import SectionType


def test_detect_standard_sections(sample_resume_text: str) -> None:
    sections = detect_sections(sample_resume_text)
    types_found = {s.section_type for s in sections}

    assert SectionType.SUMMARY in types_found
    assert SectionType.EXPERIENCE in types_found
    assert SectionType.PROJECTS in types_found
    assert SectionType.EDUCATION in types_found
    assert SectionType.SKILLS in types_found

    # Verify offsets accuracy
    for s in sections:
        assert s.end_char > s.start_char
        if s.section_type != SectionType.PERSONAL_DETAILS:
            assert s.heading in sample_resume_text[s.start_char : s.end_char]


def test_parse_date_range() -> None:
    dr = parse_date_range("Jan 2021", "Dec 2022")
    assert dr.start_year == 2021
    assert dr.start_month == 1
    assert dr.end_year == 2022
    assert dr.end_month == 12
    assert dr.duration_months == 24.0


def test_compute_total_experience_years_merged_overlapping() -> None:
    # Two overlapping roles: Jan 2020 - Dec 2021 (24 mo) and Jun 2020 - Jun 2022 (25 mo)
    # Total merged: Jan 2020 - Jun 2022 = 30 months = 2.5 years
    dr1 = parse_date_range("Jan 2020", "Dec 2021")
    dr2 = parse_date_range("Jun 2020", "Jun 2022")

    total_years = compute_total_experience_years([dr1, dr2])
    assert total_years == 2.5


def test_parse_experience_entries() -> None:
    exp_text = """
Software Engineer | Acme Corp
Jan 2022 – Present | Bengaluru
• Developed web APIs using Flask and Python.

Junior Developer | Startup Inc
Jan 2020 – Dec 2021 | Pune
• Maintained database schema in PostgreSQL.
"""
    entries = parse_experience_entries(exp_text)
    assert len(entries) == 2
    assert entries[0].company == "Acme Corp" or "Acme Corp" in entries[0].text
    assert entries[0].years > 0
    assert entries[1].years == 2.0
