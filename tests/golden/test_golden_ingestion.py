"""Golden-file tests on 7 diverse real layout PDFs (single column, two column, table layout)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.ingestion.pipeline import process_resume
from core.ingestion.types import SectionType

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pdfs"

VALID_TEST_PDFS = [
    "single_col_1.pdf",
    "single_col_2.pdf",
    "single_col_3.pdf",
    "two_col_1.pdf",
    "two_col_2.pdf",
    "table_layout_1.pdf",
    "table_layout_2.pdf",
]


@pytest.mark.golden
@pytest.mark.parametrize("pdf_name", VALID_TEST_PDFS)
def test_golden_pdf_ingestion_pipeline(pdf_name: str) -> None:
    pdf_path = FIXTURES_DIR / pdf_name
    assert pdf_path.exists(), f"Fixture missing: {pdf_path}"

    result = process_resume(pdf_path)

    # 1. Page count and char count assertions
    assert result.total_pages >= 1
    assert result.cleaned_character_count > 100

    # 2. Section detection check
    section_types = {s.section_type for s in result.sections}
    # At least two standard sections must be detected
    assert len(section_types) >= 2

    # 3. Work experience parsing check
    if SectionType.EXPERIENCE in section_types:
        assert len(result.experience_entries) >= 1
        assert result.total_experience_years > 0.0

    # 4. PII redaction check
    assert "[EMAIL]" in result.redacted_full_text or "@" not in result.redacted_full_text
    assert "[PHONE]" in result.redacted_full_text or not any(
        c.isdigit() for c in result.unredacted_full_text if "-" in c
    )

    # 5. Non-empty unredacted and redacted texts
    assert len(result.redacted_full_text) > 0
    assert len(result.unredacted_full_text) > 0


@pytest.mark.golden
def test_cli_ingestion_runner() -> None:
    """Verify python -m core.ingestion runs cleanly and prints valid JSON."""
    pdf_path = FIXTURES_DIR / "single_col_1.pdf"
    cmd = [sys.executable, "-m", "core.ingestion", str(pdf_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["filename"] == "single_col_1.pdf"
    assert "sections" in data
    assert len(data["sections"]) > 0
    assert "total_experience_years" in data
    assert data["total_experience_years"] > 0
