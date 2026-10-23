"""Unit tests for document precondition validation and scan detection."""

from pathlib import Path

import pytest

from core.ingestion.pipeline import process_resume
from core.ingestion.scan_detect import check_pdf_preconditions
from core.ingestion.types import UnsupportedDocument, UnsupportedReason

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pdfs"


def test_reject_non_pdf_content(tmp_path: Path) -> None:
    """Verify that a plain text file pretending to be PDF is rejected by magic bytes."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("This is plain text without PDF header.", encoding="utf-8")

    with pytest.raises(UnsupportedDocument) as exc_info:
        check_pdf_preconditions(fake_pdf)
    assert exc_info.value.reason == UnsupportedReason.NOT_A_PDF


def test_reject_encrypted_pdf() -> None:
    """Verify that password-protected PDFs are rejected loudly with ENCRYPTED_PDF."""
    pdf_path = FIXTURES_DIR / "encrypted_pw.pdf"
    with pytest.raises(UnsupportedDocument) as exc_info:
        process_resume(pdf_path)
    assert exc_info.value.reason == UnsupportedReason.ENCRYPTED_PDF


def test_reject_oversized_page_count() -> None:
    """Verify that PDFs exceeding the max page limit are rejected."""
    pdf_path = FIXTURES_DIR / "oversized_pages.pdf"
    with pytest.raises(UnsupportedDocument) as exc_info:
        process_resume(pdf_path)
    assert exc_info.value.reason == UnsupportedReason.EXCEEDS_PAGE_LIMIT


def test_reject_scanned_image_only_pdf() -> None:
    """Verify that raster image-only PDFs are rejected as SCANNED_OR_IMAGE_ONLY."""
    pdf_path = FIXTURES_DIR / "scanned_image_only.pdf"
    with pytest.raises(UnsupportedDocument) as exc_info:
        process_resume(pdf_path)
    assert exc_info.value.reason == UnsupportedReason.SCANNED_OR_IMAGE_ONLY


def test_reject_empty_blank_pdf() -> None:
    """Verify that blank PDFs are rejected."""
    pdf_path = FIXTURES_DIR / "empty_blank.pdf"
    with pytest.raises(UnsupportedDocument) as exc_info:
        process_resume(pdf_path)
    assert exc_info.value.reason == UnsupportedReason.SCANNED_OR_IMAGE_ONLY


def test_reject_non_english_resume() -> None:
    """Verify that resumes predominantly in non-English languages are rejected."""
    pdf_path = FIXTURES_DIR / "non_english_heavy.pdf"
    with pytest.raises(UnsupportedDocument) as exc_info:
        process_resume(pdf_path)
    assert exc_info.value.reason == UnsupportedReason.NON_ENGLISH
