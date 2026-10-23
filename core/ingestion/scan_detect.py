"""Document validation and scan detection.

Enforces Non-Negotiable Principle 3:
"Honest engineering: unsupported inputs are flagged loudly, never silently degraded."
Rejects:
- Non-PDF files (checks magic bytes, not just extension)
- Encrypted / password-protected PDFs
- Files > max_file_size_mb (default 5 MB)
- Files > max_pages (default 10 pages)
- Scanned / image-only PDFs (where character yield per page is below threshold)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pypdf

from core.ingestion.types import UnsupportedDocument, UnsupportedReason
from resumefit.config import get_config

# Standard PDF header magic bytes (%PDF-)
PDF_MAGIC_BYTES = b"%PDF-"


def check_pdf_preconditions(
    file_source: str | Path | bytes | BinaryIO,
) -> tuple[bytes, pypdf.PdfReader]:
    """Inspect raw bytes and PDF metadata for size, magic bytes, encryption, and page limits.

    Returns:
        tuple of (raw_bytes, initialized PdfReader)
    Raises:
        UnsupportedDocument if any precondition check fails.
    """
    config = get_config()
    max_bytes = config.ingestion.max_file_size_mb * 1024 * 1024
    max_pages = config.ingestion.max_pages

    # 1. Read bytes into memory
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.is_file():
            raise UnsupportedDocument(
                reason=UnsupportedReason.NOT_A_PDF,
                message=f"File not found: {path}",
            )
        file_size = path.stat().st_size
        if file_size > max_bytes:
            raise UnsupportedDocument(
                reason=UnsupportedReason.OVERSIZED_FILE,
                message=f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed size ({config.ingestion.max_file_size_mb} MB).",
                details={"file_size_bytes": file_size, "max_bytes": max_bytes},
            )
        with open(path, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_source, bytes):
        raw_bytes = file_source
        if len(raw_bytes) > max_bytes:
            raise UnsupportedDocument(
                reason=UnsupportedReason.OVERSIZED_FILE,
                message=f"File size ({len(raw_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed size ({config.ingestion.max_file_size_mb} MB).",
                details={"file_size_bytes": len(raw_bytes), "max_bytes": max_bytes},
            )
    elif hasattr(file_source, "read"):
        raw_bytes = file_source.read()
        if len(raw_bytes) > max_bytes:
            raise UnsupportedDocument(
                reason=UnsupportedReason.OVERSIZED_FILE,
                message=f"File size ({len(raw_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed size ({config.ingestion.max_file_size_mb} MB).",
                details={"file_size_bytes": len(raw_bytes), "max_bytes": max_bytes},
            )
    else:
        raise UnsupportedDocument(
            reason=UnsupportedReason.NOT_A_PDF,
            message=f"Unsupported file source type: {type(file_source)}",
        )

    # 2. Magic byte verification
    if not raw_bytes.startswith(PDF_MAGIC_BYTES) and PDF_MAGIC_BYTES not in raw_bytes[:1024]:
        raise UnsupportedDocument(
            reason=UnsupportedReason.NOT_A_PDF,
            message="Uploaded file is not a valid PDF document (missing PDF magic header).",
            details={"header_sample": raw_bytes[:16].decode("latin-1", errors="replace")},
        )

    # 3. Parse with pypdf to inspect encryption and page structure
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise UnsupportedDocument(
            reason=UnsupportedReason.CORRUPTED_PDF,
            message=f"Corrupted or malformed PDF file could not be parsed: {exc}",
        ) from exc

    # 4. Encryption check
    if reader.is_encrypted:
        raise UnsupportedDocument(
            reason=UnsupportedReason.ENCRYPTED_PDF,
            message="This PDF is encrypted or password-protected. Please provide an unencrypted PDF.",
        )

    # 5. Page count check
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise UnsupportedDocument(
            reason=UnsupportedReason.SCANNED_OR_IMAGE_ONLY,
            message="PDF contains 0 pages.",
        )

    if total_pages > max_pages:
        raise UnsupportedDocument(
            reason=UnsupportedReason.EXCEEDS_PAGE_LIMIT,
            message=f"Document has {total_pages} pages, which exceeds the allowed limit of {max_pages} pages.",
            details={"total_pages": total_pages, "max_pages": max_pages},
        )

    return raw_bytes, reader


def evaluate_scan_quality(pages_text: list[str]) -> None:
    """Verify that the extracted text across pages meets the extractable character threshold.

    Raises:
        UnsupportedDocument(reason=SCANNED_OR_IMAGE_ONLY) if text is absent or predominantly whitespace.
    """
    config = get_config()
    min_chars = config.ingestion.min_chars_per_page

    total_chars = sum(len(p.strip()) for p in pages_text)
    total_pages = max(1, len(pages_text))
    avg_chars_per_page = total_chars / total_pages

    if avg_chars_per_page < min_chars:
        raise UnsupportedDocument(
            reason=UnsupportedReason.SCANNED_OR_IMAGE_ONLY,
            message=(
                f"Document appears to be a scanned or image-only PDF (yielded {avg_chars_per_page:.1f} "
                f"chars/page, minimum required is {min_chars}). ResumeFit requires a text-based digital PDF."
            ),
            details={
                "avg_chars_per_page": avg_chars_per_page,
                "min_chars_required": min_chars,
                "total_pages": total_pages,
                "total_extractable_chars": total_chars,
            },
        )
