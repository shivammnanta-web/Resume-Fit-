"""Dual PDF text extraction engine combining pdfplumber and pypdf.

Implements decision ADR-017 / Q17:
Run both extractors per page, pick the better output using a deterministic
quality heuristic (character density, garbled-token ratio, line length sanity).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO

import pdfplumber
import pypdf

from core.ingestion.scan_detect import check_pdf_preconditions, evaluate_scan_quality
from core.ingestion.types import ExtractionResult, PageExtractionResult

logger = logging.getLogger(__name__)


def compute_garbled_token_ratio(text: str) -> float:
    """Calculate the fraction of tokens in text that appear garbled/corrupted.

    A token is flagged as garbled if:
    - It contains unusual control characters or replacement characters (e.g. \ufffd, CID font artifacts)
    - It contains sequences of unprintable characters
    - It has an implausible consonant/vowel or symbol ratio
    """
    if not text.strip():
        return 1.0

    tokens = text.split()
    if not tokens:
        return 1.0

    garbled_count = 0
    for token in tokens:
        # Replacement character or common PDF font extraction corruption patterns
        if "\ufffd" in token or "(cid:" in token or "/CID" in token:
            garbled_count += 1
            continue

        # Very long sequence with no letters or digits
        alpha_count = sum(1 for c in token if c.isalnum())
        if len(token) > 3 and (alpha_count / len(token)) < 0.3:
            garbled_count += 1

    return garbled_count / len(tokens)


def evaluate_page_text_quality(text: str) -> tuple[int, float, float]:
    """Return quality metrics for page text: (char_count, garbled_ratio, line_sanity_score)."""
    cleaned = text.strip()
    char_count = len(cleaned)
    if char_count == 0:
        return 0, 1.0, 0.0

    garbled_ratio = compute_garbled_token_ratio(cleaned)

    # Line sanity: check if lines are not unnaturally fragmented (e.g. 1 word per line)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return char_count, garbled_ratio, 0.0

    avg_line_len = sum(len(line) for line in lines) / len(lines)
    # Prefer pages where average line length is in a healthy reading range (20 to 120 chars)
    if avg_line_len < 10:
        line_sanity = 0.4
    elif avg_line_len > 200:
        line_sanity = 0.7  # Likely collapsed paragraphs without newlines
    else:
        line_sanity = 1.0

    return char_count, garbled_ratio, line_sanity


def extract_page_with_pypdf(reader: pypdf.PdfReader, page_idx: int) -> str:
    """Extract text from a specific page using pypdf."""
    try:
        page = reader.pages[page_idx]
        text = page.extract_text() or ""
        return text
    except Exception as exc:
        logger.debug("pypdf failed on page %d: %s", page_idx + 1, exc)
        return ""


def extract_page_with_pdfplumber(pdf_bytes: bytes, page_idx: int) -> str:
    """Extract text from a specific page using pdfplumber (layout-aware)."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_idx < len(pdf.pages):
                page = pdf.pages[page_idx]
                # Extract text preserving approximate visual layout
                text = page.extract_text(layout=True) or page.extract_text() or ""
                return text
        return ""
    except Exception as exc:
        logger.debug("pdfplumber failed on page %d: %s", page_idx + 1, exc)
        return ""


def extract_pdf_text(
    file_source: str | Path | bytes | BinaryIO,
) -> ExtractionResult:
    """Extract text from a PDF document using dual-extractor strategy.

    Validates preconditions (magic bytes, size, encryption, pages), runs both
    extractors per page, chooses the higher-quality output, and verifies that
    the document is not scanned or empty.

    Returns:
        ExtractionResult containing page details and concatenated raw text.
    Raises:
        UnsupportedDocument on scanned, corrupted, encrypted, or invalid files.
    """
    raw_bytes, pypdf_reader = check_pdf_preconditions(file_source)
    total_pages = len(pypdf_reader.pages)

    page_results: list[PageExtractionResult] = []
    chosen_texts: list[str] = []

    for page_idx in range(total_pages):
        text_pypdf = extract_page_with_pypdf(pypdf_reader, page_idx)
        text_plumber = extract_page_with_pdfplumber(raw_bytes, page_idx)

        c_pypdf, g_pypdf, s_pypdf = evaluate_page_text_quality(text_pypdf)
        c_plumb, g_plumb, s_plumb = evaluate_page_text_quality(text_plumber)

        # Composite quality score: Higher char yield (capped), lower garbled ratio, line sanity
        # Score = (min(chars, 3000) / 1000) * (1.0 - garbled_ratio) * line_sanity
        score_pypdf = (min(c_pypdf, 3000) / 1000.0) * (1.0 - g_pypdf) * s_pypdf
        score_plumber = (min(c_plumb, 3000) / 1000.0) * (1.0 - g_plumb) * s_plumb

        if score_plumber >= score_pypdf:
            chosen_extractor = "pdfplumber"
            chosen_text = text_plumber
            chosen_char_count = c_plumb
            chosen_garbled = g_plumb
        else:
            chosen_extractor = "pypdf"
            chosen_text = text_pypdf
            chosen_char_count = c_pypdf
            chosen_garbled = g_pypdf

        logger.debug(
            "Page %d: chosen %s (score=%.2f vs %.2f, chars=%d, garbled=%.2f)",
            page_idx + 1,
            chosen_extractor,
            max(score_plumber, score_pypdf),
            min(score_plumber, score_pypdf),
            chosen_char_count,
            chosen_garbled,
        )

        page_results.append(
            PageExtractionResult(
                page_number=page_idx + 1,
                extractor_chosen=chosen_extractor,
                char_count=chosen_char_count,
                garbled_token_ratio=chosen_garbled,
                raw_text=chosen_text,
                metadata={
                    "pypdf_chars": c_pypdf,
                    "pdfplumber_chars": c_plumb,
                    "pypdf_garbled": g_pypdf,
                    "pdfplumber_garbled": g_plumb,
                },
            )
        )
        chosen_texts.append(chosen_text)

    # Enforce scan check: Fail loudly if characters per page are below threshold
    evaluate_scan_quality(chosen_texts)

    avg_garbled = (
        sum(p.garbled_token_ratio for p in page_results) / len(page_results)
        if page_results
        else 0.0
    )

    combined_text = "\n\n".join(chosen_texts)

    return ExtractionResult(
        total_pages=total_pages,
        pages=page_results,
        combined_raw_text=combined_text,
        is_scanned=False,
        average_garbled_ratio=avg_garbled,
    )
