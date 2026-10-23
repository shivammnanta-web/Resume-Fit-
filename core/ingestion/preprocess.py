"""Text preprocessing and cleaning pipeline for extracted PDF text.

Performs:
- Unicode repair (ligature expansion, quotation/dash normalization)
- De-hyphenation (rejoining split words across line breaks)
- Line rejoining (mid-sentence continuation)
- Bullet normalization
- Header / footer deduplication across multi-page resumes
- Non-English character ratio assessment
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from core.ingestion.types import UnsupportedDocument, UnsupportedReason
from resumefit.config import get_config

# Common typographic ligatures in PDF fonts
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

# Bullet character variants to normalize into standard bullet
BULLET_VARIANTS = re.compile(
    r"^[\s]*[•\*\u2022\u25cf\u25aa\u25e6\u25cb\u2023\u2043\u2219▪►–—\-][\s]+", re.MULTILINE
)

# De-hyphenation pattern: word ending with hyphen followed by newline and continuation
DEHYPHEN_REGEX = re.compile(r"(\b[A-Za-z]{2,})-\s*\n\s*([a-z]{2,}\b)")

# Repeated page numbering patterns (e.g. "Page 1 of 2", "1 / 3", "- 2 -")
PAGE_NUMBER_REGEX = re.compile(
    r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|-\s*\d+\s*-)\s*$", re.IGNORECASE
)


def repair_unicode(text: str) -> str:
    """Normalize unicode representations, expand font ligatures, and normalize quotes/dashes."""
    # Expand standard ligatures first
    for lig, replacement in LIGATURE_MAP.items():
        text = text.replace(lig, replacement)

    # Normalize unicode to NFKC (combining characters to precomposed)
    text = unicodedata.normalize("NFKC", text)

    # Replace smart quotes with standard ASCII equivalents
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("`", "'")

    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # Normalize unicode dashes to standard hyphen
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )

    return text


def normalize_bullets(text: str) -> str:
    """Standardize disparate bullet glyphs at line starts to '• '."""
    lines = []
    for line in text.splitlines():
        if BULLET_VARIANTS.match(line):
            cleaned_line = BULLET_VARIANTS.sub("• ", line)
            lines.append(cleaned_line)
        else:
            lines.append(line)
    return "\n".join(lines)


def dehyphenate(text: str) -> str:
    """Rejoin words split across line breaks with a hyphen (e.g. 'develop-\\nment' -> 'development')."""
    return DEHYPHEN_REGEX.sub(r"\1\2", text)


def remove_repeated_headers_footers(page_texts: list[str]) -> list[str]:
    """Identify and remove lines that repeat identically across multiple pages."""
    if len(page_texts) <= 1:
        return page_texts

    # Collect first and last 3 non-empty lines from each page
    boundary_lines: list[str] = []
    for p in page_texts:
        lines = [line.strip() for line in p.splitlines() if line.strip()]
        if lines:
            boundary_lines.extend(lines[:2])
            boundary_lines.extend(lines[-2:])

    counts = Counter(boundary_lines)
    repeated = {line for line, cnt in counts.items() if cnt >= len(page_texts) and len(line) > 3}

    cleaned_pages: list[str] = []
    for p in page_texts:
        out_lines: list[str] = []
        for line in p.splitlines():
            sline = line.strip()
            if sline in repeated or PAGE_NUMBER_REGEX.match(sline):
                continue
            out_lines.append(line)
        cleaned_pages.append("\n".join(out_lines))

    return cleaned_pages


def rejoin_broken_lines(text: str) -> str:
    """Rejoin lines broken mid-sentence while preserving bullet points and headings."""
    lines = text.splitlines()
    if not lines:
        return ""

    rejoined: list[str] = []
    current_buffer = ""

    for line in lines:
        # Collapse multiple spaces on the line
        collapsed = re.sub(r"[ \t]+", " ", line)
        stripped = collapsed.strip()
        if not stripped:
            if current_buffer:
                rejoined.append(current_buffer)
                current_buffer = ""
            rejoined.append("")
            continue

        # Never merge if current line or previous buffer is a bullet or heading or title/separator
        is_bullet = stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*")
        is_heading_curr = stripped.isupper() and len(stripped) < 60
        is_heading_prev = current_buffer.isupper() and len(current_buffer) < 60
        has_separator = "|" in stripped or "|" in current_buffer

        if not current_buffer:
            current_buffer = stripped
        elif is_bullet or is_heading_curr or is_heading_prev or has_separator:
            rejoined.append(current_buffer)
            current_buffer = stripped
        else:
            ends_with_terminal = current_buffer[-1] in {".", ":", ";", "!", "?"}
            starts_with_lower = stripped[0].islower()

            if starts_with_lower or (
                not ends_with_terminal and len(current_buffer) > 60 and not stripped[0].isupper()
            ):
                current_buffer += " " + stripped
            else:
                rejoined.append(current_buffer)
                current_buffer = stripped

    if current_buffer:
        rejoined.append(current_buffer)

    return "\n".join(rejoined)


def check_language_precondition(text: str) -> None:
    """Enforce English-only scope per ADR-029 / Q29.

    Rejects documents where non-ASCII character ratio exceeds the configured threshold.
    """
    config = get_config()
    threshold = config.ingestion.non_english_ascii_ratio_threshold

    non_ws_chars = [c for c in text if not c.isspace()]
    if not non_ws_chars:
        return

    ascii_chars = sum(1 for c in non_ws_chars if ord(c) < 128)
    ascii_ratio = ascii_chars / len(non_ws_chars)

    if ascii_ratio < threshold:
        raise UnsupportedDocument(
            reason=UnsupportedReason.NON_ENGLISH,
            message=(
                f"Resume appears to be in an unsupported non-English language (ASCII ratio: {ascii_ratio:.2%}, "
                f"expected >= {threshold:.0%}). ResumeFit currently supports English-language resumes only."
            ),
            details={"ascii_ratio": ascii_ratio, "threshold": threshold},
        )


def preprocess_text(text: str, page_texts: list[str] | None = None) -> str:
    """Full preprocessing pipeline for extracted resume text."""
    if page_texts and len(page_texts) > 1:
        cleaned_pages = remove_repeated_headers_footers(page_texts)
        text = "\n\n".join(cleaned_pages)

    text = repair_unicode(text)
    check_language_precondition(text)
    text = dehyphenate(text)
    text = normalize_bullets(text)
    text = rejoin_broken_lines(text)

    # Collapse excessive multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text
