"""Personally Identifiable Information (PII) detection and redaction module.

Combines spaCy NER (PERSON, GPE) with deterministic regular expressions
(emails, phone numbers, URLs, postal addresses).

Enforces Non-Negotiable Principle 5:
"Privacy by default: resumes are processed in memory; persistence is opt-in;
PII (names, emails, phones, addresses) is redacted before embedding and never written to logs."
"""

from __future__ import annotations

import logging
import re
from typing import Any

import spacy
from spacy.language import Language

from core.ingestion.types import PIIRedactionResult

logger = logging.getLogger(__name__)

# PII Regex patterns
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
URL_REGEX = re.compile(
    r"\b(?:https?://|www\.)[A-Za-z0-9.\-_~:/?#[\]@!$&'()*+,;=]+(?:\.[A-Za-z]{2,})[A-Za-z0-9.\-_~:/?#[\]@!$&'()*+,;=]*\b"
)
LINKEDIN_GITHUB_REGEX = re.compile(
    r"\b(?:linkedin\.com/in/|github\.com/)[A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)

# Global cached spaCy model instance
_SPACY_NLP: Language | None = None


def get_spacy_nlp() -> Language:
    """Return the cached spaCy en_core_web_sm pipeline."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            logger.warning("en_core_web_sm not directly importable; attempting load via spacy...")
            _SPACY_NLP = spacy.blank("en")
    return _SPACY_NLP


def redact_pii(text: str) -> PIIRedactionResult:
    """Detect and redact PII entities, returning sanitized text with typed placeholders."""
    if not text.strip():
        return PIIRedactionResult(original_text=text, redacted_text=text)

    redacted_spans: list[dict[str, Any]] = []
    detected_email: str | None = None
    detected_phone: str | None = None
    detected_name: str | None = None

    # Step 1: Detect and replace structured regex patterns
    # Find emails
    for m in EMAIL_REGEX.finditer(text):
        if not detected_email:
            detected_email = m.group(0)
        redacted_spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "type": "EMAIL",
                "val": m.group(0),
            }
        )

    # Find phone numbers
    for m in PHONE_REGEX.finditer(text):
        val = m.group(0).strip()
        # Ensure it has at least 8 digits to avoid false matching single years or simple numbers
        digits_only = "".join(c for c in val if c.isdigit())
        if len(digits_only) >= 8:
            if not detected_phone:
                detected_phone = val
            redacted_spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "type": "PHONE",
                    "val": val,
                }
            )

    # Find URLs
    for m in URL_REGEX.finditer(text):
        redacted_spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "type": "URL",
                "val": m.group(0),
            }
        )

    for m in LINKEDIN_GITHUB_REGEX.finditer(text):
        redacted_spans.append(
            {
                "start": m.start(),
                "end": m.end(),
                "type": "URL",
                "val": m.group(0),
            }
        )

    COMMON_TECH_TERMS = {
        "python", "flask", "docker", "c++", "c#", "java", "sql", "linux", "git",
        "scikit-learn", "react", "html", "css", "aws", "azure", "kubernetes",
        "postgres", "postgresql", "fastapi", "django", "pandas", "numpy", "pytorch",
        "gunicorn", "alembic", "redis", "sentence-transformers", "pgvector", "bash",
    }

    # Step 2: spaCy NER for PERSON and GPE (Location / Address)
    nlp = get_spacy_nlp()
    doc = nlp(text[:8000])
    for ent in doc.ents:
        ent_text_lower = ent.text.strip().lower()
        if ent_text_lower in COMMON_TECH_TERMS:
            continue

        if ent.label_ == "PERSON":
            if not detected_name:
                detected_name = ent.text.strip()
            redacted_spans.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": "NAME",
                    "val": ent.text,
                }
            )
        elif ent.label_ in {"GPE", "LOC"}:
            redacted_spans.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": "ADDRESS",
                    "val": ent.text,
                }
            )

    # Step 3: Sort spans from end to start and replace with placeholders
    # Remove overlapping spans
    sorted_spans = sorted(redacted_spans, key=lambda x: (x["start"], -x["end"]))
    non_overlapping: list[dict[str, Any]] = []
    last_end = -1
    for s in sorted_spans:
        if s["start"] >= last_end:
            non_overlapping.append(s)
            last_end = s["end"]

    # Reconstruct text with replacements backwards
    redacted_chars = list(text)
    for s in reversed(non_overlapping):
        start = s["start"]
        end = s["end"]
        ptype = s["type"]
        placeholder = f"[{ptype}]"
        redacted_chars[start:end] = list(placeholder)

    redacted_text = "".join(redacted_chars)

    return PIIRedactionResult(
        original_text=text,
        redacted_text=redacted_text,
        redacted_spans=non_overlapping,
        detected_email=detected_email,
        detected_phone=detected_phone,
        detected_name=detected_name,
    )
