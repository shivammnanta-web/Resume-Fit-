"""Skill evidence classification and contextual span extraction (ADR-006).

Classifies extracted skills into three deterministic confidence tiers:
1. Demonstrated — Appears in Work Experience or Projects with action verbs (e.g. built, deployed, optimized).
2. Listed-only — Appears in Skills lists, certifications, or bulleted tool lists without demonstrated context.
3. Tentative — Expressed with hedging or learning-intent keywords (e.g. learning, familiar with, exposure to).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from core.ingestion.types import SectionType


class EvidenceLevel(StrEnum):
    """Tier of empirical evidence backing a skill claim."""

    DEMONSTRATED = "demonstrated"
    LISTED_ONLY = "listed_only"
    TENTATIVE = "tentative"


# Action verbs indicating demonstrated, active capability in real projects
ACTION_VERBS = {
    "architected",
    "authored",
    "automated",
    "built",
    "configured",
    "created",
    "delivered",
    "deployed",
    "designed",
    "developed",
    "engineered",
    "established",
    "executed",
    "implemented",
    "improved",
    "integrated",
    "launched",
    "led",
    "maintained",
    "managed",
    "migrated",
    "modeled",
    "optimised",
    "optimized",
    "orchestrated",
    "overhauled",
    "programmed",
    "provisioned",
    "reduced",
    "refactored",
    "resolved",
    "scaled",
    "streamlined",
    "supervised",
    "trained",
    "upgraded",
}

# Hedging phrases signaling tentative or novice familiarity
TENTATIVE_PATTERNS = [
    re.compile(r"\b(?:familiar\s+with)\b", re.IGNORECASE),
    re.compile(r"\b(?:exposure\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(?:basic\s+knowledge\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(?:learning|beginner|novice)\b", re.IGNORECASE),
    re.compile(r"\b(?:interested\s+in)\b", re.IGNORECASE),
    re.compile(r"\b(?:working\s+knowledge\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(?:academic\s+exposure)\b", re.IGNORECASE),
    re.compile(r"\b(?:pursuing|exploring)\b", re.IGNORECASE),
]


@dataclass
class SkillEvidence:
    """Detailed contextual evidence backing a candidate skill occurrence."""

    skill_name: str
    canonical_label: str
    esco_uri: str | None
    evidence_level: EvidenceLevel
    section_type: SectionType
    context_span: str
    start_char: int
    end_char: int
    action_verbs: list[str] = field(default_factory=list)
    is_alias: bool = False
    match_type: str = "exact"  # "exact", "alias", or "fuzzy"


def detect_action_verbs(text: str) -> list[str]:
    """Identify action verbs present in the sentence/line."""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    found: list[str] = []
    for token in tokens:
        if token in ACTION_VERBS and token not in found:
            found.append(token)
    return found


def is_tentative_span(text: str) -> bool:
    """Check whether a text span contains hedging or learning-intent keywords."""
    return any(pattern.search(text) for pattern in TENTATIVE_PATTERNS)


def classify_evidence_level(
    context_span: str,
    section_type: SectionType,
) -> tuple[EvidenceLevel, list[str]]:
    """Determine the evidence tier for a skill occurrence.

    Rules:
    - If hedging/learning indicators are present -> TENTATIVE
    - If in EXPERIENCE or PROJECTS section with action verbs -> DEMONSTRATED
    - If in SKILLS, EDUCATION, or personal details -> LISTED_ONLY
    - Fallback: if in EXPERIENCE/PROJECTS without action verbs -> LISTED_ONLY
    """
    if is_tentative_span(context_span):
        return EvidenceLevel.TENTATIVE, []

    action_verbs = detect_action_verbs(context_span)

    if section_type in (SectionType.EXPERIENCE, SectionType.PROJECTS):
        if action_verbs:
            return EvidenceLevel.DEMONSTRATED, action_verbs
        return EvidenceLevel.LISTED_ONLY, []

    return EvidenceLevel.LISTED_ONLY, []
