"""ESCO-grounded skill extraction pipeline with match cascade and keyword-stuffing detection.

Implements decision ADR-005 and tasks P2.5 through P2.9:
1. Candidate extraction via tech token regex + spaCy noun-chunks / n-grams
2. Match cascade: Exact ESCO match -> Alias overlay match -> Guarded fuzzy match (rapidfuzz)
3. Strict short-token guards (min 3 chars; contextual validation for C, R, Go)
4. ESCO taxonomy hierarchy & relation linking
5. Contextual evidence classification (Demonstrated / Listed-only / Tentative)
6. Keyword-stuffing detector
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import spacy
from rapidfuzz import fuzz
from spacy.language import Language

from core.ingestion.types import Section
from core.skills.aliases import AliasIndex, get_cached_alias_index
from core.skills.esco_loader import EscoTaxonomy, get_cached_taxonomy
from core.skills.evidence import (
    EvidenceLevel,
    SkillEvidence,
    classify_evidence_level,
)
from resumefit.config import get_config

logger = logging.getLogger(__name__)

# Specialized regex to preserve technical tokens with punctuation
TECH_TOKEN_PATTERNS = [
    re.compile(r"\bC\+\+\b", re.IGNORECASE),
    re.compile(r"\bC#\b", re.IGNORECASE),
    re.compile(r"\b\.NET(?:\s+Core)?\b", re.IGNORECASE),
    re.compile(r"\bASP\.NET(?:\s+Core)?\b", re.IGNORECASE),
    re.compile(r"\bNode\.js\b", re.IGNORECASE),
    re.compile(r"\bVue\.js\b", re.IGNORECASE),
    re.compile(r"\bReact\.js\b", re.IGNORECASE),
    re.compile(r"\bNext\.js\b", re.IGNORECASE),
    re.compile(r"\bD3\.js\b", re.IGNORECASE),
    re.compile(r"\bChart\.js\b", re.IGNORECASE),
    re.compile(r"\bElectron\.js\b", re.IGNORECASE),
    re.compile(r"\bCI/CD\b", re.IGNORECASE),
    re.compile(r"\bscikit-learn\b", re.IGNORECASE),
    re.compile(r"\bSentence-Transformers\b", re.IGNORECASE),
    re.compile(r"\bpgvector\b", re.IGNORECASE),
    re.compile(r"\bOAuth\s*2(?:\.0)?\b", re.IGNORECASE),
    re.compile(r"\bRESTful\s+APIs?\b", re.IGNORECASE),
    re.compile(r"\bREST\s+APIs?\b", re.IGNORECASE),
]

# Context patterns required to validate ambiguous short tokens
C_LANG_CONTEXT = re.compile(
    r"(?:languages?:.*?\bC\b)|(?:\bC\s*[/,]\s*(?:C\+\+|Python|Java|C#))|(?:\b(?:C\+\+|Python|Java|C#)\s*[/,]\s*C\b)|(?:\bC\s+programming\b)|(?:\bC\s+language\b)|(?:\bANSI\s+C\b)",
    re.IGNORECASE,
)
R_LANG_CONTEXT = re.compile(
    r"(?:languages?:.*?\bR\b)|(?:\bR\s*[/,]\s*(?:Python|SQL|SAS|SPSS))|(?:\b(?:Python|SQL|SAS|SPSS)\s*[/,]\s*R\b)|(?:\bR\s+programming\b)|(?:\bR\s+statistical\b)|(?:\bR\s+language\b)",
    re.IGNORECASE,
)
GO_LANG_CONTEXT = re.compile(
    r"(?:languages?:.*?\bGo\b)|(?:\bGo\s*[/,]\s*(?:Python|Rust|Java|C\+\+))|(?:\b(?:Python|Rust|Java|C\+\+)\s*[/,]\s*Go\b)|(?:\bGo\s+programming\b)|(?:\bGo\s+language\b)|(?:\bGolang\b)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def get_spacy_nlp() -> Language:
    """Return cached spaCy English model with entity & parser disabled where unneeded."""
    return spacy.load("en_core_web_sm")


@dataclass
class KeywordStuffingReport:
    """Diagnostic report evaluating potential keyword-stuffing patterns."""

    is_stuffed: bool
    listed_only_count: int
    demonstrated_count: int
    ratio: float
    threshold: float
    warning_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_stuffed": self.is_stuffed,
            "listed_only_count": self.listed_only_count,
            "demonstrated_count": self.demonstrated_count,
            "ratio": round(self.ratio, 2),
            "threshold": self.threshold,
            "warning_message": self.warning_message,
        }


@dataclass
class ExtractedSkill:
    """A canonical skill extracted from resume or job description text."""

    canonical_label: str
    esco_uri: str | None
    category: str = "skill/competence"
    evidence_level: EvidenceLevel = EvidenceLevel.LISTED_ONLY
    evidence_records: list[SkillEvidence] = field(default_factory=list)
    related_esco_skills: list[str] = field(default_factory=list)
    match_type: str = "exact"  # "exact", "alias", or "fuzzy"
    confidence: float = 1.0

    def add_evidence(self, ev: SkillEvidence) -> None:
        """Add an evidence occurrence and elevate evidence level if demonstrated."""
        self.evidence_records.append(ev)
        # Hierarchy: DEMONSTRATED > LISTED_ONLY > TENTATIVE
        if ev.evidence_level == EvidenceLevel.DEMONSTRATED:
            self.evidence_level = EvidenceLevel.DEMONSTRATED
        elif (
            ev.evidence_level == EvidenceLevel.LISTED_ONLY
            and self.evidence_level == EvidenceLevel.TENTATIVE
        ):
            self.evidence_level = EvidenceLevel.LISTED_ONLY


@dataclass
class SkillExtractionResult:
    """Complete aggregated results of skill extraction across all sections."""

    skills: list[ExtractedSkill]
    stuffing_report: KeywordStuffingReport

    @property
    def canonical_names(self) -> list[str]:
        return [s.canonical_label for s in self.skills]

    @property
    def demonstrated_skills(self) -> list[str]:
        return [
            s.canonical_label for s in self.skills if s.evidence_level == EvidenceLevel.DEMONSTRATED
        ]

    @property
    def listed_only_skills(self) -> list[str]:
        return [
            s.canonical_label for s in self.skills if s.evidence_level == EvidenceLevel.LISTED_ONLY
        ]

    @property
    def tentative_skills(self) -> list[str]:
        return [
            s.canonical_label for s in self.skills if s.evidence_level == EvidenceLevel.TENTATIVE
        ]

    def get_skill(self, canonical_label: str) -> ExtractedSkill | None:
        norm = canonical_label.strip().lower()
        for s in self.skills:
            if s.canonical_label.strip().lower() == norm:
                return s
        return None


class SkillExtractor:
    """Robust skill extraction engine combining ESCO taxonomy, alias overlay, and fuzzy guards."""

    def __init__(
        self,
        taxonomy: EscoTaxonomy | None = None,
        alias_index: AliasIndex | None = None,
    ) -> None:
        self.config = get_config()
        self.taxonomy = taxonomy or get_cached_taxonomy()
        self.alias_index = alias_index or get_cached_alias_index()
        self.nlp = get_spacy_nlp()

        self.fuzzy_threshold = self.config.skills.fuzzy_threshold
        self.min_token_length = self.config.skills.min_token_length
        self.short_tokens = set(self.config.skills.short_token_guard_tokens)

        # Precompile lookup terms for fuzzy candidate matching
        self._all_target_terms: dict[str, tuple[str, str | None, str, str]] = {}
        # normalized_term -> (canonical_label, esco_uri, category, match_type)

        # 1. Terms from alias overlay
        for alias_key, alias_obj in self.alias_index.aliases.items():
            if len(alias_key) >= self.min_token_length and alias_key not in self.short_tokens:
                self._all_target_terms[alias_key] = (
                    alias_obj.canonical_label,
                    alias_obj.parent_esco_uri,
                    alias_obj.category,
                    "alias",
                )

        # 2. Terms from ESCO taxonomy
        for label_key, uri in self.taxonomy.skills_by_label.items():
            if len(label_key) >= self.min_token_length and label_key not in self.short_tokens:
                skill = self.taxonomy.skills_by_uri.get(uri)
                if skill and label_key not in self._all_target_terms:
                    self._all_target_terms[label_key] = (
                        skill.preferred_label,
                        skill.concept_uri,
                        "esco_skill",
                        "exact",
                    )

    def is_valid_short_token(self, token: str, surrounding_text: str) -> bool:
        """Validate whether a 1-2 char programming language token is legitimately referenced."""
        tok = token.strip().lower()
        if tok == "c":
            # Must not be substring of CSS, CI/CD, C++, C#
            return bool(C_LANG_CONTEXT.search(surrounding_text))
        elif tok == "r":
            # Must not be substring of REST, Redis, R&D
            return bool(R_LANG_CONTEXT.search(surrounding_text))
        elif tok == "go":
            # Must not be the common verb 'go'
            return bool(GO_LANG_CONTEXT.search(surrounding_text))
        return False

    def match_cascade(
        self,
        candidate: str,
        surrounding_text: str,
    ) -> tuple[str, str | None, str, str, float] | None:
        """Execute matching cascade: Exact -> Alias -> Fuzzy.

        Returns:
            (canonical_label, esco_uri, category, match_type, confidence) or None.
        """
        norm = candidate.strip().lower()
        if not norm:
            return None

        # Guard for short ambiguous tokens (C, R, Go)
        if norm in self.short_tokens:
            if not self.is_valid_short_token(norm, surrounding_text):
                return None
            # Validated short token: resolve to canonical
            short_token_map = {
                "c": (
                    "C (programming language)",
                    "http://data.europa.eu/esco/skill/c-lang",
                    "language",
                    "exact",
                    1.0,
                ),
                "r": (
                    "R (programming language)",
                    "http://data.europa.eu/esco/skill/r-lang",
                    "language",
                    "exact",
                    1.0,
                ),
                "go": ("Go", "http://data.europa.eu/esco/skill/go-lang", "language", "exact", 1.0),
            }
            if norm in short_token_map:
                return short_token_map[norm]

        # Step 1: Exact Alias match
        alias_record = self.alias_index.resolve(norm)
        if alias_record:
            return (
                alias_record.canonical_label,
                alias_record.parent_esco_uri,
                alias_record.category,
                "alias",
                1.0,
            )

        # Step 2: Exact ESCO match (preferredLabel or altLabel)
        esco_skill = self.taxonomy.find_skill_by_label(norm)
        if esco_skill:
            return (
                esco_skill.preferred_label,
                esco_skill.concept_uri,
                "esco_skill",
                "exact",
                1.0,
            )

        # Step 3: Fuzzy match (guarded by min length, stop word filters, and threshold)
        if len(norm) < 4 or norm in self.nlp.Defaults.stop_words:
            return None

        best_score = 0.0
        best_match: tuple[str, str | None, str, str] | None = None

        # Compare against pre-indexed terms
        for target_key, info in self._all_target_terms.items():
            # Acronyms and short abbreviations (<= 4 chars) require exact match only
            if len(target_key) <= 4:
                continue

            # Length guard: true typos have almost identical length
            if abs(len(norm) - len(target_key)) > 2:
                continue

            score = fuzz.ratio(norm, target_key) / 100.0
            if score > best_score:
                best_score = score
                best_match = info

        # Effective threshold: ensure at least 0.82 for fuzzy similarity
        effective_threshold = max(self.fuzzy_threshold, 0.82)
        if best_match and best_score >= effective_threshold:
            canonical, uri, cat, _ = best_match
            return canonical, uri, cat, "fuzzy", round(best_score, 2)

        return None

    def extract_candidates_from_text(self, text: str) -> list[tuple[str, int, int]]:
        """Extract candidate phrases and tech tokens along with their character offsets."""
        candidates: list[tuple[str, int, int]] = []
        seen_spans: set[tuple[int, int]] = set()

        # 1. Tech tokens via regex
        for pattern in TECH_TOKEN_PATTERNS:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    candidates.append((match.group(), match.start(), match.end()))

        # 2. Extract noun chunks and sub-parts via spaCy
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            # Strip leading determiners ('the', 'a', 'an')
            clean_text = chunk.text.strip()
            if clean_text.lower().startswith(("a ", "an ", "the ")):
                first_space = clean_text.find(" ")
                clean_text = clean_text[first_space + 1 :]
                start_c = chunk.start_char + first_space + 1
            else:
                start_c = chunk.start_char
            end_c = start_c + len(clean_text)

            span = (start_c, end_c)
            if span not in seen_spans and len(clean_text) >= 2:
                seen_spans.add(span)
                candidates.append((clean_text, start_c, end_c))

            # Split conjunctions / lists within noun chunks (e.g. 'Flask, PostgreSQL, and Redis')
            sub_parts = [
                p.strip() for p in re.split(r"[,/|]|\b(?:and|or)\b", clean_text) if p.strip()
            ]
            if len(sub_parts) > 1:
                for sub in sub_parts:
                    sub_pos = text.find(sub, start_c)
                    if sub_pos != -1:
                        sub_span = (sub_pos, sub_pos + len(sub))
                        if sub_span not in seen_spans and len(sub) >= 2:
                            seen_spans.add(sub_span)
                            candidates.append((sub, sub_pos, sub_pos + len(sub)))

        # 3. Individual alphanumeric tokens for languages, databases, and single-word tools
        for token in doc:
            clean_tok = token.text.strip()
            if token.is_stop or token.is_punct or token.like_num or len(clean_tok) < 2:
                continue
            span = (token.idx, token.idx + len(clean_tok))
            if span not in seen_spans and (
                clean_tok.isalnum() or clean_tok in {"C++", "C#", ".NET"}
            ):
                seen_spans.add(span)
                candidates.append((clean_tok, token.idx, token.idx + len(clean_tok)))

        return candidates

    def get_related_esco_labels(self, uri: str | None) -> list[str]:
        """Fetch related concept names from ESCO taxonomy hierarchy."""
        if not uri:
            return []
        related_skills = self.taxonomy.get_related_skills(uri)
        return [r.preferred_label for r in related_skills]

    def extract_from_sections(self, sections: list[Section]) -> SkillExtractionResult:
        """Perform end-to-end skill extraction and evidence attribution across all resume sections."""
        skills_map: dict[str, ExtractedSkill] = {}

        for sec in sections:
            lines = sec.text.splitlines()
            for line in lines:
                sline = line.strip()
                if not sline:
                    continue

                candidates = self.extract_candidates_from_text(sline)
                for cand_text, start_idx, end_idx in candidates:
                    match_res = self.match_cascade(cand_text, sline)
                    if not match_res:
                        continue

                    canonical, esco_uri, category, match_type, conf = match_res
                    evidence_lvl, action_verbs = classify_evidence_level(sline, sec.section_type)

                    ev_record = SkillEvidence(
                        skill_name=cand_text,
                        canonical_label=canonical,
                        esco_uri=esco_uri,
                        evidence_level=evidence_lvl,
                        section_type=sec.section_type,
                        context_span=sline,
                        start_char=sec.start_char + start_idx,
                        end_char=sec.start_char + end_idx,
                        action_verbs=action_verbs,
                        is_alias=(match_type == "alias"),
                        match_type=match_type,
                    )

                    if canonical not in skills_map:
                        related = self.get_related_esco_labels(esco_uri)
                        skills_map[canonical] = ExtractedSkill(
                            canonical_label=canonical,
                            esco_uri=esco_uri,
                            category=category,
                            evidence_level=evidence_lvl,
                            evidence_records=[ev_record],
                            related_esco_skills=related,
                            match_type=match_type,
                            confidence=conf,
                        )
                    else:
                        skills_map[canonical].add_evidence(ev_record)

        extracted_skills = list(skills_map.values())

        # Evaluate keyword stuffing
        stuffing_report = self.evaluate_keyword_stuffing(extracted_skills)

        return SkillExtractionResult(
            skills=extracted_skills,
            stuffing_report=stuffing_report,
        )

    def evaluate_keyword_stuffing(self, skills: list[ExtractedSkill]) -> KeywordStuffingReport:
        """Flag resumes where uncorroborated listed-only skills disproportionately exceed demonstrated skills."""
        listed_cnt = sum(1 for s in skills if s.evidence_level == EvidenceLevel.LISTED_ONLY)
        demonstrated_cnt = sum(1 for s in skills if s.evidence_level == EvidenceLevel.DEMONSTRATED)

        ratio = listed_cnt / max(demonstrated_cnt, 1)
        threshold = self.config.skills.stuffing_ratio_threshold
        max_uncorroborated = self.config.skills.max_uncorroborated_skills

        is_stuffed = (ratio >= threshold) and (listed_cnt >= max_uncorroborated)

        warning_msg = None
        if is_stuffed:
            warning_msg = (
                f"High volume of uncorroborated skills detected ({listed_cnt} listed vs {demonstrated_cnt} "
                f"demonstrated in projects/experience; ratio {ratio:.1f} >= {threshold:.1f}). "
                f"Candidates should substantiate claimed skills with concrete project outcomes."
            )

        return KeywordStuffingReport(
            is_stuffed=is_stuffed,
            listed_only_count=listed_cnt,
            demonstrated_count=demonstrated_cnt,
            ratio=ratio,
            threshold=threshold,
            warning_message=warning_msg,
        )
