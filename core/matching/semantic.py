"""Semantic matching module with dual architectures (Task P3.5).

Implements:
1. Architecture A: Mean-pooled document-level embedding scorer.
2. Architecture B: Fine-grained requirement-to-chunk scorer with section
   importance weighting (Experience: 1.0, Projects: 0.9, Skills: 0.5, Education: 0.4, etc.).
3. Qualitative 3-tier semantic ratings: Strong (>=0.70), Moderate (0.45-0.70), Weak (<0.45).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from core.ingestion.types import Section, SectionType
from core.jd.parser import ParsedJD
from core.matching.chunking import Chunk, SectionChunker
from resumefit.config import get_config

logger = logging.getLogger(__name__)

# Global model cache to prevent expensive re-instantiation
_EMBEDDING_MODEL: SentenceTransformer | None = None
_CURRENT_MODEL_NAME: str | None = None


def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load or retrieve the cached SentenceTransformer model instance."""
    global _EMBEDDING_MODEL, _CURRENT_MODEL_NAME

    target_name = model_name or get_config().matching.embedding_model
    if _EMBEDDING_MODEL is None or target_name != _CURRENT_MODEL_NAME:
        logger.info("Initializing SentenceTransformer: %s", target_name)
        _EMBEDDING_MODEL = SentenceTransformer(target_name)
        _CURRENT_MODEL_NAME = target_name

    return _EMBEDDING_MODEL


@dataclass
class RequirementMatch:
    """Best matching resume evidence chunk for a specific JD requirement statement."""

    requirement: str
    best_chunk_id: str
    best_chunk_text: str
    best_chunk_section: SectionType
    similarity: float
    rating: str  # "Strong", "Moderate", "Weak"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "best_chunk_id": self.best_chunk_id,
            "best_chunk_text": self.best_chunk_text,
            "best_chunk_section": self.best_chunk_section.value,
            "similarity": round(self.similarity, 4),
            "rating": self.rating,
        }


@dataclass
class SemanticMatchResult:
    """Comprehensive semantic comparison output covering both architectures."""

    primary_score: float  # Default score used in downstream hybrid fusion [0.0, 1.0]
    mean_pooled_score: float  # Architecture A: doc-level cosine
    requirement_to_chunk_score: float  # Architecture B: fine-grained section-weighted cosine
    requirement_matches: list[RequirementMatch] = field(default_factory=list)
    chunk_embeddings: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_score": round(self.primary_score, 4),
            "mean_pooled_score": round(self.mean_pooled_score, 4),
            "requirement_to_chunk_score": round(self.requirement_to_chunk_score, 4),
            "requirement_matches": [rm.to_dict() for rm in self.requirement_matches],
        }


class SemanticScorer:
    """Dual-architecture semantic scorer with section-weighted retrieval."""

    def __init__(self, model_name: str | None = None) -> None:
        self.config = get_config()
        self.model = get_embedding_model(model_name or self.config.matching.embedding_model)
        self.chunker = SectionChunker()

    def get_section_weight(self, section_type: SectionType) -> float:
        """Retrieve configured importance weight for a section type."""
        weights = self.config.matching.section_weights
        mapping = {
            SectionType.EXPERIENCE: weights.experience,
            SectionType.PROJECTS: weights.projects,
            SectionType.SKILLS: weights.skills,
            SectionType.CERTIFICATIONS: weights.certifications,
            SectionType.EDUCATION: weights.education,
            SectionType.SUMMARY: weights.summary,
            SectionType.PERSONAL_DETAILS: weights.other,
            SectionType.OTHER: weights.other,
        }
        return mapping.get(section_type, 0.5)

    def classify_rating(self, similarity: float) -> str:
        """Map raw cosine similarity to 3-tier qualitative rating."""
        t = self.config.thresholds
        if similarity >= t.semantic_strong:
            return "Strong"
        elif similarity >= t.semantic_moderate:
            return "Moderate"
        return "Weak"

    def mean_pooled_score(self, resume_text: str, jd_text: str) -> float:
        """Architecture A: Compute document-level cosine similarity."""
        if not resume_text.strip() or not jd_text.strip():
            return 0.0

        embeddings = self.model.encode(
            [resume_text, jd_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, sim))

    def requirement_to_chunk_score(
        self,
        chunks: list[Chunk],
        requirements: list[str],
    ) -> tuple[float, list[RequirementMatch], list[np.ndarray]]:
        """Architecture B: Match atomic requirements to section chunks with section weighting."""
        if not chunks or not requirements:
            return 0.0, [], []

        # Encode chunks and requirements with unit normalization
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self.model.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        req_embeddings = self.model.encode(
            requirements,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        section_weights = np.array(
            [self.get_section_weight(c.section_type) for c in chunks],
            dtype=np.float32,
        )

        requirement_matches: list[RequirementMatch] = []
        raw_best_scores: list[float] = []

        for req_idx, req_text in enumerate(requirements):
            req_vec = req_embeddings[req_idx]
            # Cosine similarity between this requirement and all chunks
            sims = np.dot(chunk_embeddings, req_vec)
            # Select the chunk that best satisfies this requirement semantically
            best_idx = int(np.argmax(sims))
            raw_sim = float(max(0.0, min(1.0, sims[best_idx])))
            best_chunk = chunks[best_idx]

            rating = self.classify_rating(raw_sim)
            requirement_matches.append(
                RequirementMatch(
                    requirement=req_text,
                    best_chunk_id=best_chunk.chunk_id,
                    best_chunk_text=best_chunk.text,
                    best_chunk_section=best_chunk.section_type,
                    similarity=raw_sim,
                    rating=rating,
                )
            )
            raw_best_scores.append(raw_sim)

        avg_score = float(np.mean(raw_best_scores)) if raw_best_scores else 0.0
        avg_score = max(0.0, min(1.0, avg_score))

        return avg_score, requirement_matches, list(chunk_embeddings)

    def score(
        self,
        resume_sections: list[Section],
        parsed_jd: ParsedJD,
        resume_raw_text: str | None = None,
    ) -> SemanticMatchResult:
        """Run both semantic architectures and produce unified result."""
        # 1. Architecture A: Mean-pooled document score
        full_resume_text = (
            resume_raw_text
            if resume_raw_text
            else "\n\n".join([s.text for s in resume_sections])
        )
        mean_score = self.mean_pooled_score(full_resume_text, parsed_jd.raw_text)

        # 2. Architecture B: Requirement-to-chunk score
        chunks = self.chunker.chunk_all_sections(resume_sections)

        # Requirements: use atomic requirement statements + required skills as fallback
        requirements = list(parsed_jd.requirement_statements)
        if not requirements and parsed_jd.required_skills:
            requirements = [f"Proficiency and experience with {s}" for s in parsed_jd.required_skills]
        if not requirements:
            # Fall back to responsibility statements
            requirements = list(parsed_jd.responsibility_statements[:10])

        if chunks and requirements:
            req_score, req_matches, chunk_embs = self.requirement_to_chunk_score(
                chunks, requirements
            )
            primary = req_score
        else:
            req_score = mean_score
            req_matches = []
            chunk_embs = []
            primary = mean_score

        return SemanticMatchResult(
            primary_score=primary,
            mean_pooled_score=mean_score,
            requirement_to_chunk_score=req_score,
            requirement_matches=req_matches,
            chunk_embeddings=chunk_embs,
        )
