"""Hybrid retrieval fusion engine (Task P3.7).

Combines lexical (TF-IDF) and semantic (Sentence-Transformers) channels using:
1. Weighted Linear Combination: alpha * S_lex + beta * S_sem (default alpha=0.4, beta=0.6).
2. Reciprocal Rank Fusion (RRF): Sum 1 / (k + rank_m) across channels (default k=60).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from core.matching.lexical import LexicalMatchResult
from core.matching.normalize import ScoreNormalizer
from core.matching.semantic import SemanticMatchResult
from resumefit.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class HybridMatchScore:
    """Detailed score breakdown across all channels."""

    lexical_raw: float
    lexical_normalized: float
    semantic_raw: float
    semantic_normalized: float
    alpha: float
    beta: float
    hybrid_score: float  # [0.0, 1.0]
    fit_index: float  # [0.0, 100.0]
    fusion_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lexical_raw": round(self.lexical_raw, 4),
            "lexical_normalized": round(self.lexical_normalized, 4),
            "semantic_raw": round(self.semantic_raw, 4),
            "semantic_normalized": round(self.semantic_normalized, 4),
            "alpha": self.alpha,
            "beta": self.beta,
            "hybrid_score": round(self.hybrid_score, 4),
            "fit_index": round(self.fit_index, 1),
            "fusion_method": self.fusion_method,
        }


class HybridFusionEngine:
    """Fuses lexical and semantic channels with config-driven weights and normalizers."""

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        fusion_method: Literal["weighted", "rrf"] | None = None,
        rrf_k: int | None = None,
        normalizer: ScoreNormalizer | None = None,
    ) -> None:
        cfg = get_config().matching
        self.alpha = alpha if alpha is not None else cfg.alpha
        self.beta = beta if beta is not None else cfg.beta
        self.fusion_method = fusion_method or cfg.fusion_method
        self.rrf_k = rrf_k if rrf_k is not None else cfg.rrf_k
        self.normalizer = normalizer or ScoreNormalizer(method=cfg.normalizer)

        # Validate weights sum to 1.0 for weighted fusion
        if self.fusion_method == "weighted":
            total = round(self.alpha + self.beta, 6)
            if abs(total - 1.0) > 1e-4:
                raise ValueError(
                    f"Hybrid weights must sum to 1.0; received alpha={self.alpha}, beta={self.beta} (sum={total})"
                )

    def fuse(
        self,
        lexical_result: LexicalMatchResult,
        semantic_result: SemanticMatchResult,
    ) -> HybridMatchScore:
        """Combine single-pair lexical and semantic scores."""
        lex_norm = self.normalizer.normalize_lexical(lexical_result.score)
        sem_norm = self.normalizer.normalize_semantic(semantic_result.primary_score)

        if self.fusion_method == "weighted":
            score = (self.alpha * lex_norm) + (self.beta * sem_norm)
            score = max(0.0, min(1.0, score))
        else:
            # Fallback for single pair when RRF configured
            score = (self.alpha * lex_norm) + (self.beta * sem_norm)
            score = max(0.0, min(1.0, score))

        fit_idx = round(score * 100.0, 1)

        return HybridMatchScore(
            lexical_raw=lexical_result.score,
            lexical_normalized=lex_norm,
            semantic_raw=semantic_result.primary_score,
            semantic_normalized=sem_norm,
            alpha=self.alpha,
            beta=self.beta,
            hybrid_score=score,
            fit_index=fit_idx,
            fusion_method=self.fusion_method,
        )

    def fuse_rankings(
        self,
        candidate_ids: list[str],
        lexical_scores: list[float],
        semantic_scores: list[float],
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion across a batch of candidates."""
        # Rank by lexical descending
        lex_order = sorted(
            range(len(candidate_ids)),
            key=lambda i: lexical_scores[i],
            reverse=True,
        )
        lex_ranks = {candidate_ids[idx]: rank + 1 for rank, idx in enumerate(lex_order)}

        # Rank by semantic descending
        sem_order = sorted(
            range(len(candidate_ids)),
            key=lambda i: semantic_scores[i],
            reverse=True,
        )
        sem_ranks = {candidate_ids[idx]: rank + 1 for rank, idx in enumerate(sem_order)}

        rrf_scores: list[tuple[str, float]] = []
        for cid in candidate_ids:
            r_lex = lex_ranks[cid]
            r_sem = sem_ranks[cid]
            rrf = (1.0 / (self.rrf_k + r_lex)) + (1.0 / (self.rrf_k + r_sem))
            rrf_scores.append((cid, rrf))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores
