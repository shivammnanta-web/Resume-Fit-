"""Optional cross-encoder reranking module (Task P3.9).

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank candidate documents.
Disabled by default via config (matching.cross_encoder_enabled=false) to fit
Render free-tier memory constraints while remaining available for evaluation (E3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from resumefit.config import get_config

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL: Any | None = None


def get_cross_encoder(model_name: str | None = None) -> Any:
    """Lazy loader for cross-encoder model."""
    global _CROSS_ENCODER_MODEL
    target_name = model_name or get_config().matching.cross_encoder_model

    if _CROSS_ENCODER_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Initializing CrossEncoder: %s", target_name)
            _CROSS_ENCODER_MODEL = CrossEncoder(target_name)
        except Exception as e:
            logger.warning("Could not load CrossEncoder %s: %s", target_name, e)
            return None

    return _CROSS_ENCODER_MODEL


@dataclass
class RerankResult:
    """Outcome of reranking candidates."""

    candidate_id: str
    original_score: float
    rerank_score: float
    rank: int


class CandidateReranker:
    """Applies cross-encoder scoring when enabled via configuration."""

    def __init__(self, enabled: bool | None = None, model_name: str | None = None) -> None:
        cfg = get_config().matching
        self.enabled = enabled if enabled is not None else cfg.cross_encoder_enabled
        self.model_name = model_name or cfg.cross_encoder_model

    def rerank(
        self,
        jd_text: str,
        candidates: list[dict[str, Any]],
        text_key: str = "text",
        _id_key: str = "id",
        score_key: str = "score",
    ) -> list[dict[str, Any]]:
        """Rerank candidates if cross-encoder is enabled; otherwise preserve order."""
        if not self.enabled or not candidates or not jd_text.strip():
            return candidates

        model = get_cross_encoder(self.model_name)
        if model is None:
            logger.warning("CrossEncoder model unavailable, returning original ranking.")
            return candidates

        pairs = [[jd_text, c.get(text_key, "")] for c in candidates]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
            # Attach rerank score and sort
            augmented: list[dict[str, Any]] = []
            for idx, c in enumerate(candidates):
                item = dict(c)
                item["original_score"] = item.get(score_key, 0.0)
                item["rerank_score"] = float(scores[idx])
                augmented.append(item)

            augmented.sort(key=lambda x: x["rerank_score"], reverse=True)
            return augmented
        except Exception as e:
            logger.error("Reranking failed: %s. Preserving original order.", e)
            return candidates
