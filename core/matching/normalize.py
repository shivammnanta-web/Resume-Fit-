"""Score normalization and calibration module (Task P3.6).

Aligns score distributions between lexical (TF-IDF cosine ~ [0.05, 0.50]) and
semantic (Sentence-Transformers cosine ~ [0.35, 0.85]) retrieval channels.
Supports:
1. Min-Max normalization with empirical clipping.
2. Percentile calibration mapping raw cosine scores to empirical percentiles.
3. Identity passthrough (for baseline ablation experiments).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("data/artifacts/calibration.json")

# Default empirical distribution quantiles derived from engineering benchmark pairs
DEFAULT_LEXICAL_QUANTILES = [0.00, 0.08, 0.15, 0.22, 0.30, 0.40, 0.52, 0.65, 0.80, 1.00]
DEFAULT_SEMANTIC_QUANTILES = [0.20, 0.35, 0.45, 0.55, 0.62, 0.68, 0.74, 0.80, 0.88, 1.00]


@dataclass
class ChannelBounds:
    """Min, max, and quantiles for a retrieval channel."""

    min_val: float
    max_val: float
    quantiles: list[float]  # 10 deciles (0th, 10th, ..., 100th)


class ScoreNormalizer:
    """Normalizes and calibrates raw channel scores into comparable [0.0, 1.0] scale."""

    def __init__(
        self,
        method: Literal["percentile", "minmax", "identity"] = "percentile",
        calibration_path: Path | str | None = None,
    ) -> None:
        self.method = method
        self.calibration_path = Path(calibration_path or CALIBRATION_FILE)
        self.lexical_bounds, self.semantic_bounds = self._load_or_init_bounds()

    def _load_or_init_bounds(self) -> tuple[ChannelBounds, ChannelBounds]:
        if self.calibration_path.exists():
            try:
                with open(self.calibration_path, encoding="utf-8") as f:
                    data = json.load(f)
                lex = ChannelBounds(**data["lexical"])
                sem = ChannelBounds(**data["semantic"])
                return lex, sem
            except Exception as e:
                logger.warning("Failed to load calibration from %s: %s", self.calibration_path, e)

        # Use robust empirical defaults
        lex = ChannelBounds(
            min_val=0.02,
            max_val=0.75,
            quantiles=DEFAULT_LEXICAL_QUANTILES,
        )
        sem = ChannelBounds(
            min_val=0.25,
            max_val=0.88,
            quantiles=DEFAULT_SEMANTIC_QUANTILES,
        )
        return lex, sem

    def save_calibration(self) -> None:
        """Persist current calibration boundaries to artifact file."""
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "lexical": {
                "min_val": self.lexical_bounds.min_val,
                "max_val": self.lexical_bounds.max_val,
                "quantiles": self.lexical_bounds.quantiles,
            },
            "semantic": {
                "min_val": self.semantic_bounds.min_val,
                "max_val": self.semantic_bounds.max_val,
                "quantiles": self.semantic_bounds.quantiles,
            },
        }
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved score calibration to %s", self.calibration_path)

    def _minmax(self, val: float, bounds: ChannelBounds) -> float:
        if bounds.max_val <= bounds.min_val:
            return float(np.clip(val, 0.0, 1.0))
        scaled = (val - bounds.min_val) / (bounds.max_val - bounds.min_val)
        return float(np.clip(scaled, 0.0, 1.0))

    def _percentile(self, val: float, bounds: ChannelBounds) -> float:
        qs = bounds.quantiles
        if not qs or len(qs) < 2:
            return self._minmax(val, bounds)

        # Linear interpolation across deciles
        decile_percentiles = np.linspace(0.0, 1.0, len(qs))
        # np.interp expects increasing x
        pct = float(np.interp(val, qs, decile_percentiles))
        return float(np.clip(pct, 0.0, 1.0))

    def normalize_lexical(self, raw_score: float) -> float:
        """Normalize raw TF-IDF cosine score."""
        if self.method == "identity":
            return float(np.clip(raw_score, 0.0, 1.0))
        if self.method == "minmax":
            return self._minmax(raw_score, self.lexical_bounds)
        return self._percentile(raw_score, self.lexical_bounds)

    def normalize_semantic(self, raw_score: float) -> float:
        """Normalize raw semantic cosine score."""
        if self.method == "identity":
            return float(np.clip(raw_score, 0.0, 1.0))
        if self.method == "minmax":
            return self._minmax(raw_score, self.semantic_bounds)
        return self._percentile(raw_score, self.semantic_bounds)
