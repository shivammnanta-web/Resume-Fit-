"""Tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError

from resumefit.config import MatchingConfig, Settings, ThresholdsConfig


def test_load_default_config(default_config: Settings) -> None:
    """Verify that default config loads and validates all defaults."""
    assert default_config.app.name == "ResumeFit"
    assert default_config.matching.alpha == 0.4
    assert default_config.matching.beta == 0.6
    assert default_config.matching.alpha + default_config.matching.beta == 1.0
    assert default_config.thresholds.semantic_strong == 0.70
    assert default_config.thresholds.semantic_moderate == 0.45
    assert default_config.skills.fuzzy_threshold == 0.70
    assert "c" in default_config.skills.short_token_guard_tokens


def test_weights_sum_validation() -> None:
    """Verify that alpha + beta must sum to 1.0 when weighted fusion is selected."""
    with pytest.raises(ValidationError):
        MatchingConfig(alpha=0.5, beta=0.8, fusion_method="weighted")


def test_threshold_order_validation() -> None:
    """Verify that semantic_moderate must be strictly less than semantic_strong."""
    with pytest.raises(ValidationError):
        ThresholdsConfig(semantic_strong=0.50, semantic_moderate=0.80)
