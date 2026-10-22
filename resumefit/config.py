"""Configuration loader and schema validation for ResumeFit.

Loads default configuration from config/default.yaml and supports
environment variable overrides prefixed with RESUMEFIT_.
Enforces non-negotiable principle 6: Config over constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "ResumeFit"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    secret_key: str = "dev-insecure-secret-key-change-in-production"
    rate_limit_analyze: str = "10 per minute"
    rate_limit_default: str = "30 per minute"


class IngestionConfig(BaseModel):
    min_chars_per_page: int = Field(default=50, ge=1)
    max_file_size_mb: int = Field(default=5, ge=1, le=50)
    max_pages: int = Field(default=10, ge=1, le=100)
    allowed_mime_types: list[str] = Field(default_factory=lambda: ["application/pdf"])
    non_english_ascii_ratio_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_garbled_token_ratio: float = Field(default=0.25, ge=0.0, le=1.0)


class SkillsConfig(BaseModel):
    esco_version: str = "v1.2.1"
    fuzzy_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    min_token_length: int = Field(default=3, ge=1)
    short_token_guard_tokens: list[str] = Field(default_factory=lambda: ["c", "r", "go"])
    stuffing_ratio_threshold: float = Field(default=3.0, ge=1.0)
    max_uncorroborated_skills: int = Field(default=15, ge=1)


class SectionWeights(BaseModel):
    experience: float = Field(default=1.0, ge=0.0)
    projects: float = Field(default=0.9, ge=0.0)
    skills: float = Field(default=0.5, ge=0.0)
    certifications: float = Field(default=0.5, ge=0.0)
    education: float = Field(default=0.4, ge=0.0)
    summary: float = Field(default=0.3, ge=0.0)
    other: float = Field(default=0.2, ge=0.0)


class MatchingConfig(BaseModel):
    alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    beta: float = Field(default=0.6, ge=0.0, le=1.0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = Field(default=200, ge=50, le=512)
    chunk_overlap_tokens: int = Field(default=30, ge=0, le=100)
    fusion_method: Literal["weighted", "rrf"] = "weighted"
    rrf_k: int = Field(default=60, ge=1)
    normalizer: Literal["percentile", "minmax"] = "percentile"
    cross_encoder_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    section_weights: SectionWeights = Field(default_factory=SectionWeights)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> MatchingConfig:
        if self.fusion_method == "weighted":
            total = round(self.alpha + self.beta, 6)
            if abs(total - 1.0) > 1e-4:
                raise ValueError(
                    f"Hybrid weights must sum to 1.0; received alpha={self.alpha}, beta={self.beta} (sum={total})"
                )
        return self


class ConstraintsConfig(BaseModel):
    mode: Literal["soft", "hard"] = "soft"
    min_years_penalty_per_year: float = Field(default=0.85, ge=0.1, le=1.0)
    missing_degree_penalty: float = Field(default=0.70, ge=0.1, le=1.0)
    missing_must_have_skill_penalty: float = Field(default=0.80, ge=0.1, le=1.0)


class ThresholdsConfig(BaseModel):
    semantic_strong: float = Field(default=0.70, ge=0.0, le=1.0)
    semantic_moderate: float = Field(default=0.45, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> ThresholdsConfig:
        if self.semantic_moderate >= self.semantic_strong:
            raise ValueError(
                f"semantic_moderate ({self.semantic_moderate}) must be strictly less than semantic_strong ({self.semantic_strong})"
            )
        return self


class DatabaseConfig(BaseModel):
    url: str = "postgresql://resumefit:resumefit@localhost:5432/resumefit"
    pool_size: int = 5
    max_overflow: int = 10
    hnsw_m: int = Field(default=16, ge=4, le=64)
    hnsw_ef_construction: int = Field(default=64, ge=16, le=512)
    hnsw_ef_search: int = Field(default=40, ge=10, le=256)


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    model_config = SettingsConfigDict(
        env_prefix="RESUMEFIT_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


def load_config(config_path: Path | str | None = None) -> Settings:
    """Load configuration from a YAML file, overlaid with environment variables."""
    if config_path is None:
        # Default path relative to workspace or repository root
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "default.yaml"

    config_path = Path(config_path)
    yaml_data: dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    return Settings(**yaml_data)


# Global singleton instance loaded once at module import
_current_config: Settings | None = None


def get_config() -> Settings:
    """Return the global configuration instance."""
    global _current_config
    if _current_config is None:
        _current_config = load_config()
    return _current_config
