"""Skill alias registry and resolver (ADR-005).

Loads the curated YAML overlay of modern libraries, frameworks, tools, and abbreviations,
resolving them to their canonical labels and parent ESCO v1.2.1 concept URIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_ALIASES_PATH = Path("data/aliases.yaml")


@dataclass(frozen=True)
class SkillAlias:
    """Mapping of an alias/abbreviation to a canonical skill label and optional ESCO URI."""

    alias: str
    canonical_label: str
    category: str = "library"
    parent_esco_uri: str | None = None


@dataclass
class AliasIndex:
    """In-memory index for fast case-insensitive alias lookup."""

    version: str = "1.2.1"
    overlay_version: str = "2026.1"
    aliases: dict[str, SkillAlias] = field(default_factory=dict)  # normalized_text -> SkillAlias
    canonicals: dict[str, list[str]] = field(
        default_factory=dict
    )  # canonical_label -> list of aliases

    def resolve(self, phrase: str) -> SkillAlias | None:
        """Resolve a candidate phrase or token to its SkillAlias record."""
        norm = phrase.strip().lower()
        return self.aliases.get(norm)

    def is_known_alias(self, phrase: str) -> bool:
        """Return True if phrase matches a registered alias."""
        norm = phrase.strip().lower()
        return norm in self.aliases

    def get_aliases_for_canonical(self, canonical_label: str) -> list[str]:
        """Return all registered alias variants for a canonical skill."""
        return self.canonicals.get(canonical_label.strip(), [])


def normalize_alias_key(text: str) -> str:
    """Normalize alias text for dictionary lookup."""
    return text.strip().lower()


def load_alias_index(filepath: Path | str | None = None) -> AliasIndex:
    """Load the alias overlay YAML file into an indexed AliasIndex instance."""
    path = Path(filepath) if filepath else DEFAULT_ALIASES_PATH

    if not path.exists():
        logger.warning("Alias file not found at %s; returning empty AliasIndex", path)
        return AliasIndex()

    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to parse alias YAML file %s: %s", path, exc)
        return AliasIndex()

    version = str(data.get("version", "1.2.1"))
    overlay_ver = str(data.get("overlay_version", "2026.1"))
    raw_entries = data.get("aliases", [])

    index = AliasIndex(version=version, overlay_version=overlay_ver)

    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        alias_str = entry.get("alias", "").strip()
        canonical = entry.get("canonical_label", "").strip()
        if not alias_str or not canonical:
            continue

        skill_alias = SkillAlias(
            alias=alias_str,
            canonical_label=canonical,
            category=entry.get("category", "tool").strip(),
            parent_esco_uri=entry.get("parent_esco_uri"),
        )

        norm_key = normalize_alias_key(alias_str)
        index.aliases[norm_key] = skill_alias

        if canonical not in index.canonicals:
            index.canonicals[canonical] = []
        if alias_str not in index.canonicals[canonical]:
            index.canonicals[canonical].append(alias_str)

    logger.info("Loaded %d skill aliases from %s", len(index.aliases), path)
    return index


@lru_cache(maxsize=1)
def get_cached_alias_index(filepath: str | None = None) -> AliasIndex:
    """Return a cached singleton instance of the loaded skill alias index."""
    path = Path(filepath) if filepath else DEFAULT_ALIASES_PATH
    return load_alias_index(path)
