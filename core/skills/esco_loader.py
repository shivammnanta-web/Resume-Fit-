"""ESCO v1.2.1 taxonomy loader and normalized in-memory index.

Loads and indexes ESCO skills, altLabels, descriptions, and relationship graphs.
Supports lookup by URI, preferredLabel, or altLabel with fast in-memory indexing.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.fetch_esco import generate_seed_dataset

logger = logging.getLogger(__name__)

DEFAULT_ESCO_DIR = Path("data/esco")


@dataclass
class EscoSkill:
    """Normalized representation of an ESCO skill or knowledge concept."""

    concept_uri: str
    preferred_label: str
    alt_labels: list[str] = field(default_factory=list)
    skill_type: str = "skill/competence"  # "skill/competence" or "knowledge"
    reuse_level: str = "cross-sector"
    description: str = ""
    related_skill_uris: list[str] = field(default_factory=list)
    essential_for_occupations: list[str] = field(default_factory=list)
    optional_for_occupations: list[str] = field(default_factory=list)

    @property
    def all_labels(self) -> list[str]:
        """Return combined list of preferred and alternative labels."""
        return [self.preferred_label] + self.alt_labels


@dataclass
class EscoOccupation:
    """Normalized representation of an ESCO occupation."""

    concept_uri: str
    preferred_label: str
    isco_group: str
    alt_labels: list[str] = field(default_factory=list)
    description: str = ""
    essential_skill_uris: list[str] = field(default_factory=list)
    optional_skill_uris: list[str] = field(default_factory=list)


@dataclass
class EscoTaxonomy:
    """Complete in-memory ESCO taxonomy graph and normalized lookup indices."""

    version: str
    skills_by_uri: dict[str, EscoSkill] = field(default_factory=dict)
    skills_by_label: dict[str, str] = field(default_factory=dict)  # normalized_label -> concept_uri
    occupations_by_uri: dict[str, EscoOccupation] = field(default_factory=dict)
    occupations_by_label: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def get_skill(self, uri: str) -> EscoSkill | None:
        """Retrieve skill by exact concept URI."""
        return self.skills_by_uri.get(uri)

    def find_skill_by_label(self, label: str) -> EscoSkill | None:
        """Find skill by case-insensitive preferred or alternative label."""
        norm = label.strip().lower()
        uri = self.skills_by_label.get(norm)
        if uri:
            return self.skills_by_uri.get(uri)
        return None

    def get_related_skills(self, uri: str) -> list[EscoSkill]:
        """Retrieve all related skills for a given skill URI."""
        skill = self.get_skill(uri)
        if not skill:
            return []
        return [
            self.skills_by_uri[r_uri]
            for r_uri in skill.related_skill_uris
            if r_uri in self.skills_by_uri
        ]


def normalize_label(label: str) -> str:
    """Normalize label for index matching (lowercase, stripped)."""
    return label.strip().lower()


def load_esco_taxonomy(data_dir: Path | str | None = None) -> EscoTaxonomy:
    """Load ESCO CSV datasets from data directory into an in-memory EscoTaxonomy instance.

    If data files are missing, automatically seeds the reference dataset.
    """
    esco_path = Path(data_dir) if data_dir else DEFAULT_ESCO_DIR

    skills_file = esco_path / "skills_en.csv"
    if not skills_file.exists():
        logger.warning("ESCO data missing at %s; generating seed reference dataset", esco_path)
        generate_seed_dataset(esco_path)

    # 1. Load manifest
    manifest_file = esco_path / "manifest.json"
    manifest: dict[str, Any] = {}
    version = "v1.2.1"
    if manifest_file.exists():
        try:
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)
                version = manifest.get("esco_version", version)
        except Exception as exc:
            logger.debug("Failed to read manifest: %s", exc)

    taxonomy = EscoTaxonomy(version=version, manifest=manifest)

    # 2. Load skills
    if skills_file.exists():
        with open(skills_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = row.get("conceptUri", "").strip()
                pref_label = row.get("preferredLabel", "").strip()
                if not uri or not pref_label:
                    continue

                raw_alts = row.get("altLabels", "") or ""
                # altLabels in ESCO are newline or comma delimited
                alt_list: list[str] = []
                for chunk in raw_alts.replace("\r", "").split("\n"):
                    for item in chunk.split(","):
                        s = item.strip()
                        if s and s.lower() != pref_label.lower() and s not in alt_list:
                            alt_list.append(s)

                skill = EscoSkill(
                    concept_uri=uri,
                    preferred_label=pref_label,
                    alt_labels=alt_list,
                    skill_type=row.get("skillType", "skill/competence").strip(),
                    reuse_level=row.get("reuseLevel", "cross-sector").strip(),
                    description=row.get("description", "").strip(),
                )
                taxonomy.skills_by_uri[uri] = skill

                # Index preferred and alt labels
                norm_pref = normalize_label(pref_label)
                taxonomy.skills_by_label[norm_pref] = uri

                for alt in alt_list:
                    norm_alt = normalize_label(alt)
                    if norm_alt not in taxonomy.skills_by_label:
                        taxonomy.skills_by_label[norm_alt] = uri

    # 3. Load skill-skill relations
    relations_file = esco_path / "skillSkillRelations.csv"
    if relations_file.exists():
        with open(relations_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_uri = row.get("originalSkillUri", "").strip()
                rel_uri = row.get("relatedSkillUri", "").strip()
                if (
                    orig_uri in taxonomy.skills_by_uri
                    and rel_uri
                    and rel_uri not in taxonomy.skills_by_uri[orig_uri].related_skill_uris
                ):
                    taxonomy.skills_by_uri[orig_uri].related_skill_uris.append(rel_uri)

    # 4. Load occupations
    occupations_file = esco_path / "occupations_en.csv"
    if occupations_file.exists():
        with open(occupations_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = row.get("conceptUri", "").strip()
                pref_label = row.get("preferredLabel", "").strip()
                if not uri or not pref_label:
                    continue

                raw_alts = row.get("altLabels", "") or ""
                alt_list = [
                    a.strip()
                    for a in raw_alts.replace("\r", "").split("\n")
                    if a.strip() and a.strip().lower() != pref_label.lower()
                ]

                occ = EscoOccupation(
                    concept_uri=uri,
                    preferred_label=pref_label,
                    isco_group=row.get("iscoGroup", "").strip(),
                    alt_labels=alt_list,
                    description=row.get("description", "").strip(),
                )
                taxonomy.occupations_by_uri[uri] = occ
                taxonomy.occupations_by_label[normalize_label(pref_label)] = uri
                for alt in alt_list:
                    norm_alt = normalize_label(alt)
                    if norm_alt not in taxonomy.occupations_by_label:
                        taxonomy.occupations_by_label[norm_alt] = uri

    # 5. Load occupation-skill relations
    occ_skills_file = esco_path / "occupationSkillRelations.csv"
    if occ_skills_file.exists():
        with open(occ_skills_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                occ_uri = row.get("occupationUri", "").strip()
                rel_type = row.get("relationType", "essential").strip().lower()
                skill_uri = row.get("skillUri", "").strip()

                if occ_uri in taxonomy.occupations_by_uri:
                    occ = taxonomy.occupations_by_uri[occ_uri]
                    if rel_type == "essential":
                        if skill_uri not in occ.essential_skill_uris:
                            occ.essential_skill_uris.append(skill_uri)
                    else:
                        if skill_uri not in occ.optional_skill_uris:
                            occ.optional_skill_uris.append(skill_uri)

                if skill_uri in taxonomy.skills_by_uri:
                    sk = taxonomy.skills_by_uri[skill_uri]
                    if rel_type == "essential":
                        if occ_uri not in sk.essential_for_occupations:
                            sk.essential_for_occupations.append(occ_uri)
                    else:
                        if occ_uri not in sk.optional_for_occupations:
                            sk.optional_for_occupations.append(occ_uri)

    logger.info(
        "Loaded ESCO %s: %d skills, %d occupations, %d label index entries",
        taxonomy.version,
        len(taxonomy.skills_by_uri),
        len(taxonomy.occupations_by_uri),
        len(taxonomy.skills_by_label),
    )
    return taxonomy


@lru_cache(maxsize=1)
def get_cached_taxonomy(data_dir: str | None = None) -> EscoTaxonomy:
    """Return a cached singleton instance of the loaded ESCO taxonomy."""
    path = Path(data_dir) if data_dir else DEFAULT_ESCO_DIR
    return load_esco_taxonomy(path)
