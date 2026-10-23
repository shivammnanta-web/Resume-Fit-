"""Unit tests for Phase 2 ESCO loading, alias overlay, skill extraction, and stuffing detection."""

from __future__ import annotations

from core.ingestion.types import Section, SectionType
from core.skills.aliases import load_alias_index
from core.skills.esco_loader import load_esco_taxonomy
from core.skills.evidence import EvidenceLevel, classify_evidence_level
from core.skills.extractor import SkillExtractor


def test_esco_loader() -> None:
    """Verify ESCO v1.2.1 taxonomy loads skills, relations, and occupations."""
    taxonomy = load_esco_taxonomy()
    assert taxonomy.version == "v1.2.1"
    assert len(taxonomy.skills_by_uri) >= 20
    assert len(taxonomy.occupations_by_uri) >= 3

    # Check preferred and altLabel lookups
    py = taxonomy.find_skill_by_label("Python")
    assert py is not None
    assert "Python" in py.preferred_label

    sql = taxonomy.find_skill_by_label("Structured Query Language")
    assert sql is not None
    assert sql.preferred_label == "SQL"

    # Check relation graph
    dl = taxonomy.find_skill_by_label("deep learning")
    assert dl is not None
    related = taxonomy.get_related_skills(dl.concept_uri)
    assert any("machine learning" in r.preferred_label.lower() for r in related)


def test_alias_overlay_300_plus_entries() -> None:
    """Verify curated alias overlay contains 300+ entries with parent ESCO mappings."""
    index = load_alias_index()
    assert len(index.aliases) >= 300

    # Modern software libraries
    skl = index.resolve("scikit-learn")
    assert skl is not None
    assert skl.canonical_label == "scikit-learn"
    assert skl.parent_esco_uri is not None
    assert "machine-learning" in skl.parent_esco_uri

    pgv = index.resolve("pgvector")
    assert pgv is not None
    assert pgv.canonical_label == "pgvector"
    assert pgv.parent_esco_uri is not None
    assert "postgresql" in pgv.parent_esco_uri

    fastapi = index.resolve("fastapi")
    assert fastapi is not None
    assert fastapi.canonical_label == "FastAPI"

    k8s = index.resolve("k8s")
    assert k8s is not None
    assert k8s.canonical_label == "Kubernetes"


def test_short_token_guards() -> None:
    """Ensure ambiguous short tokens (C, R, Go) are protected by context guards."""
    extractor = SkillExtractor()

    # 'C' guards
    assert (
        extractor.is_valid_short_token("c", "Front-end developer working with CSS and HTML")
        is False
    )
    assert extractor.is_valid_short_token("c", "Automated deployment with CI/CD") is False
    assert extractor.is_valid_short_token("c", "Languages: C, C++, Python") is True
    assert extractor.is_valid_short_token("c", "Low-level C programming on Linux") is True

    # 'R' guards
    assert extractor.is_valid_short_token("r", "Designed RESTful API endpoints") is False
    assert extractor.is_valid_short_token("r", "Cached queries in Redis") is False
    assert extractor.is_valid_short_token("r", "Statistical data analysis in R/Python") is True
    assert extractor.is_valid_short_token("r", "Proficient in R programming") is True

    # 'Go' guards
    assert extractor.is_valid_short_token("go", "We will go to production next week") is False
    assert extractor.is_valid_short_token("go", "Our go-to-market strategy") is False
    assert (
        extractor.is_valid_short_token("go", "Microservices developed in Go, Python, and Rust")
        is True
    )
    assert extractor.is_valid_short_token("go", "High concurrency in Golang") is True


def test_evidence_classification_tiers() -> None:
    """Verify classification into Demonstrated, Listed-only, and Tentative."""
    # Demonstrated: In experience with action verbs
    span_dem = "Architected and deployed distributed microservices using Flask and PostgreSQL."
    lvl, verbs = classify_evidence_level(span_dem, SectionType.EXPERIENCE)
    assert lvl == EvidenceLevel.DEMONSTRATED
    assert "architected" in verbs or "deployed" in verbs

    # Listed-only: In skills section
    span_list = "Flask, PostgreSQL, Redis, Docker, Git"
    lvl, verbs = classify_evidence_level(span_list, SectionType.SKILLS)
    assert lvl == EvidenceLevel.LISTED_ONLY

    # Tentative: Hedging keywords
    span_tent = "Currently learning Kubernetes and familiar with basic Docker concepts."
    lvl, _ = classify_evidence_level(span_tent, SectionType.EXPERIENCE)
    assert lvl == EvidenceLevel.TENTATIVE


def test_keyword_stuffing_detection() -> None:
    """Ensure honest resumes are not flagged, while keyword-stuffed resumes trigger warnings."""
    extractor = SkillExtractor()

    # Honest resume: 5 skills in experience, 6 skills in skills list -> ratio ~ 1.2
    honest_sections = [
        Section(
            section_type=SectionType.EXPERIENCE,
            heading="EXPERIENCE",
            text="Developed scalable backend APIs using Python, Flask, and PostgreSQL.\nDeployed containers using Docker on AWS.",
            start_char=0,
            end_char=200,
        ),
        Section(
            section_type=SectionType.SKILLS,
            heading="SKILLS",
            text="Python, Flask, PostgreSQL, Docker, AWS, Git",
            start_char=200,
            end_char=300,
        ),
    ]
    honest_res = extractor.extract_from_sections(honest_sections)
    assert honest_res.stuffing_report.is_stuffed is False

    # Stuffed resume: only 1 demonstrated skill in experience, but 20 uncorroborated skills listed
    stuffed_sections = [
        Section(
            section_type=SectionType.EXPERIENCE,
            heading="EXPERIENCE",
            text="Built web page using HTML.",
            start_char=0,
            end_char=50,
        ),
        Section(
            section_type=SectionType.SKILLS,
            heading="SKILLS",
            text=(
                "Python, Java, C++, Rust, Go, Kotlin, Swift, Scala, PHP, Ruby, "
                "Kubernetes, Docker, Terraform, Ansible, Jenkins, AWS, Azure, GCP, "
                "PostgreSQL, MongoDB, Cassandra, Redis, Elasticsearch, Kafka, Spark"
            ),
            start_char=50,
            end_char=350,
        ),
    ]
    stuffed_res = extractor.extract_from_sections(stuffed_sections)
    assert stuffed_res.stuffing_report.is_stuffed is True
    assert stuffed_res.stuffing_report.ratio >= 3.0
    assert stuffed_res.stuffing_report.warning_message is not None


def test_hierarchy_linking_and_aliases() -> None:
    """Verify that modern libraries link to their parent ESCO family and related skills."""
    extractor = SkillExtractor()
    sections = [
        Section(
            section_type=SectionType.EXPERIENCE,
            heading="PROJECTS",
            text="Trained predictive models with scikit-learn and implemented vector search using pgvector.",
            start_char=0,
            end_char=200,
        )
    ]
    res = extractor.extract_from_sections(sections)
    skl = res.get_skill("scikit-learn")
    assert skl is not None
    assert skl.evidence_level == EvidenceLevel.DEMONSTRATED
    assert "machine learning" in str(skl.related_esco_skills).lower() or skl.esco_uri is not None
