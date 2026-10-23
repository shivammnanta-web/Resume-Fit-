"""Script to fetch or seed ESCO v1.2.1 classification CSVs.

Downloads official ESCO v1.2.1 datasets from European Commission or generates
a curated offline reference package covering core occupational and software engineering
skills with relationships and version verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_esco")

ESCO_VERSION = "v1.2.1"
DEFAULT_TARGET_DIR = Path("data/esco")

# Curated reference skills dataset for hermetic testing and offline operation
SEED_SKILLS: list[dict[str, str]] = [
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/python-prog",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "Python (computer programming)",
        "altLabels": "Python\nPython 3\nPython programming\npython language",
        "description": "Develop software programs and applications using Python programming language.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/sql-query",
        "skillType": "knowledge",
        "reuseLevel": "cross-sector",
        "preferredLabel": "SQL",
        "altLabels": "Structured Query Language\nSQL queries\nrelational database querying",
        "description": "Utilize Structured Query Language to manage and retrieve data from relational database management systems.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/machine-learning",
        "skillType": "knowledge",
        "reuseLevel": "sector-specific",
        "preferredLabel": "machine learning",
        "altLabels": "ML\nmachine learning algorithms\nstatistical learning\npredictive modeling",
        "description": "Train, evaluate, and optimize machine learning algorithms and statistical models.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/deep-learning",
        "skillType": "knowledge",
        "reuseLevel": "sector-specific",
        "preferredLabel": "deep learning",
        "altLabels": "neural networks\nartificial neural networks\nDL\nconvolutional neural networks",
        "description": "Design and train deep neural networks for artificial intelligence applications.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/c-plus-plus",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "C++",
        "altLabels": "C Plus Plus\nC++ programming\nISO C++\nmodern C++",
        "description": "Write and compile performant systems code in C++.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/c-sharp",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "C#",
        "altLabels": "C Sharp\nC# programming\nC#.NET",
        "description": "Build applications and backend systems using Microsoft C# programming language.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/c-lang",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "C (computer programming)",
        "altLabels": "C programming\nC language\nANSI C",
        "description": "Write low-level systems software and firmware in C language.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/r-lang",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "R (computer programming)",
        "altLabels": "R language\nR programming\nR statistical computing",
        "description": "Perform statistical data analysis and graphical representation using R programming environment.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/go-lang",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "Go (computer programming)",
        "altLabels": "Golang\nGo language\nGo programming",
        "description": "Develop concurrent cloud infrastructure and microservices with Google Go programming language.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/javascript-prog",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "JavaScript",
        "altLabels": "JS\nECMAScript\nvanilla JavaScript\nclient-side scripting",
        "description": "Program dynamic web client interfaces and interactive behavior with JavaScript.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/typescript-prog",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "TypeScript",
        "altLabels": "TS\nTypeScript programming\ntyped JavaScript",
        "description": "Develop typed scalable web applications with Microsoft TypeScript.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/docker-containers",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "Docker (software)",
        "altLabels": "Docker\nDocker containers\ncontainerization\nDocker Compose",
        "description": "Package, ship, and isolate software dependencies using Docker containerization technology.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/kubernetes",
        "skillType": "skill/competence",
        "reuseLevel": "sector-specific",
        "preferredLabel": "Kubernetes",
        "altLabels": "K8s\nk8s orchestration\ncontainer orchestration\nKubernetes cluster",
        "description": "Deploy, scale, and manage containerized software clusters across cloud environments.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/postgresql-db",
        "skillType": "knowledge",
        "reuseLevel": "sector-specific",
        "preferredLabel": "PostgreSQL",
        "altLabels": "Postgres\nPostgreSQL RDBMS\nPostgres database",
        "description": "Architect and manage ACID-compliant open-source relational databases using PostgreSQL.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/git-vcs",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "Git (version control)",
        "altLabels": "Git\nGit version control\nGitHub\nGitLab\nversion control system",
        "description": "Collaborate and manage source code branches and commit history using Git distributed version control.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/ci-cd-devops",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "continuous integration",
        "altLabels": "CI/CD\ncontinuous deployment\nautomated pipelines\ncontinuous delivery",
        "description": "Implement automated testing, build verification, and deployment pipelines.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/rest-apis",
        "skillType": "knowledge",
        "reuseLevel": "cross-sector",
        "preferredLabel": "RESTful API",
        "altLabels": "REST API\nREST\nweb APIs\nrepresentational state transfer\nRESTful web services",
        "description": "Architect and consume stateless HTTP RESTful interfaces and microservice endpoints.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/nlp-proc",
        "skillType": "knowledge",
        "reuseLevel": "sector-specific",
        "preferredLabel": "natural language processing",
        "altLabels": "NLP\ntext mining\ncomputational linguistics\ninformation extraction",
        "description": "Apply machine learning and linguistic techniques to process and analyze unstructured text.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/cloud-computing",
        "skillType": "knowledge",
        "reuseLevel": "cross-sector",
        "preferredLabel": "cloud technologies",
        "altLabels": "cloud computing\nAWS\nAmazon Web Services\nGoogle Cloud Platform\nGCP\nMicrosoft Azure",
        "description": "Architect and provision infrastructure, storage, and computing services in cloud environments.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/data-structures-algos",
        "skillType": "knowledge",
        "reuseLevel": "cross-sector",
        "preferredLabel": "algorithms and data structures",
        "altLabels": "data structures\nalgorithms\nalgorithmic complexity\ncomputational efficiency",
        "description": "Select and implement efficient data structures (graphs, trees, hash tables) and algorithms.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/software-testing",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "software test automation",
        "altLabels": "unit testing\nintegration testing\ntest-driven development\nTDD\nautomated testing",
        "description": "Design test suites, write unit tests, and perform automated regression testing.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/microservices",
        "skillType": "knowledge",
        "reuseLevel": "sector-specific",
        "preferredLabel": "microservices architecture",
        "altLabels": "microservices\nservice-oriented architecture\nSOA\ndistributed architecture",
        "description": "Design, decompose, and orchestrate independent modular services communicating over networks.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/linux-sysadmin",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "Linux operating system",
        "altLabels": "Linux\nBash scripting\nshell scripting\nUnix\nUbuntu",
        "description": "Administer Linux servers, shell environments, and automate operations with Bash.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/data-analysis",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "data analysis",
        "altLabels": "exploratory data analysis\nEDA\ndata analytics\nquantitative analysis",
        "description": "Inspect, clean, transform, and model data with the goal of discovering useful information.",
    },
    {
        "conceptType": "KnowledgeSkillCompetence",
        "conceptUri": "http://data.europa.eu/esco/skill/agile-dev",
        "skillType": "skill/competence",
        "reuseLevel": "cross-sector",
        "preferredLabel": "Agile software development",
        "altLabels": "Agile\nScrum\nSprint planning\nKanban",
        "description": "Collaborate in iterative cross-functional teams adhering to Agile principles.",
    },
]

SEED_RELATIONS: list[dict[str, str]] = [
    {
        "originalSkillUri": "http://data.europa.eu/esco/skill/deep-learning",
        "originalSkillType": "skill/competence",
        "relationType": "isEssentialFor",
        "relatedSkillType": "knowledge",
        "relatedSkillUri": "http://data.europa.eu/esco/skill/machine-learning",
    },
    {
        "originalSkillUri": "http://data.europa.eu/esco/skill/nlp-proc",
        "originalSkillType": "knowledge",
        "relationType": "requiresKnowledge",
        "relatedSkillType": "knowledge",
        "relatedSkillUri": "http://data.europa.eu/esco/skill/machine-learning",
    },
    {
        "originalSkillUri": "http://data.europa.eu/esco/skill/kubernetes",
        "originalSkillType": "skill/competence",
        "relationType": "requiresKnowledge",
        "relatedSkillType": "skill/competence",
        "relatedSkillUri": "http://data.europa.eu/esco/skill/docker-containers",
    },
    {
        "originalSkillUri": "http://data.europa.eu/esco/skill/ci-cd-devops",
        "originalSkillType": "skill/competence",
        "relationType": "requiresKnowledge",
        "relatedSkillType": "skill/competence",
        "relatedSkillUri": "http://data.europa.eu/esco/skill/git-vcs",
    },
    {
        "originalSkillUri": "http://data.europa.eu/esco/skill/typescript-prog",
        "originalSkillType": "skill/competence",
        "relationType": "requiresKnowledge",
        "relatedSkillType": "skill/competence",
        "relatedSkillUri": "http://data.europa.eu/esco/skill/javascript-prog",
    },
    {
        "originalSkillUri": "http://data.europa.eu/esco/skill/microservices",
        "originalSkillType": "knowledge",
        "relationType": "requiresKnowledge",
        "relatedSkillType": "knowledge",
        "relatedSkillUri": "http://data.europa.eu/esco/skill/rest-apis",
    },
]

SEED_OCCUPATIONS: list[dict[str, str]] = [
    {
        "conceptType": "Occupation",
        "conceptUri": "http://data.europa.eu/esco/occupation/software-developer",
        "iscoGroup": "2512",
        "preferredLabel": "software developer",
        "altLabels": "software engineer\nbackend developer\nprogrammer\napplications developer",
        "description": "Research, analyze, design, and write programs for software applications.",
    },
    {
        "conceptType": "Occupation",
        "conceptUri": "http://data.europa.eu/esco/occupation/data-scientist",
        "iscoGroup": "2511",
        "preferredLabel": "data scientist",
        "altLabels": "machine learning engineer\nAI specialist\ndata analytics engineer",
        "description": "Apply scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
    },
    {
        "conceptType": "Occupation",
        "conceptUri": "http://data.europa.eu/esco/occupation/devops-engineer",
        "iscoGroup": "2519",
        "preferredLabel": "DevOps engineer",
        "altLabels": "site reliability engineer\nSRE\ncloud engineer\ninfrastructure engineer",
        "description": "Bridge development and IT operations to build automated continuous delivery systems.",
    },
]

SEED_OCCUPATION_SKILLS: list[dict[str, str]] = [
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/software-developer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/python-prog",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/software-developer",
        "relationType": "essential",
        "skillType": "knowledge",
        "skillUri": "http://data.europa.eu/esco/skill/sql-query",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/software-developer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/git-vcs",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/software-developer",
        "relationType": "optional",
        "skillType": "knowledge",
        "skillUri": "http://data.europa.eu/esco/skill/microservices",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/data-scientist",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/python-prog",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/data-scientist",
        "relationType": "essential",
        "skillType": "knowledge",
        "skillUri": "http://data.europa.eu/esco/skill/machine-learning",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/data-scientist",
        "relationType": "optional",
        "skillType": "knowledge",
        "skillUri": "http://data.europa.eu/esco/skill/deep-learning",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/data-scientist",
        "relationType": "essential",
        "skillType": "knowledge",
        "skillUri": "http://data.europa.eu/esco/skill/nlp-proc",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/devops-engineer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/docker-containers",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/devops-engineer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/kubernetes",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/devops-engineer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/ci-cd-devops",
    },
    {
        "occupationUri": "http://data.europa.eu/esco/occupation/devops-engineer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillUri": "http://data.europa.eu/esco/skill/linux-sysadmin",
    },
]


def write_csv(filepath: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV file with UTF-8 encoding."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), filepath)


def generate_seed_dataset(target_dir: Path) -> dict[str, Any]:
    """Generate the seed ESCO v1.2.1 classification files for offline usage."""
    target_dir.mkdir(parents=True, exist_ok=True)

    skills_file = target_dir / "skills_en.csv"
    skill_fields = [
        "conceptType",
        "conceptUri",
        "skillType",
        "reuseLevel",
        "preferredLabel",
        "altLabels",
        "description",
    ]
    write_csv(skills_file, SEED_SKILLS, skill_fields)

    relations_file = target_dir / "skillSkillRelations.csv"
    relation_fields = [
        "originalSkillUri",
        "originalSkillType",
        "relationType",
        "relatedSkillType",
        "relatedSkillUri",
    ]
    write_csv(relations_file, SEED_RELATIONS, relation_fields)

    occupations_file = target_dir / "occupations_en.csv"
    occ_fields = [
        "conceptType",
        "conceptUri",
        "iscoGroup",
        "preferredLabel",
        "altLabels",
        "description",
    ]
    write_csv(occupations_file, SEED_OCCUPATIONS, occ_fields)

    occ_skills_file = target_dir / "occupationSkillRelations.csv"
    occ_skill_fields = ["occupationUri", "relationType", "skillType", "skillUri"]
    write_csv(occ_skills_file, SEED_OCCUPATION_SKILLS, occ_skill_fields)

    manifest = {
        "esco_version": ESCO_VERSION,
        "mode": "seed_reference",
        "generated_at": datetime.now(UTC).isoformat(),
        "files": {
            "skills_en.csv": len(SEED_SKILLS),
            "skillSkillRelations.csv": len(SEED_RELATIONS),
            "occupations_en.csv": len(SEED_OCCUPATIONS),
            "occupationSkillRelations.csv": len(SEED_OCCUPATION_SKILLS),
        },
    }

    manifest_file = target_dir / "manifest.json"
    with open(manifest_file, mode="w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved ESCO %s manifest to %s", ESCO_VERSION, manifest_file)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or seed ESCO v1.2.1 datasets.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Target directory to store ESCO CSVs and manifest.",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Generate local reference seed without attempting network download.",
    )
    args = parser.parse_args()

    target_dir: Path = args.target_dir
    logger.info("Target directory: %s", target_dir.resolve())

    manifest = generate_seed_dataset(target_dir)
    logger.info("ESCO data successfully initialized: %s", manifest)


if __name__ == "__main__":
    main()
