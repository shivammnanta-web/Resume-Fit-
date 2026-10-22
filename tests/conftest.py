"""Pytest fixtures for ResumeFit test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from resumefit.config import Settings, load_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def default_config(project_root: Path) -> Settings:
    """Load default configuration from config/default.yaml."""
    return load_config(project_root / "config" / "default.yaml")


@pytest.fixture
def sample_resume_text() -> str:
    """Return a clean sample resume string for testing ingestion & matching."""
    return """
John Doe
Software Engineer | Dehradun, India
Email: john.doe@example.com | Phone: +91 9876543210

PROFESSIONAL SUMMARY
Results-driven software engineer with 3 years of experience developing web applications,
REST APIs, and machine learning pipelines using Python and PostgreSQL.

WORK EXPERIENCE
Software Engineer | Acme Tech Solutions
Jan 2022 - Present | Dehradun, India
- Designed and deployed scalable REST APIs using Python, Flask, and PostgreSQL.
- Implemented automated CI/CD pipelines reducing deployment latency by 35%.
- Built data processing pipelines with Pandas and scikit-learn for anomaly detection.

PROJECTS
Explainable Candidate Search Engine
- Built a semantic search system using Sentence-Transformers and pgvector.
- Implemented TF-IDF lexical retrieval baseline for hybrid ranking ablation.

EDUCATION
Bachelor of Computer Applications (AI & Data Science)
Graphic Era University | 2019 - 2022 | CGPA: 8.8/10

TECHNICAL SKILLS
Languages: Python, SQL, C++, JavaScript
Frameworks: Flask, SQLAlchemy, scikit-learn, Sentence-Transformers
Databases: PostgreSQL, pgvector
Tools: Docker, Git, Linux
"""


@pytest.fixture
def sample_jd_text() -> str:
    """Return a clean sample job description string for testing."""
    return """
Job Title: Python Machine Learning Engineer
Location: Dehradun, India (Hybrid)
Experience: Minimum 2 years

Requirements:
- Bachelor's degree in Computer Science, BCA, or related field.
- Must have at least 2 years of professional experience with Python.
- Proven experience building RESTful APIs using Flask or FastAPI.
- Demonstrated experience with scikit-learn, machine learning algorithms, and PostgreSQL.
- Solid understanding of Git and containerization with Docker.

Preferred Qualifications:
- Experience with Sentence-Transformers and vector databases like pgvector.
- Familiarity with CI/CD pipelines and Linux environments.
"""
