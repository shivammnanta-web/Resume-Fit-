"""Build and persist TF-IDF reference corpus artifact (Task P3.3).

Satisfies Non-Negotiable Principle 3:
"TF-IDF Reference Corpus. The TF-IDF vectorizer must be fit once against a reference
corpus of ~100 diverse resumes + JDs, saved as a versioned artifact
(data/artifacts/tfidf_reference.joblib), and used across all scoring sessions.
Ad-hoc two-document fitting is strictly prohibited because the inverse document
frequency (IDF) of a term in a 2-document corpus is unstable."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("data/artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "tfidf_reference.joblib"
GOLD_DIR = Path("data/gold/skill_labels")

# Diverse synthetic JD & Resume domain templates to reach 100+ documents
DOMAIN_PROFILES = [
    {
        "role": "Senior Backend Engineer - Python",
        "keywords": "python django flask fastapi postgresql redis rabbitmq celery docker kubernetes aws microservices rest grpc pytest sql git ci/cd linux pgvector kafka",
        "description": "Architect and develop high-throughput distributed microservices in Python. Optimize PostgreSQL relational databases and Redis caching layers. Implement asynchronous background task workers using Celery and RabbitMQ. Containerize microservices with Docker and deploy to Kubernetes clusters on AWS. Write comprehensive unit and integration tests with pytest.",
    },
    {
        "role": "Frontend Engineer - React / TypeScript",
        "keywords": "typescript react next.js redux tailwindcss html5 css3 javascript vite webpack responsive design web performance jest cypress graphql rest",
        "description": "Build responsive web applications using React, TypeScript, and Next.js. Manage complex client-side state with Redux Toolkit. Create performant UI layouts with modern CSS and TailwindCSS. Conduct end-to-end testing with Cypress and unit testing with Jest.",
    },
    {
        "role": "Full Stack Developer - Node.js & React",
        "keywords": "javascript typescript node.js express react mongodb postgresql graphql rest docker aws git jest microservices html5 css3",
        "description": "Design end-to-end web architectures using Node.js, Express, React, and PostgreSQL. Develop secure RESTful and GraphQL APIs. Configure Docker containers for automated continuous integration and continuous deployment.",
    },
    {
        "role": "Machine Learning Engineer - NLP & Vision",
        "keywords": "python pytorch tensorflow scikit-learn transformers huggingface pandas numpy opencv nlp computer vision llms bert mlops mlflow docker gpu cuda",
        "description": "Develop and fine-tune transformer models, BERT embeddings, and deep neural networks in PyTorch. Build automated MLOps training pipelines using MLflow and Docker. Optimize GPU inference latency with ONNX and TensorRT.",
    },
    {
        "role": "DevOps & Cloud Platform Engineer",
        "keywords": "aws gcp azure terraform kubernetes docker linux bash ansible ci/cd prometheus grafana helm jenkins security networking cloudformation",
        "description": "Manage multi-cloud infrastructure on AWS and GCP using Terraform Infrastructure as Code. Deploy resilient Kubernetes microservice clusters using Helm charts. Build automated CI/CD deployment pipelines and configure Prometheus and Grafana observability.",
    },
    {
        "role": "Data Engineer - Big Data & Pipelines",
        "keywords": "python sql spark pyspark kafka airflow snowflake databricks dbt hadoop aws s3 postgresql etl data warehousing data lakes",
        "description": "Design and build scalable ETL/ELT data pipelines using Apache Spark, PySpark, and Airflow. Ingest streaming real-time event logs with Apache Kafka. Model enterprise data marts in Snowflake and Databricks with dbt.",
    },
    {
        "role": "Mobile App Developer - iOS & Android",
        "keywords": "swift kotlin swiftui jetpack compose flutter react native ios android mobile mobile architecture sqlite rest apis xcode android studio",
        "description": "Build cross-platform and native mobile applications using Flutter, Kotlin, and Swift. Implement reactive UI patterns with SwiftUI and Jetpack Compose. Integrate offline SQLite persistence and background push notification services.",
    },
    {
        "role": "Security Engineer - Application Security",
        "keywords": "cybersecurity appsec owasp penetration testing cryptography vulnerability assessment siem soc python bash linux tls oauth2 jwt burp suite",
        "description": "Perform application vulnerability assessments and threat modeling based on OWASP Top 10 guidelines. Conduct penetration testing using Burp Suite and automate vulnerability scans with Python. Implement secure identity management with OAuth2 and JWT.",
    },
    {
        "role": "QA Automation Engineer",
        "keywords": "qa automation selenium cypress playwright pytest testng junit java python ci/cd postman api testing performance testing jmeter",
        "description": "Architect automated UI and API test frameworks using Playwright, Cypress, and Pytest. Automate end-to-end regressions inside CI/CD pipelines. Execute load and performance testing with Apache JMeter.",
    },
    {
        "role": "Systems / Embedded Software Engineer",
        "keywords": "c c++ rust linux embedded rtos arm microcontrollers gdb memory management concurrency tcp/ip sockets posix cmake",
        "description": "Develop low-level systems software in C and C++ on Linux platforms. Implement multi-threaded network socket protocols and real-time embedded software for ARM microcontrollers. Optimize memory footprint and eliminate data race conditions.",
    },
]


def build_reference_corpus() -> list[str]:
    """Assemble a balanced, diverse reference corpus of ~100 documents."""
    corpus: list[str] = []

    # 1. Load gold benchmark resumes if available
    if GOLD_DIR.exists():
        for json_path in sorted(GOLD_DIR.glob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("text", "")
                    if text.strip():
                        corpus.append(text)
            except Exception as e:
                logger.warning("Error reading %s: %s", json_path, e)

    logger.info("Loaded %d gold resumes into reference corpus", len(corpus))

    # 2. Generate multi-variant synthetic documents across 10 domains
    # Variations: Senior JD, Mid JD, Junior Resume, Lead Resume, Tech Profile
    experience_levels = [
        ("Junior", "1-2 years experience, foundational knowledge, academic projects, entry-level contributions."),
        ("Mid-level", "3-5 years experience, feature ownership, code review, automated testing, API integration."),
        ("Senior", "5-8 years experience, architecture design, system scalability, mentoring, cross-functional leadership."),
        ("Staff / Lead", "8+ years experience, distributed system design, tech roadmap, org-wide standards, high reliability."),
    ]

    for profile in DOMAIN_PROFILES:
        for level_name, level_desc in experience_levels:
            # Synthetic Job Description
            jd_doc = (
                f"Job Title: {level_name} {profile['role']}\n"
                f"Requirements and Qualifications:\n"
                f"{level_desc}\n"
                f"Role Responsibilities: {profile['description']}\n"
                f"Core Required Tech Stack: {profile['keywords']}\n"
                f"Strong proficiency in software engineering best practices, agile methodologies, and clean code principles."
            )
            corpus.append(jd_doc)

            # Synthetic Candidate Resume
            resume_doc = (
                f"Candidate Profile: {level_name} Engineer\n"
                f"Summary: Results-oriented engineer with {level_desc.lower()} specializing in {profile['role']}.\n"
                f"Professional Experience:\n"
                f"• {profile['description']}\n"
                f"• Leveraged modern technologies including: {profile['keywords']}.\n"
                f"• Participated in agile standups, wrote unit tests, and resolved critical production incidents.\n"
                f"Education: Bachelor of Science in Computer Science.\n"
                f"Technical Skills: {profile['keywords']}"
            )
            corpus.append(resume_doc)

    logger.info("Total reference corpus size: %d documents", len(corpus))
    return corpus


def main() -> None:
    from core.matching.lexical import tech_tokenize

    corpus = build_reference_corpus()
    logger.info("Fitting TfidfVectorizer with custom tech tokenizer...")

    vectorizer = TfidfVectorizer(
        tokenizer=tech_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    vectorizer.fit(corpus)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, ARTIFACT_PATH)

    vocab_size = len(vectorizer.get_feature_names_out())
    logger.info(
        "Successfully saved TF-IDF reference vectorizer to %s (vocabulary: %d terms)",
        ARTIFACT_PATH,
        vocab_size,
    )


if __name__ == "__main__":
    main()
