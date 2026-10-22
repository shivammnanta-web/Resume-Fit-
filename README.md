# ResumeFit: Explainable Hybrid Resume–Job Matching System

> **Degree Project & Portfolio System**  
> Built by **Shivam**, BCA (AI & Data Science), Graphic Era (Deemed to be University).  
> Based on the synopsis: *"ResumeFit: An Explainable Hybrid Resume–Job Matching System using Lexical and Semantic Retrieval"*.

---

## 1. Overview
ResumeFit is a transparent, explainable resume-job matching engine combining lexical retrieval (TF-IDF with custom technical tokenization) and semantic retrieval (Sentence-Transformers with section-weighted requirement-to-chunk matching), grounded in the European Skills, Competences, Qualifications and Occupations (ESCO v1.2.1) taxonomy.

### Key Principles
- **Explainability First**: Every score component and skill requirement status is grounded in traceable spans with character offsets.
- **Decision Support, Not Automated Hiring**: Provides fit analysis and structured gap analysis. Never claims automated hiring authority; never uses the term "probability".
- **Deterministic Explanations**: No generative LLM in the scoring or explanation path.
- **Privacy by Default**: In-memory processing by default, automated PII redaction before embedding, and zero PII logging.
- **Honest Engineering**: Scanned and non-text PDFs are detected and rejected with actionable diagnostic messages rather than silently degraded.

---

## 2. Quickstart (Development)

Prerequisites: Python 3.11+, `uv`, Docker (for PostgreSQL + pgvector in Phase 4).

```bash
# Setup virtual environment and dependencies
uv sync

# Run tests
uv run pytest

# Run linting and type checking
uv run ruff check .
uv run mypy core resumefit
```

---

## 3. Architecture & Documentation
- [Architecture Details](docs/ARCHITECTURE.md)
- [Architecture Decision Records (ADRs)](docs/DECISIONS.md)
- [Ethics, Privacy & NIST AI RMF Mapping](docs/ETHICS_AND_PRIVACY.md)
- [Evaluation Protocol & Results](docs/EVALUATION.md)
- [Viva & Interview Q&A Guide](docs/INTERVIEW_QA.md)
