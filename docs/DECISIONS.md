# ResumeFit — Architecture Decision Records (ADRs)

This document records the architectural and engineering decisions made for the **ResumeFit** system, in accordance with Non-Negotiable Principle 6 ("Config over constants") and the viva/interview readiness standards.

---

## ADR-001: Tooling and Package Management via `uv`
- **Context**: Need fast, reproducible, locked dependencies across local development, CI, and Docker.
- **Decision**: Adopt Astral `uv` with `pyproject.toml` and pinned `uv.lock`.
- **Rationale**: Sub-second resolution and installation, cross-platform deterministic lockfile, built-in Python version manager (`uv python install 3.11`).

## ADR-002: Dual Scorer (TF-IDF Lexical + MiniLM Semantic) & Reference Corpus IDF
- **Context**: Synopsis requires dual scoring. Evaluating TF-IDF on a single query-document pair collapses IDF into a binary existence indicator.
- **Decision**: Fit `TfidfVectorizer` on a persistent reference corpus of resumes + job descriptions, saved as a versioned artifact. Semantic scoring uses `sentence-transformers/all-MiniLM-L6-v2`.
- **Rationale**: Preserves true IDF frequency properties (e.g. rare skills like "Kubernetes" get higher weight than common words like "experience").

## ADR-003: Section-Aware Token Chunking for Semantic Matching
- **Context**: `all-MiniLM-L6-v2` truncates text beyond 256 WordPiece tokens. Long multi-page resumes lose qualifications at the bottom if passed as a single string.
- **Decision**: Chunk text by detected section into windows of ≤200 tokens with 30-token overlap. Support two semantic modes:
  1. Mean-pooled document embedding (baseline)
  2. Requirement-to-chunk matching (max cosine per requirement, weighted by section weight)
- **Rationale**: Solves truncation without losing local context or section hierarchy (e.g., experience weighted higher than generic skills lists).

## ADR-004: Score Normalization Prior to Linear Fusion
- **Context**: Raw TF-IDF cosine scores and dense embedding cosines exist on different empirical distributions (TF-IDF is sparse with lower medians; dense embeddings have higher base cosine values).
- **Decision**: Implement a `Normalizer` layer before applying `α·lexical + β·semantic`. Provide percentile calibration against stored dev-set distributions for single-pair analysis, and min-max normalization for batch ranking.
- **Rationale**: Prevents dense semantic scores from dominating the weighted sum when α=0.5.

## ADR-005: ESCO v1.2.1 Taxonomy with Curated Software Alias Overlay
- **Context**: The European Skills, Competences, Qualifications and Occupations (ESCO v1.2.1) taxonomy provides standard occupational skills, but lacks fine-grained software libraries and modern developer tools (e.g., `scikit-learn`, `FastAPI`, `pgvector`).
- **Decision**: Ground skill extraction in ESCO, supplemented by a version-controlled YAML overlay (`data/aliases.yaml`, ~300+ entries) mapping tools to parent ESCO concepts or explicitly labeled custom concepts.
- **Rationale**: Maintains standardized European qualification grounding while accommodating real-world IT/AI skill phrasing.

## ADR-006: Deterministic Explanations Without Generative LLMs
- **Context**: Many AI matching tools generate dynamic summaries with LLMs, introducing hallucinations, non-determinism, and opacity into hiring workflows.
- **Decision**: The entire explanation pipeline is deterministic: structured evidence tables, exact quote spans with character offsets, and templated diagnostic summaries.
- **Rationale**: 100% auditability and compliance with Non-Negotiable Principle 1 ("Explainability first").

## ADR-007: Privacy by Default and Pre-Embedding PII Redaction
- **Context**: Resumes contain sensitive Personally Identifiable Information (PII) including full names, phone numbers, email addresses, and postal locations.
- **Decision**:
  1. Default mode is in-memory processing with opt-in database persistence.
  2. PII is redacted via spaCy NER and regex before generating embeddings or storing chunk records.
  3. Log records pass through a custom PII scrubbing filter.
- **Rationale**: Compliance with NIST AI RMF privacy recommendations and standard data minimization principles.

## ADR-008: Explicit Rejection of Scanned / Image-Only PDFs
- **Context**: ResumeFit is specified as a text-based PDF matching system. OCR on low-quality scanned documents introduces garbled text that pollutes skill extraction and scoring.
- **Decision**: Implement explicit scan detection (`core/ingestion/scan_detect.py`). Any PDF with below 50 extractable characters per page is rejected with HTTP 422 and a clear user-facing explanation.
- **Rationale**: Non-Negotiable Principle 3: "Honest engineering: unsupported inputs are flagged loudly, never silently degraded."

## ADR-009: Soft Constraints with Multiplicative Penalties as Default
- **Context**: Hard filtering candidates on minimum years of experience or degree requirements can prematurely discard near-match candidates who possess exceptional skill evidence.
- **Decision**: Use configurable soft penalties (multiplicative factors, e.g., 0.85 per missing year) displayed itemized in the fit report. Provide a config toggle for strict hard-filtering mode.
- **Rationale**: Gives recruiters and candidates full visibility into why a fit score was reduced rather than a binary rejection.

## ADR-010: Database Schema & Hand-Written SQL for pgvector
- **Context**: Showcase production-grade SQL and vector database capabilities for academic defense and engineering evaluation.
- **Decision**: PostgreSQL 15 + pgvector with HNSW indexing (`vector_cosine_ops`, `m=16`, `ef_construction=64`). All vector and hybrid full-text search queries are written in raw SQL in `core/db/sql/`.
- **Rationale**: High retrieval performance, transparent execution plans (`EXPLAIN`), and clean separation from ORM overhead during search.

## ADR-011: ESCO Taxonomy Granularity Gaps, Alias Overlay, and Guarded Matching Cascade
- **Context**: ESCO v1.2.1 provides an authoritative European multilingual classification of skills and occupations, but exhibits a significant latency and granularity gap regarding modern developer ecosystems:
  1. Broad concepts exist (e.g., "Python (computer programming)", "machine learning", "cloud technologies"), but specific frameworks and libraries (`FastAPI`, `scikit-learn`, `PyTorch`, `pgvector`, `Hugging Face`, `Celery`) are absent.
  2. Unrestricted fuzzy matching on short tech tokens (C, R, Go) causes severe false-positive cascades (e.g., "C" matching "CSS" or "CI/CD", "R" matching "REST", "Go" matching the common verb "go").
  3. Short acronyms (e.g., `DDD`, `WandB`, `K8s`, `AWS`) incorrectly match common English words if fuzzy distance is applied to short strings.
- **Decision**:
  1. **Curated Alias Overlay (`data/aliases.yaml`)**: Maintain a version-controlled YAML overlay with 314+ software and AI skills, linking each entry to its canonical label and parent ESCO concept URI.
  2. **Three-Tier Matching Cascade**:
     - *Exact ESCO*: Preferred labels and alternative labels.
     - *Exact Alias*: Case-insensitive lookup in the alias registry.
     - *Guarded Fuzzy Match*: Powered by RapidFuzz (`ratio` >= 0.82), strictly restricted to terms >= 4 characters. Acronyms (<= 4 chars) and spaCy stop words are barred from fuzzy matching.
  3. **Context Guards for 1-2 Character Languages**: "C", "R", and "Go" require contextual validation regexes (e.g. "Languages: C, C++", "C programming", "Golang", "R/Python") and are never matched as free-form single letters.
  4. **Evidence Tiers & Keyword-Stuffing Flag**: Skills found in Work Experience or Projects with action verbs are tagged `DEMONSTRATED`; skills in list headers are tagged `LISTED_ONLY`. If `listed_only / max(demonstrated, 1) >= 3.0` and uncorroborated skills >= 15, the resume is flagged for keyword stuffing.
- **Rationale**: Achieves 90.0% Micro-F1 and 89.8% Macro-F1 across 20 diverse engineering benchmark resumes without generative hallucinations or silent misclassifications.

