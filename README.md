# ResumeFit: Explainable Hybrid Resume–Job Matching System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test Coverage: 83%](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)]()
[![ESCO Taxonomy: v1.2.1](https://img.shields.io/badge/ESCO-v1.2.1-teal.svg)](https://esco.ec.europa.eu/)

> **Academic Defense & Production-Grade Portfolio System**  
> **Author:** Shivam  
> **Degree:** BCA (Artificial Intelligence & Data Science), Graphic Era (Deemed to be University)  
> **Core Premise:** *"ResumeFit: An Explainable Hybrid Resume–Job Matching System using Lexical and Semantic Retrieval"*

---

## 1. Executive Summary & Core Philosophy

Conventional ATS (Applicant Tracking Systems) and modern LLM wrappers suffer from two fatal failure modes in high-stakes candidate evaluation:
1. **Opaque Hallucinations:** Generative models make untraceable, non-deterministic claims about candidate competence.
2. **Brittle Keyword Matching:** Naive lexical parsers miss semantic synonyms and reward superficial keyword stuffing.

**ResumeFit** solves this with an **explainable-by-design, auditable hybrid retrieval architecture**:
* **Deterministic Scoring (Fit Index 0–100):** No generative LLMs in the scoring or explanation path. Every score decomposes linearly into verified lexical terms, semantic requirement alignments, and itemized constraint deductions.
* **Dual Retrieval Engine:** Blends frozen reference TF-IDF cosine similarity ($\alpha=0.4$) with Sentence-Transformers (`all-MiniLM-L6-v2`, $\beta=0.6$) using section-weighted requirement-to-chunk cosine scoring.
* **Taxonomy-Grounded Skill Extraction:** Matches candidate terms against **ESCO v1.2.1** (European Skills, Competences, Qualifications and Occupations) enriched with a curated 314-term tech alias overlay, short-token context guards (preventing "C", "R", "Go" false positives), and 3-tier evidence grading (*Demonstrated*, *Listed-only*, *Tentative*).
* **Anti-Gaming Stuffing Penalty:** Quantifies the ratio of uncorroborated skills to demonstrated experience, actively penalizing keyword stuffing.
* **GDPR Right-to-Erasure:** Native cascading deletion removing candidates, chunk embeddings, and audit reports with cryptographic and database integrity (`ON DELETE CASCADE`).
* **Interactive UI:** Click any skill chip in the web report to immediately scroll to and highlight its supporting resume text span with a glowing pulse animation.

---

## 2. End-to-End System Architecture

```
                 +-------------------------------------------------+
                 |                Input Documents                  |
                 |  - Candidate Resume (PDF / Text)                |
                 |  - Structured Job Description (JD)              |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |           Phase 1: Ingestion Pipeline           |
                 |  - Magic-byte & Encryption Validation           |
                 |  - Dual Extractor (pdfplumber + pypdf)          |
                 |  - Section Segmentation & Date Parsing          |
                 |  - PII Redaction (spaCy NER + Regex)            |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |       Phase 2: ESCO v1.2.1 Skill Extraction     |
                 |  - Curated Alias Overlay (314 entries)          |
                 |  - Short-Token Guards (C / R / Go context)      |
                 |  - 3-Tier Evidence Classification               |
                 |  - Keyword Stuffing Ratio Detector              |
                 +-----------------------+-------------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
  +-----------------------------------+     +-----------------------------------+
  | Phase 3A: Lexical Retrieval       |     | Phase 3B: Semantic Retrieval      |
  | - Frozen Reference TF-IDF Vocab   |     | - all-MiniLM-L6-v2 Embeddings     |
  | - Custom Tech-Aware Tokenizer     |     | - Section-Weighted Req-to-Chunk   |
  | - Percentile-Calibrated Score     |     | - Token-Aware Section Chunking    |
  +-----------------+-----------------+     +-----------------+-----------------+
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |        Phase 3C: Hybrid Fusion Engine           |
                 |  - Convex Linear Fusion: S_hybrid = α·S_lex + β·S_sem
                 |  - Optional Reciprocal Rank Fusion (k=60)       |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |      Phase 3D: Multiplicative Constraints       |
                 |  - Tenure Penalty: (0.85)^Δyears                |
                 |  - Degree Requirement Penalty: 0.70x            |
                 |  - Must-Have Skill Penalty: 0.80x               |
                 |  - Keyword Stuffing Multiplier: 0.85x           |
                 +-----------------------+-------------------------+
                                         |
                                         v
                 +-------------------------------------------------+
                 |          Phase 3E: Explainable FitReport        |
                 |  - Calibrated Fit Index (0–100)                 |
                 |  - Deterministic Natural Language Summary       |
                 |  - Itemized Audit Trail & Evidence Spans        |
                 +-----------------------+-------------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
  +-----------------------------------+     +-----------------------------------+
  | PostgreSQL 15 + pgvector (HNSW)   |     | Modern Web UI & REST API          |
  | - HNSW Cosine Index (m=16, ef=64) |     | - Drag & Drop PDF Dropzone        |
  | - Full-Text GIN Search            |     | - Interactive Evidence Glow       |
  | - GDPR Cascading Deletions        |     | - Side-by-Side Candidate Compare  |
  +-----------------------------------+     +-----------------------------------+
```

---

## 3. Calibrated Fit Index Mathematical Formulation

The final **Fit Index** $F \in [0.0, 100.0]$ is computed deterministically:

$$S_{\text{hybrid}} = \alpha \cdot \Phi_{\text{calib}}(S_{\text{lex}}) + \beta \cdot \Phi_{\text{calib}}(S_{\text{sem}})$$

Where:
* $\alpha = 0.4$, $\beta = 0.6$ ($\alpha + \beta = 1.0$)
* $\Phi_{\text{calib}}$ represents percentile-calibrated empirical normalisation against reference distributions.
* $S_{\text{sem}} = \frac{1}{\sum w_i} \sum_{i \in \text{Reqs}} w_i \max_{c \in \text{Chunks}} \cos(\mathbf{e}_{\text{req}_i}, \mathbf{e}_{c}) \cdot W_{\text{section}}(c)$

### Multiplicative Constraint Deductions:
$$S_{\text{final}} = S_{\text{hybrid}} \times \prod_{k} \gamma_k$$

| Constraint Factor ($\gamma_k$) | Condition | Multiplier Formula |
|---|---|---|
| **Experience Tenure** | $\text{Years}_{\text{cand}} < \text{Years}_{\text{req}}$ | $(0.85)^{\Delta \text{years}}$ |
| **Academic Credential** | $\text{Degree}_{\text{cand}} < \text{Degree}_{\text{req}}$ | $0.70$ |
| **Must-Have Skill** | Missing mandatory core competency | $0.80$ |
| **Keyword Stuffing** | $\text{Ratio}(\text{Listed} / \text{Demonstrated}) > 3.0$ | $0.85$ |

$$\text{Fit Index} = S_{\text{final}} \times 100.0$$

---

## 4. Empirical Evaluation Results (Phase 6 Benchmarks)

Evaluated across **50 structured JDs**, **100 synthetic resumes** across 7 personas (Senior, Mid, Junior, Stuffed, Missing-Exp, Career-Changer, Unrelated), and **500 gold-labeled relevance pairs** (relevance $\in \{0, 1, 2\}$) with a clean 30/70 Dev/Test split.

### Table 1: E1 Retrieval Architecture Benchmark (Test Set, $N=350$ pairs, 35 JDs)

| Model Architecture | P@5 | Recall@5 | MRR | nDCG@5 | Bootstrap 95% CI | Wilcoxon $p$-val vs Hybrid |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF Lexical** | 0.8171 | 0.8929 | 0.9857 | **0.9192** | [0.884, 0.950] | $p = 0.441$ |
| **Semantic (MiniLM)** | 0.8057 | 0.8643 | 0.9429 | **0.8844** | [0.841, 0.923] | $p = 0.082$ |
| **Hybrid (ResumeFit)** | 0.8171 | 0.8810 | 0.9762 | **0.9010** | [0.860, 0.938] | *Reference Baseline* |

### Table 2: E2 Alpha Parameter Sensitivity Sweep (Dev Set)

| Weight Configuration ($\alpha$ Lexical / $\beta$ Semantic) | nDCG@5 | P@5 | Recall@5 | MRR |
|---|:---:|:---:|:---:|:---:|
| $\alpha = 0.0$ (Pure Semantic) | 0.9056 | 0.8267 | 0.9333 | 0.9667 |
| $\alpha = 0.2$ | 0.9168 | 0.8533 | 0.9444 | 0.9667 |
| $\alpha = 0.4$ (Default Config) | 0.9400 | 0.8667 | 0.9556 | 1.0000 |
| $\alpha = 0.5$ (Balanced Equal) | 0.9400 | 0.8667 | 0.9556 | 1.0000 |
| **$\alpha = 0.7$ (Empirical Optimum on Dev)** | **0.9539** | **0.8800** | **0.9667** | **1.0000** |
| $\alpha = 1.0$ (Pure Lexical) | 0.9377 | 0.8800 | 0.9667 | 0.9667 |

### Table 3: E3 Component Ablations

| Ablation Condition | Test nDCG@5 | $\Delta$ vs Full Hybrid | Key Takeaway |
|---|:---:|:---:|---|
| **Full Hybrid Pipeline** | **0.9010** | — | Complete explainable pipeline |
| **No Constraints (Raw Fusion)** | 0.8911 | **-0.0099** | Constraints properly penalize under-qualified profiles |
| **Mean-Pooled Semantic** | 0.8997 | **-0.0013** | Requirement-to-chunk matching outperforms global pooling |
| **No Section Weights** | 0.9010 | 0.0000 | Robust on clean resumes; prevents false section inflation |
| **Reciprocal Rank Fusion (RRF)** | 0.9010 | 0.0000 | RRF provides equivalent ranking without score calibration |

### Key Robustness & Fairness Findings:
* **E4 Anti-Gaming Validation:** Keyword-stuffed resumes suffer an average **0.1464** score reduction due to uncorroborated skill penalties.
* **E5 Demographic Fairness Probe:** Swapping candidate names across diverse cultural and gender backgrounds (e.g., *John Smith*, *Aisha Al-Mansoor*, *Carlos Rodriguez*, *Priya Patel*, *Chen Wei*, *Keiko Tanaka*) with pre-embedding PII redaction active yielded **variance $\sigma^2 = 0.0000$** and **max difference $\Delta = 0.0000$**, proving strict demographic invariance.
* **Skill Extraction Precision/Recall:** Micro-F1 of **90.0%** and Macro-F1 of **89.8%** across 20 hand-labeled gold resumes.

---

## 5. Web Interface & UI Features

ResumeFit features a handcrafted, dark-mode first design system (Deep Teal `#0D7377`, Cyan Glow `#14FFEC`, Dark Slate `#0B0F14`) built with vanilla CSS tokens:
1. **Interactive Evidence Span Highlighting:** Clicking any extracted skill chip instantly scrolls the candidate's resume preview pane into view and wraps the exact corroborating text span in an animated pulsing cyan highlight.
2. **Side-by-Side Comparison (`/compare`):** Enter a job description and paste 2–3 candidate profiles to generate an auditable ranking matrix.
3. **Candidate Pool Management (`/dashboard`):** Real-time client-side search, filtering by fit tier, and one-click GDPR Right-to-Erasure cascading deletions.
4. **Academic Benchmarks Dashboard (`/eval`):** Renders live system metrics, ESCO taxonomy stats, TF-IDF vocabulary volume, and E1–E6 benchmark tables.
5. **Theme Switcher:** Seamless Dark/Light theme toggle with zero-flicker `localStorage` persistence.

---

## 6. Quickstart & Installation

### Local Development (Python 3.11 + uv)

```bash
# 1. Clone repository
git clone https://github.com/shivam/ResumeFit.git
cd ResumeFit

# 2. Install dependencies with uv
uv sync

# 3. Seed ESCO taxonomy and fit reference TF-IDF vectorizer
uv run python scripts/fetch_esco.py
uv run python scripts/build_tfidf_reference.py

# 4. Run tests and lint checks
uv run pytest tests/
uv run ruff check .

# 5. Launch web application
uv run python -m resumefit.app
# Open http://localhost:5000 in your browser
```

### Docker Deployment

```bash
# Build and launch web app + PostgreSQL with pgvector
docker compose up --build

# Web UI: http://localhost:5000
# Health check: http://localhost:5000/api/health
```

### Cloud Deployment (Render Blueprint)
The project includes a production-ready `render.yaml` blueprint. Connect your GitHub repository to Render and deploy both the web container and managed PostgreSQL instance with one click.

---

## 7. Command Line Interface (CLI)

ResumeFit includes a full-featured terminal CLI for batch processing, scripting, and offline audits:

```bash
# Analyze a resume PDF against a Job Description
python -m core.cli analyze \
  --resume tests/fixtures/pdfs/single_col_1.pdf \
  --jd "Role: Backend Engineer\nRequirements: Python, PostgreSQL, Docker" \
  --output report.json

# Compare Lexical vs Semantic vs Hybrid models side-by-side
python -m core.cli compare \
  --resume tests/fixtures/pdfs/two_col_1.pdf \
  --jd data/synthetic/jds/jd_001.txt
```

---

## 8. Limitations & Honest Engineering Constraints

1. **Language Scope:** Only English resumes are supported. Non-English resumes are detected via ASCII-to-Unicode ratio thresholds and rejected with an explicit `UnsupportedDocument(reason="non-english")` error.
2. **Text Layer Requirement:** Image-only scanned PDFs without an OCR text layer are intentionally rejected with `UnsupportedDocument(reason="scanned/image-only")` rather than silently passing degraded content.
3. **Hardware Constraints:** All embeddings (`all-MiniLM-L6-v2`) and TF-IDF inferences are CPU-optimized; cold start model loading takes ~1.2s on standard x86_64 hardware.
4. **Decision Support Disclaimer:** ResumeFit is strictly an explainable decision support tool for recruiters and candidates. It does not automate hiring decisions or predict candidate job success.

---

## 9. Comprehensive Documentation Sitemap

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Technical blueprint, component sequence diagrams, and pipeline-to-code mapping |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 29+ Architecture Decision Records (ADRs) detailing technical rationales |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Complete empirical protocol, statistical tests, metrics formulation, and validity threats |
| [`docs/ETHICS_AND_PRIVACY.md`](docs/ETHICS_AND_PRIVACY.md) | NIST AI RMF framework alignment and GDPR Right-to-Erasure compliance |
| [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md) | 30 academic viva and technical defense interview questions with codebase-grounded answers |
| [`docs/THESIS_SUPPORT.md`](docs/THESIS_SUPPORT.md) | Report figures, tables, and chapter mapping for degree defense submission |
| [`docs/EXPLAIN_HNSW.md`](docs/EXPLAIN_HNSW.md) | PostgreSQL `EXPLAIN (ANALYZE, BUFFERS)` execution plans and HNSW index performance |

---

## 10. License

Released under the **MIT License**. Created by Shivam for academic defense and research in Explainable AI (XAI) and Natural Language Processing.
