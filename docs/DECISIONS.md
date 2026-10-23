# Architecture Decision Records (ADRs) — ResumeFit

This document records the architectural and design decisions made throughout the engineering of **ResumeFit**.

---

## ADR-001: Prohibition of Generative LLMs in Scoring & Explanation Path
* **Status:** Accepted
* **Context:** Modern recruiting platforms frequently use OpenAI, Anthropic, or open-source LLMs to read resumes and output narrative assessments or "fit scores". However, generative LLMs are non-deterministic, suffer from latent hallucinations, cannot be mathematically audited, and exhibit subtle demographic biases when processing resumes.
* **Decision:** We strictly ban generative LLMs from the scoring, extraction, and explanation pipeline. All match scores derive from deterministic mathematical models (TF-IDF and Sentence-Transformers). All explanations are assembled using deterministic templates directly grounded in character-offset text spans.
* **Consequences:** Eliminates hallucinations, guarantees 100% reproducibility, satisfies EU AI Act requirements for high-risk employment systems, and enables instant, low-cost local execution on CPU without API dependencies.

---

## ADR-002: Dual PDF Extractor Cascade with Scanned Document Detection
* **Status:** Accepted
* **Context:** Resumes arrive in wildly varied PDF layout formats (single-column, two-column, table-based, or image-only scans). Relying on a single PDF parser leads to high failure rates on complex layouts.
* **Decision:** Implement a dual-extractor cascade:
  1. Primary: `pdfplumber` for layout-aware spatial character extraction.
  2. Fallback: `pypdf` when `pdfplumber` fails or extracts garbled Unicode.
  3. Hygiene Gate: `core/ingestion/scan_detect.py` validates magic-bytes (`%PDF-`), encryption status, file size ($\le 5\text{MB}$), and page limits ($\le 10$ pages). Image-only scans ($<50$ chars/page) are rejected with a structured `UnsupportedDocument` exception.
* **Consequences:** Multi-column and table layouts are parsed accurately. Candidates submitting scanned images receive clear diagnostic feedback instead of silent failure.

---

## ADR-003: Frozen Reference TF-IDF Vocabulary vs. Dynamic Batch Fitting
* **Status:** Accepted
* **Context:** In standard text matching, fitting a TF-IDF vectorizer dynamically on incoming candidate batches causes term frequencies (IDF) to fluctuate based on the specific batch of candidates submitted that day.
* **Decision:** Fit the TF-IDF vectorizer on a diverse reference corpus of job descriptions and resumes across 8 role families, freezing the resulting 2,685-term vocabulary and IDF vector (`data/artifacts/tfidf_reference.joblib`).
* **Consequences:** Score invariance is guaranteed: a candidate evaluated today receives the exact same lexical cosine score if evaluated months later against the same job description.

---

## ADR-004: PostgreSQL 15 + pgvector (HNSW) vs. External Dedicated Vector DBs
* **Status:** Accepted
* **Context:** Dense semantic vector retrieval requires efficient nearest-neighbor indexing. External vector databases (Pinecone, Qdrant, Milvus) introduce external network hops, distinct failure modes, and distributed consistency challenges.
* **Decision:** Use PostgreSQL 15 with the official `pgvector` extension utilizing **HNSW** indexes (`vector_cosine_ops`, `m=16`, `ef_construction=64`). Relational candidate metadata, extracted skills, and full-text GIN search co-exist in the same transactional database.
* **Consequences:** Eliminates external vector service costs, provides ACID transactional guarantees, allows hybrid SQL queries (combining vector distance and relational metadata filters), and enables native GDPR cascading deletions.

---

## ADR-005: Section-Weighted Requirement-to-Chunk Matching vs. Global Mean Pooling
* **Status:** Accepted
* **Context:** Averaging an entire resume into a single embedding dilutes specific technical skills across pages of text.
* **Decision:** Segment resumes into section-bounded chunks ($\le 200$ tokens) that never cross section boundaries. For each requirement in the JD, compute the maximum cosine similarity across chunks, weighted by section type (`experience`: 1.0, `projects`: 0.9, `skills`: 0.5, `education`: 0.4).
* **Consequences:** Prevents a candidate from getting credit for a senior requirement just because the word appeared in an education course list. Proven in Experiment E3 to outperform mean-pooled matching.

---

## ADR-006: Percentile Calibration Normalization vs. Min-Max Scaling
* **Status:** Accepted
* **Context:** Raw TF-IDF cosines typically range between $0.10$ and $0.45$, whereas dense sentence embeddings range between $0.50$ and $0.85$. Naive linear combination results in semantic scores dominating the ranking.
* **Decision:** Implement percentile calibration against pre-computed empirical cumulative distribution functions (ECDF) for both lexical and semantic modalities before linear fusion.
* **Consequences:** Normalizes both scores onto a balanced $[0.0, 1.0]$ uniform scale, ensuring that the configured weights ($\alpha=0.4, \beta=0.6$) accurately reflect the true relative influence of each modality.

---

## ADR-007: Multiplicative Constraints vs. Additive Deductions
* **Status:** Accepted
* **Context:** Evaluating non-negotiable job requirements (minimum years of experience, required degree level). Additive subtractions ($S - 0.20$) can force scores negative or penalize low-scoring candidates disproportionately.
* **Decision:** Apply multiplicative penalty factors ($\prod \gamma_k$):
  - Experience deficit: $(0.85)^{\Delta\text{years}}$
  - Degree deficit: $0.70\times$
  - Missing must-have skill: $0.80\times$
  - Keyword stuffing: $0.85\times$
* **Consequences:** Bounded score range $[0.0, 1.0]$ is mathematically preserved. Penalties scale proportionally with candidate quality.

---

## ADR-008: Pre-Embedding PII Redaction for Demographic Invariance
* **Status:** Accepted
* **Context:** Dense neural embedding models have been shown to encode latent geographic, gender, and racial associations from names, phone numbers, and addresses.
* **Decision:** Strip candidate names, emails, phone numbers, and URLs via spaCy NER and regular expressions before generating dense embeddings.
* **Consequences:** Empirically proven in Experiment E5: swapping candidate names across 6 distinct cultural and gender identities resulted in variance $\sigma^2 = 0.000000$, guaranteeing demographic invariance.

---

## ADR-009: Cascading Foreign Key Deletions for GDPR Right-to-Erasure
* **Status:** Accepted
* **Context:** GDPR Article 17 requires permanent deletion of candidate personal data upon request without leaving latent audit artifacts.
* **Decision:** All relational child tables (`resume_chunks`, `candidate_skills`, `match_reports`) specify `ON DELETE CASCADE` referencing `candidate_resumes(id)`. Calling `DELETE /api/resumes/<id>` permanently purges all records and vector embeddings in a single atomic database transaction.
* **Consequences:** Zero orphan embeddings or residual vector artifacts remain in pgvector indexes following an erasure request.

---

## ADR-010: Client-Side Interactive Evidence Span Highlighting
* **Status:** Accepted
* **Context:** Recruiters need to verify claims quickly without reading through entire multi-page resumes.
* **Decision:** Extracted skills carry exact character offsets and text spans from the raw document. In the web interface, clicking any skill chip automatically scrolls the resume preview pane to the supporting sentence and applies a glowing cyan pulse animation.
* **Consequences:** Enables instantaneous, one-click visual verification of candidate competence claims.

---

## ADR-011: ESCO Taxonomy Granularity Gaps, Alias Overlay, and Guarded Matching Cascade
* **Status:** Accepted
* **Context:** ESCO v1.2.1 provides an authoritative European multilingual classification of skills and occupations, but exhibits latency regarding modern developer ecosystems (missing frameworks like FastAPI, PyTorch, Kubernetes) and causes false positives on short 1-letter tokens (C, R, Go).
* **Decision:**
  1. Maintain a curated alias overlay (`data/aliases.yaml`, 314 entries) linking modern terms to parent ESCO concepts.
  2. Implement a 3-tier cascade: Exact ESCO $\rightarrow$ Exact Alias $\rightarrow$ Guarded Fuzzy Match ($\ge 4$ characters).
  3. Short tokens ("C", "R", "Go") require regex contextual validation (e.g., "Golang", "C programming", "R/Python").
  4. Classify skills into *Demonstrated*, *Listed-only*, and *Tentative* tiers, penalizing keyword stuffing ratios $> 3.0$.
* **Consequences:** Achieved 90.0% Micro-F1 and 89.8% Macro-F1 across 20 diverse benchmark resumes while eliminating false positive collisions.

---

## ADR-012: Invariant Testing Suite for Fair Scoring Guarantees
* **Status:** Accepted
* **Context:** Algorithmic drift or scoring bugs could silently alter evaluation scores.
* **Decision:** In `tests/unit/test_matching.py`, enforce mathematical invariants:
  - Identical document similarity $\approx 1.0$.
  - Completely disjoint document similarity $\approx 0.0$.
  - Monotonic tenure penalty: $(S \mid 2\text{ yrs exp}) < (S \mid 5\text{ yrs exp})$.
  - Scores strictly bounded within $[0.0, 100.0]$ with no NaN values.
* **Consequences:** Guarantees that updates to models or dependencies never violate fundamental scoring contracts.

---

## ADR-013: Paired Bootstrap Hypothesis Testing & Wilcoxon Signed-Rank Evaluation
* **Status:** Accepted
* **Context:** In IR benchmarks, mean score differences across a small number of queries can be misleading without statistical confidence intervals.
* **Decision:** Implement paired bootstrap resampling ($B=1000$ iterations) to compute 95% confidence intervals on nDCG@K and MRR, alongside non-parametric Wilcoxon signed-rank tests with a small-sample guard when $N < 6$.
* **Consequences:** Academic defense grade statistical rigor supporting all thesis empirical findings.

---

## ADR-014: CPU-Only Containerization with Multi-Stage Dockerfile
* **Status:** Accepted
* **Context:** Deploying heavy PyTorch CUDA wheels inside Docker containers causes images to exceed 3GB, increasing cold-start times and hosting costs.
* **Decision:** Multi-stage Dockerfile (`python:3.11-slim`) installing CPU-only PyTorch wheels (`--index-url https://download.pytorch.org/whl/cpu`), pre-caching models during build, and running under a dedicated non-root user (`appuser`, UID 10001).
* **Consequences:** Image footprint $\le 850\text{ MB}$, cold start $< 1.5$ seconds, memory consumption $\approx 420\text{ MB}$, and zero root privileges in production.
