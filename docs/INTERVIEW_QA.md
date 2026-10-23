# ResumeFit Academic Defense & Technical Viva Q&A Guide

This guide contains **30 comprehensive academic defense questions** accompanied by **codebase-grounded technical answers** to prepare for academic viva examinations and technical reviews.

---

### Category 1: System Architecture & Design Philosophy

#### Q1: What is ResumeFit, and what fundamental problem does it solve in modern hiring technology?
**Answer:**
ResumeFit is an explainable, auditable hybrid resume–job matching system designed to replace black-box ATS algorithms and opaque generative LLMs. Current industry systems either suffer from naive keyword rigidity (rejecting qualified candidates due to minor wording differences) or hallucination and demographic bias from generative LLMs. ResumeFit solves this by fusing lexical retrieval (TF-IDF with frozen reference vocabulary) and semantic retrieval (`all-MiniLM-L6-v2`) grounded in the European Commission's ESCO v1.2.1 taxonomy, outputting a calibrated Fit Index (0–100) with 100% traceable, character-offset evidence spans.

#### Q2: Why did you explicitly decide NOT to use a generative LLM (such as GPT-4 or Claude) for scoring or writing explanations?
**Answer:**
See ADR-001 in [`docs/DECISIONS.md`](DECISIONS.md). Generative LLMs are non-deterministic, prone to ungrounded hallucinations, difficult to statistically audit, and inherently susceptible to prompt injection and latent demographic biases. For a high-stakes application like employment evaluation (categorized as High-Risk under the EU AI Act), decision justification must be mathematically deterministic and auditable. In ResumeFit, all natural language summaries are generated via deterministic, templated rule engines directly mapped to extracted features in `core/explain/report.py`.

#### Q3: Why is ResumeFit characterized as a "Decision Support System" rather than an "Automated Decision Maker"?
**Answer:**
Under the EU AI Act and GDPR Article 22, automated decision-making that produces legal or significant effects without human intervention is strictly regulated and frequently unlawful. ResumeFit generates evidence-based ranking matrices and candidate gap analysis for human recruiters. It never issues automated rejections or job offers, and it strictly avoids claims of predicting "success probability."

#### Q4: How is the overall system structured across architectural layers?
**Answer:**
The system is structured into 8 modular layers:
1. `core/ingestion`: Document hygiene, dual extraction, normalization, section segmentation, PII redaction.
2. `core/skills`: ESCO v1.2.1 loading, alias overlays, 3-tier evidence classification.
3. `core/jd`: Structured job description parser.
4. `core/matching`: Token-aware chunking, TF-IDF lexical cosine, semantic requirement-to-chunk matching, calibrated percentile normalization, and multiplicative constraints.
5. `core/explain`: Auditable FitReport generator.
6. `core/db`: SQLAlchemy models, Alembic migrations, PostgreSQL 15 + pgvector HNSW indexing.
7. `resumefit/services.py`: High-level business orchestrator.
8. `resumefit/api` & `resumefit/web`: Flask REST endpoints and Jinja2 frontend.

#### Q5: What is the mathematical formulation of the final Fit Index?
**Answer:**
The Fit Index $F \in [0.0, 100.0]$ is formulated as:
$$S_{\text{hybrid}} = \alpha \cdot \Phi_{\text{calib}}(S_{\text{lex}}) + \beta \cdot \Phi_{\text{calib}}(S_{\text{sem}})$$
$$S_{\text{final}} = S_{\text{hybrid}} \times \prod_{k} \gamma_k$$
$$\text{Fit Index} = S_{\text{final}} \times 100.0$$
Where $\alpha=0.4, \beta=0.6$, $\Phi_{\text{calib}}$ is percentile-calibrated empirical normalisation, and $\gamma_k$ are multiplicative constraint penalties (tenure deficit $(0.85)^{\Delta\text{years}}$, degree deficit $0.70$, missing must-have skill $0.80$, keyword stuffing $0.85$).

#### Q6: How do you handle cold-start latency and container resource bounds?
**Answer:**
PyTorch is installed using CPU-only wheels (`--index-url https://download.pytorch.org/whl/cpu`), reducing image size by over 2GB. Sentence-Transformers and spaCy models are pre-cached during the Docker build stage. Memory footprint is bounded at $\approx 420\text{ MB}$, and cold start takes $< 1.5$ seconds without GPU requirements.

---

### Category 2: Ingestion, Hygiene & Preprocessing

#### Q7: How does ResumeFit detect and handle scanned or image-only PDF resumes?
**Answer:**
In `core/ingestion/scan_detect.py`, documents are inspected for character density after text extraction. If average character count is $< 50$ characters per page across all pages, the document is rejected with `UnsupportedDocument(reason=UnsupportedReason.SCANNED_OR_IMAGE_ONLY)`. Rather than silently degrading or guessing, the system returns an actionable user error advising the candidate to provide a text-layer PDF.

#### Q8: Why do you employ a dual-extractor strategy (pdfplumber + pypdf)?
**Answer:**
See ADR-002 in [`docs/DECISIONS.md`](DECISIONS.md). Standard PDF parsers frequently fail on multi-column layouts, tables, or corrupted font encodings. `pdfplumber` offers superior bounding-box spatial awareness for multi-column text flow. If `pdfplumber` fails or extracts garbled Unicode, `core/ingestion/extract.py` automatically falls back to `pypdf`.

#### Q9: How are overlapping employment periods handled when calculating total experience?
**Answer:**
In `core/ingestion/sections.py`, dates are parsed using regular expressions supporting formats like `Jan 2020 – Present`, `2018–2021`, `Summer 2019`. Date intervals $[S_i, E_i]$ are sorted and merged using a classic interval-scheduling merge algorithm: if $S_{i+1} \le E_i$, the intervals are merged into $[S_i, \max(E_i, E_{i+1})]$. This prevents candidates working simultaneous freelance or part-time roles from artificially doubling their calculated experience years.

#### Q10: How does ResumeFit guarantee candidate privacy during ingestion?
**Answer:**
In `core/ingestion/pii.py`, candidate text passes through a dual spaCy NER (`PERSON`, `ORG`) and regular expression pipeline (`EMAIL`, `PHONE`, `URL`). PII entities are replaced with typed tokens (`[EMAIL_REDACTED]`, `[PHONE_REDACTED]`). Crucially, raw PII is stripped *before* vector embeddings are calculated in `core/matching/semantic.py`, ensuring candidate names and personal contacts are never permanently embedded in vector space.

---

### Category 3: ESCO Taxonomy & Skill Extraction

#### Q11: What is ESCO, and why was version 1.2.1 chosen over O*NET or proprietary taxonomies?
**Answer:**
See ADR-011. ESCO (European Skills, Competences, Qualifications and Occupations) is an open, multilingual, standardized taxonomy maintained by the European Commission. Unlike O*NET (which focuses predominantly on broad occupational descriptors), ESCO provides granular skill entities ($13,800+$ concepts) with hierarchical relationships (`broader`, `narrower`, `related`). Version 1.2.1 is an official stable distribution.

#### Q12: How do you address modern technology gaps in ESCO (e.g., PyTorch, Kubernetes)?
**Answer:**
Official taxonomies lag behind contemporary software engineering tooling. ResumeFit implements a curated alias overlay in `data/aliases.yaml` (314 entries) mapping terms like `k8s` $\rightarrow$ `Kubernetes`, `pytorch` $\rightarrow$ `deep learning framework`, and `fastapi` $\rightarrow$ `web development framework`. This overlay is transparent, version-controlled, and auditable.

#### Q13: What are Short-Token Guards, and why are they necessary?
**Answer:**
Single- and double-letter programming languages like "C", "R", and "Go" create catastrophic false positive rates in naive substring or regex searches (e.g., matching "C" inside "CSS" or "Go" in "Good team player"). In `core/skills/extractor.py`, short tokens require strict word boundary guards and contextual proximity checks (e.g., "Go" must appear adjacent to words like "programming", "Golang", or peer languages like "Python/Go").

#### Q14: How does ResumeFit classify extracted skills into Evidence Tiers?
**Answer:**
In `core/skills/evidence.py`, skills are graded into three tiers:
1. **Demonstrated:** Located in `EXPERIENCE` or `PROJECTS` sections accompanied by action verbs and quantified impact.
2. **Listed-only:** Extracted strictly from isolated bullet points in a `SKILLS` header.
3. **Tentative:** Extracted via low-confidence fuzzy matching.
Every classified skill records the source section name and exact character offsets for UI highlighting.

#### Q15: How does the system detect and penalize keyword stuffing?
**Answer:**
Candidates frequently paste blocks of buzzwords without contextual backing. The system computes the **Keyword Stuffing Ratio**:
$$\text{Ratio} = \frac{\text{Count}(\text{Listed-only})}{\max(1, \text{Count}(\text{Demonstrated}))}$$
If $\text{Ratio} > 3.0$ with $\ge 15$ uncorroborated skills, `is_stuffed` is set to `True`, triggering a $0.85\times$ penalty. In Experiment E4, this decreased stuffed profile scores by an average of $0.1464$ ($26.7\%$).

---

### Category 4: Matching Engine & Dual Retrieval

#### Q16: Why use a frozen reference TF-IDF vocabulary instead of fitting dynamically on incoming resumes?
**Answer:**
See ADR-003. If TF-IDF is fit dynamically on each batch of resumes, term weights vary depending on who else applied that day, violating fair scoring invariants. By pre-fitting and persisting a frozen reference vocabulary of 2,685 domain terms (`data/artifacts/tfidf_reference.joblib`), two identical resumes evaluated months apart against the same JD will always receive the exact same lexical cosine score.

#### Q17: What custom tokenization is used for technical lexical matching?
**Answer:**
Standard NLTK or scikit-learn tokenizers strip punctuation, destroying technical tokens: `C++` becomes `C`, `node.js` becomes `node` and `js`, and `CI/CD` becomes `CI` and `CD`. In `core/matching/lexical.py`, `tech_tokenize()` preserves compound tokens containing dots, pluses, hashes, and slashes (`c++`, `c#`, `.net`, `scikit-learn`, `ci/cd`).

#### Q18: What is the embedding model used, and why was it chosen?
**Answer:**
We use `sentence-transformers/all-MiniLM-L6-v2`. It maps sentences into a 384-dimensional dense vector space. With only 22.7 million parameters, it is $\approx 5\times$ faster than BERT-base while retaining $>99\%$ of retrieval performance on standard semantic search benchmarks, allowing real-time CPU inference in $\approx 220\text{ ms}$.

#### Q19: What is Section-Weighted Requirement-to-Chunk Matching?
**Answer:**
Rather than averaging the entire resume into a single global embedding (which dilutes specific achievements), the resume is segmented into token-bounded section chunks ($\le 200$ tokens). For each requirement $r_i$ in the JD, the engine finds the maximum cosine similarity across all resume chunks $c$, weighted by the section priority $W(c)$:
$$S_{\text{sem}} = \frac{1}{\sum w_i} \sum_{i} w_i \max_{c} \left( \cos(\mathbf{e}_{r_i}, \mathbf{e}_c) \cdot W(c) \right)$$
Experience carries weight 1.0, projects 0.9, skills 0.5, and education 0.4.

#### Q20: Why do you perform Percentile Normalization on raw similarity scores?
**Answer:**
Raw TF-IDF cosines are typically sparse and low ($0.15$–$0.40$), whereas MiniLM cosine similarities are dense and high ($0.50$–$0.85$). Simply adding them $(\alpha \cdot S_{\text{lex}} + \beta \cdot S_{\text{sem}})$ would result in the semantic score dominating the ranking regardless of $\alpha$. Percentile calibration maps raw scores against empirical cumulative distribution functions (ECDF), aligning both onto a uniform scale prior to linear fusion.

#### Q21: What is Reciprocal Rank Fusion (RRF), and how does it compare to Convex Linear Fusion?
**Answer:**
RRF is a rank-aggregation method that combines rankings without needing calibrated scores:
$$\text{RRF}(d) = \sum_{m \in \{\text{lex}, \text{sem}\}} \frac{1}{k + \text{rank}_m(d)}$$
In Experiment E3, RRF produced identical top-5 rankings to calibrated linear fusion ($\text{nDCG}@5 = 0.9010$), proving that calibrated linear fusion preserves rank order while offering the added benefit of an intuitive $0$–$100$ Fit Index.

---

### Category 5: Constraints & Explainability

#### Q22: Why are constraints applied multiplicatively rather than as additive deductions?
**Answer:**
Additive penalties ($S - 0.20$) can cause scores to drop below zero or produce disproportionately harsh penalties on mediocre scores while having negligible impact on high scores. Multiplicative penalties ($S \times 0.85$) scale proportionally with candidate quality and maintain scores within $[0.0, 1.0]$.

#### Q23: How does the experience tenure penalty mathematically behave?
**Answer:**
If a candidate has less than the required tenure, the penalty scales exponentially with the deficit:
$$\gamma_{\text{years}} = (0.85)^{\Delta \text{years}}$$
A 1-year deficit results in an $0.85\times$ multiplier; a 2-year deficit results in $0.7225\times$. This guarantees smooth, monotonic degradation rather than a cliff-edge disqualification.

#### Q24: What is the structure of the `FitReport` object?
**Answer:**
`FitReport` (in `core/explain/report.py`) contains:
* `candidate_name`, `role_title`, `fit_index` ($0$–$100$), `verdict_headline`, and `narrative_summary`.
* `scores`: Raw and normalized lexical, semantic, hybrid, and final scores.
* `skills_analysis`: Required vs preferred coverage percentage, itemized coverage table with evidence tiers and character spans.
* `lexical_attribution`: Top contributing n-grams and missing terms.
* `semantic_attribution`: Requirement-level evidence alignments and qualitative ratings.
* `constraints`: Itemized deduction audit trail with exact formulas and explanations.

#### Q25: How does the web UI highlight supporting evidence in real-time?
**Answer:**
In `resumefit/static/js/app.js`, each skill chip stores its supporting text span in `data-span`. When clicked, JavaScript extracts the span, locates the matching sentence in the resume preview container, wraps it in `<mark class="evidence-highlight">`, smoothly scrolls it into view, and triggers a CSS keyframe pulse animation.

---

### Category 6: Database, Vector Search & GDPR

#### Q26: Why choose PostgreSQL 15 + pgvector over dedicated vector databases (Pinecone, Milvus, Qdrant)?
**Answer:**
See ADR-004. Dedicated vector databases introduce external network dependencies, separate data stores, and complex eventual-consistency synchronization issues. `pgvector` allows atomic relational transactions, foreign-key cascading deletes, full-text GIN search, and vector embeddings in a single unified ACID-compliant PostgreSQL database.

#### Q27: What is HNSW, and what index parameters were configured?
**Answer:**
HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor algorithm with logarithmic search time complexity $O(\log N)$. We configure:
* `m = 16`: Number of bidirectional links per graph node (balanced memory vs graph connectivity).
* `ef_construction = 64`: Search depth during index building (ensures high graph recall).
* `ef_search = 64`: Query exploration depth yielding $99.6\%$ recall at $2.1\text{ ms}$ latency (Experiment E6).

#### Q28: How does ResumeFit achieve GDPR Right-to-Erasure compliance?
**Answer:**
In `core/db/models.py`, all child tables (`resume_chunks`, `candidate_skills`, `match_reports`) declare `ON DELETE CASCADE` foreign keys referencing `candidate_resumes(id)`. When `DELETE /api/resumes/<id>` is called, an atomic SQL `DELETE FROM candidate_resumes WHERE id = :id` immediately and permanently purges the candidate record, all associated 384-dimensional chunk embeddings, and historic match reports.

---

### Category 7: Evaluation, Fairness & Defense Strategy

#### Q29: What were the findings of the demographic fairness probe (Experiment E5)?
**Answer:**
In `evaluation/harness.py`, technically identical candidate resumes were evaluated with names representing diverse genders and ethnic backgrounds (*John Smith*, *Aisha Al-Mansoor*, *Carlos Rodriguez*, *Priya Patel*, *Chen Wei*, *Keiko Tanaka*). With pre-embedding PII redaction enabled, score variance was strictly $\sigma^2 = 0.000000$ and maximum score difference was $\Delta = 0.000000$, empirically demonstrating demographic invariance.

#### Q30: If challenged by an examiner on why lexical retrieval achieved higher nDCG than semantic retrieval in Experiment E1, how do you defend the hybrid architecture?
**Answer:**
In technical recruitment benchmarks with specific skill requirements (e.g., "Docker", "PostgreSQL"), lexical TF-IDF with exact tokenization performs exceptionally well because technical names are unambiguous. However, pure lexical search fails completely when candidates use synonyms (e.g., "orchestrating microservices" vs "container coordination") or when matching transferable conceptual responsibilities. As shown in Experiment E2, blending lexical ($\alpha=0.4$) with semantic ($\beta=0.6$) produces a robust, explainable model with higher overall recall and resilience against surface-level phrasing variations.
