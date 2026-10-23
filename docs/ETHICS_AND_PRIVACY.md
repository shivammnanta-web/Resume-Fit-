# Ethics, Privacy & NIST AI RMF Mapping

This document details the ethical principles, regulatory compliance mechanisms, and alignment with the **NIST AI Risk Management Framework (AI RMF 1.0)** implemented in **ResumeFit**.

---

## 1. System Scope & Regulatory Status

Under the European Union Artificial Intelligence Act (EU AI Act, Annex III), AI systems utilized in **recruitment, candidate screening, and employment decision-making** are categorized as **High-Risk AI Systems**.

ResumeFit is explicitly engineered as an **Explainable Decision Support System (EDSS)** rather than an autonomous decision-making engine:
* **No Autonomous Actions:** ResumeFit never issues hiring rejections or job offers. It outputs human-auditable analysis reports to assist human recruiters.
* **No Generative Hallucinations:** All scores derive from deterministic, inspectable mathematical operations.
* **Prohibition of "Probability" Claims:** The system does not claim to predict "success probability" or "performance potential"; it calculates document alignment against specified criteria.

---

## 2. NIST AI Risk Management Framework (AI RMF 1.0) Mapping

| NIST AI RMF Trustworthy Characteristic | Concrete ResumeFit Architectural Implementation | Codebase Module |
|---|---|---|
| **1. Valid & Reliable** | Dual-retrieval architecture combining TF-IDF lexical precision with MiniLM semantic alignment. Invariant tests guarantee that identical documents yield high scores and unrelated documents yield near-zero scores. | `core/matching/hybrid.py`, `tests/unit/test_matching.py` |
| **2. Safe** | Strict ingestion sanitation rejects corrupted, encrypted, or oversized inputs ($>5\text{MB}$, $>10$ pages) preventing buffer overflows, DoS attacks, and system crashes. | `core/ingestion/scan_detect.py` |
| **3. Secure & Resilient** | Isolated execution environment, rate-limited public APIs (`Flask-Limiter`), and non-root Docker runtime container (`appuser`). | `resumefit/limiter.py`, `Dockerfile` |
| **4. Accountable & Transparent** | Every score decomposes into lexical term attributions, semantic requirement matches, and itemized constraint deductions. System source code and evaluation benchmarks are 100% open and auditable. | `core/explain/report.py`, `docs/DECISIONS.md` |
| **5. Explainable & Interpretable** | Character-offset evidence linking: every matched skill links directly to the supporting resume sentence. Visual UI highlights exact corroborated spans with zero black-box summaries. | `core/skills/evidence.py`, `resumefit/static/js/app.js` |
| **6. Privacy-Preserving** | Automated PII redaction (names, emails, phones, URLs) prior to generating vector embeddings. PII scrub filters prevent personal identifiers from ever entering application logs. | `core/ingestion/pii.py`, `resumefit/logging.py` |
| **7. Fair with Harmful Bias Managed** | Demographic Invariance Validation (Experiment E5): Candidate names across 6 distinct cultural and gender identities produce identical scores ($\sigma^2 = 0.0000$). Demographic attributes are redacted before semantic encoding. | `evaluation/harness.py`, `docs/EVALUATION.md` |

---

## 3. GDPR Compliance: Right-to-Erasure (Article 17)

Under the European Union General Data Protection Regulation (GDPR, Article 17), individuals have the fundamental right to request the permanent erasure of their personal data without undue delay.

### Architectural Implementation:
1. **Database Cascading Integrity:**
   The PostgreSQL schema enforces cascading foreign key constraints:
   ```sql
   ALTER TABLE resume_chunks
     ADD CONSTRAINT fk_chunks_candidate
     FOREIGN KEY (candidate_id) REFERENCES candidate_resumes(id)
     ON DELETE CASCADE;

   ALTER TABLE candidate_skills
     ADD CONSTRAINT fk_skills_candidate
     FOREIGN KEY (candidate_id) REFERENCES candidate_resumes(id)
     ON DELETE CASCADE;

   ALTER TABLE match_reports
     ADD CONSTRAINT fk_reports_candidate
     FOREIGN KEY (candidate_id) REFERENCES candidate_resumes(id)
     ON DELETE CASCADE;
   ```
2. **REST API Endpoint (`DELETE /api/resumes/<id>`):**
   Executes atomic transaction deleting the primary candidate entity, which automatically cascades to all vector embeddings in pgvector, extracted skills, and historic match reports.
3. **Interactive UI Action:**
   The Candidate Pool Dashboard (`/dashboard`) and Report View (`/report/<id>`) feature a dedicated **Right-to-Erasure** modal button with confirmation safeguards.
4. **Zero Residual Vector Artifacts:**
   Because deletions cascade at the table row level in PostgreSQL, HNSW graph nodes corresponding to erased chunks are unlinked immediately without lingering latent data.

---

## 4. Anti-Gaming and Integrity Safeguards

Keyword stuffing has historically plagued ATS systems, where candidates insert white-text or unverified skill lists to artificially inflate ranking scores.

ResumeFit implements an explicit **3-Tier Evidence Classification**:
* **Demonstrated:** Extracted within employment history or project descriptions alongside context.
* **Listed-only:** Extracted from a standalone bulleted keyword block.
* **Tentative:** Extracted via low-confidence fuzzy matching.

If a profile's **Keyword Stuffing Ratio** ($\text{Listed-only} / \text{Demonstrated}$) exceeds $3.0$ with $\ge 15$ uncorroborated skills, the profile is flagged and penalized by $0.85\times$. Experiment E4 demonstrates that this reduces stuffed candidate scores by an average of $0.1464$ ($26.7\%$), protecting hiring integrity.
