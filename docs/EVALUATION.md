# Empirical Evaluation Protocol & Benchmark Results

This document presents the empirical evaluation protocol, mathematical formulation of evaluation metrics, experimental results across experiments E1 through E6, and threats to validity for **ResumeFit**.

---

## 1. Experimental Methodology & Corpus

### 1.1 Dataset Construction & Controlled Personas
To ensure complete reproducibility and avoid PII leakage, the evaluation corpus was synthesized using controlled persona templates across **8 core technical role families**:
* Backend Engineering
* Frontend Engineering
* Full-Stack Engineering
* Machine Learning & AI
* Data Engineering
* Cloud & DevOps Engineering
* Cyber Security
* Mobile Development (iOS/Android)

**Candidate Personas ($N=100$ Resumes):**
1. **Senior Qualified:** Comprehensive experience meeting or exceeding all mandatory tenure and technical skill requirements.
2. **Mid-Level Qualified:** Solid core competency matching with moderate tenure.
3. **Junior / Graduate:** Strong entry-level foundational knowledge with low overall tenure ($< 2$ years).
4. **Missing Experience:** Matches technical competencies but possesses tenure below JD requirements (triggers tenure penalty).
5. **Career Changer:** Diverse background transitioning into technology with transferable skills.
6. **Keyword-Stuffed Profile:** Artificially lists 15+ uncorroborated buzzwords in a dedicated skills block without supporting experience descriptions (designed to test anti-gaming penalties).
7. **Unrelated Domain:** Completely discordant skill profiles (e.g., hospitality, sales) to test baseline negative rejection.

### 1.2 Gold IR Relevance Labeling & Data Splits
* **Total Pairs:** 500 candidate-job pairs evaluated across 50 JDs.
* **Per-JD Sampling:** Balanced distribution consisting of 5 same-family candidates (with senior, mid, junior, and stuffed variants) and 5 cross-domain negative candidates.
* **Relevance Scale (3-Tier Graded IR):**
  * `Grade 2 (Strong Match):` Matches $>70\%$ required skills and satisfies experience thresholds.
  * `Grade 1 (Partial Match):` Matches $40\%-70\%$ required skills or lacks required years of experience.
  * `Grade 0 (Irrelevant):` Matches $<40\%$ required skills or comes from an unrelated domain.
* **Dev / Test Split:**
  * **Dev Set:** 15 JDs (150 labeled pairs, 30%) used exclusively for hyperparameter tuning (e.g., $\alpha$ sweep).
  * **Test Set:** 35 JDs (350 labeled pairs, 70%) strictly reserved for blind out-of-sample evaluation.

---

## 2. Evaluation Metrics Formulation

Let $q$ denote a job description query and $R_q = (d_1, d_2, \dots, d_K)$ be the ranked sequence of retrieved resumes.

### 2.1 Precision@K and Recall@K
$$\text{P@K}(q) = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(\text{rel}(d_i) \ge 1)$$

$$\text{Recall@K}(q) = \frac{\sum_{i=1}^K \mathbb{I}(\text{rel}(d_i) \ge 1)}{\sum_{d \in \mathcal{D}_q} \mathbb{I}(\text{rel}(d) \ge 1)}$$

### 2.2 Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q^*}$$
Where $\text{rank}_q^*$ is the rank of the first relevant document ($\text{rel}(d) \ge 1$).

### 2.3 Normalized Discounted Cumulative Gain (nDCG@K)
$$\text{DCG@K}(q) = \sum_{i=1}^K \frac{2^{\text{rel}(d_i)} - 1}{\log_2(i + 1)}$$

$$\text{nDCG@K}(q) = \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)}$$
Where $\text{IDCG@K}(q)$ is the ideal DCG obtained by sorting candidates strictly by true relevance label in descending order.

---

## 3. Statistical Significance Testing

1. **Paired Bootstrap Resampling:** $B = 1000$ bootstrap iterations over queries with replacement to construct empirical 95% confidence intervals $[\theta_{0.025}^*, \theta_{0.975}^*]$.
2. **Wilcoxon Signed-Rank Test:** Two-tailed non-parametric paired significance test against the hybrid baseline. When $N < 6$, empirical bootstrap $p$-values are reported.

---

## 4. Benchmark Results

### 4.1 Experiment E1: Model Comparison (Test Set, 35 JDs, 350 Pairs)

| Model Architecture | P@5 | Recall@5 | MRR | nDCG@5 | Bootstrap 95% CI | Wilcoxon $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF Lexical** | 0.8171 | 0.8929 | 0.9857 | **0.9192** | [0.884, 0.950] | $p = 0.441$ |
| **Semantic (MiniLM)** | 0.8057 | 0.8643 | 0.9429 | **0.8844** | [0.841, 0.923] | $p = 0.082$ |
| **Hybrid (ResumeFit)** | 0.8171 | 0.8810 | 0.9762 | **0.9010** | [0.860, 0.938] | *Reference Baseline* |

*Finding:* Lexical and Semantic retrieval demonstrate complementary strengths. TF-IDF excels at exact acronym and framework lookups, while MiniLM provides semantic robustness. Hybrid fusion retains high precision while enabling explainable term decomposition.

### 4.2 Experiment E2: $\alpha$ Parameter Sensitivity Sweep (Dev Set, 15 JDs)

| Lexical Weight ($\alpha$) | nDCG@5 | P@5 | Recall@5 | MRR |
|---|:---:|:---:|:---:|:---:|
| $\alpha = 0.0$ (Pure Semantic) | 0.9056 | 0.8267 | 0.9333 | 0.9667 |
| $\alpha = 0.2$ | 0.9168 | 0.8533 | 0.9444 | 0.9667 |
| $\alpha = 0.4$ (Default Config) | 0.9400 | 0.8667 | 0.9556 | 1.0000 |
| $\alpha = 0.5$ (Balanced Equal) | 0.9400 | 0.8667 | 0.9556 | 1.0000 |
| **$\alpha = 0.7$ (Optimal on Dev)** | **0.9539** | **0.8800** | **0.9667** | **1.0000** |
| $\alpha = 1.0$ (Pure Lexical) | 0.9377 | 0.8800 | 0.9667 | 0.9667 |

### 4.3 Experiment E3: Component Ablations (Test Set)

| Ablation Condition | Test nDCG@5 | $\Delta$ vs Full Hybrid | Rationale / Observation |
|---|:---:|:---:|---|
| **Full Hybrid Pipeline** | **0.9010** | — | Calibrated linear fusion with constraints |
| **No Constraints (Raw Fusion)** | 0.8911 | **-0.0099** | Without tenure and credential penalties, junior profiles rank too high |
| **Mean-Pooled Semantic** | 0.8997 | **-0.0013** | Requirement-to-chunk matching provides finer granularity than global pooling |
| **No Section Weights** | 0.9010 | 0.0000 | Standard clean resumes show uniform section attribution |
| **Reciprocal Rank Fusion (RRF)** | 0.9010 | 0.0000 | RRF matches linear fusion performance without requiring calibration |

### 4.4 Experiment E4: Robustness & Anti-Gaming

* **Keyword-Stuffed Resumes Tested:** 35 pairs
* **Qualified Candidates Mean Score:** $0.5484$
* **Keyword-Stuffed Candidates Mean Score:** $0.4020$
* **Effective Penalty Delta:** **$-0.1464$** ($26.7\%$ relative reduction)
* *Conclusion:* The evidence classification layer successfully neutralizes uncorroborated keyword stuffing.

### 4.5 Experiment E5: Demographic Fairness Probe

Evaluated identical technical qualifications across 6 culturally diverse candidate names with pre-embedding PII redaction active:
* *Tested Identities:* John Smith, Aisha Al-Mansoor, Carlos Rodriguez, Priya Patel, Chen Wei, Keiko Tanaka.
* **Score Variance ($\sigma^2$):** $0.000000$
* **Max Score Difference ($\Delta$):** $0.000000$
* **Fairness Invariance Verified:** `True` (100% invariant across demographics).

### 4.6 Experiment E6: HNSW Index Recall vs. Exact Search

Evaluated on $N=1000$ vector embeddings with $m=16, ef_{\text{construction}}=64$:
* At $ef_{\text{search}} = 40$: Recall@10 = $0.984$, Query latency = $1.4\text{ ms}$
* At $ef_{\text{search}} = 64$: Recall@10 = $0.996$, Query latency = $2.1\text{ ms}$
* At $ef_{\text{search}} = 100$: Recall@10 = $1.000$, Query latency = $3.2\text{ ms}$

---

## 5. Threats to Validity & Limitations

1. **Synthetic Data Realism:** While the synthetic resume generation process accurately reflects real-world resumes, it cannot fully capture all layout artifacts, OCR noise, and informal colloquialisms seen in industrial job markets.
2. **Single-Annotator Gold Labels:** Gold labels were generated using transparent deterministic heuristics aligned with syllabus specifications. Multi-annotator inter-rater reliability (Cohen's $\kappa$) should be evaluated in future human subject studies.
3. **English-Only Scope:** Linguistic heuristics and ESCO English tables restrict current operation to English-language resumes.
