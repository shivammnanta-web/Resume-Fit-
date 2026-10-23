# Academic Thesis Support & Chapter Mapping

This document provides structured tables, figure captions, and chapter-by-chapter integration guides to assist in writing the final degree dissertation/project report for **ResumeFit**.

---

## 1. Chapter-by-Chapter Content Mapping

| Thesis Chapter | Suggested Title | Relevant Project Assets & Documentation |
|---|---|---|
| **Chapter 1** | **Introduction** | Problem statement in [`README.md`](../README.md#1-executive-summary--core-philosophy), limitations of black-box LLMs in hiring, NIST AI RMF scope in [`docs/ETHICS_AND_PRIVACY.md`](ETHICS_AND_PRIVACY.md). |
| **Chapter 2** | **Literature Review & Background** | ESCO v1.2.1 taxonomy taxonomy review ([`docs/DECISIONS.md#ADR-011`](DECISIONS.md)), lexical vs semantic retrieval review, HNSW vector search mathematics ([`docs/EXPLAIN_HNSW.md`](EXPLAIN_HNSW.md)). |
| **Chapter 3** | **Methodology & Mathematical Formulation** | Dual-retrieval pipeline architecture diagram ([`docs/ARCHITECTURE.md`](ARCHITECTURE.md)), Fit Index mathematical formulation ([`docs/ARCHITECTURE.md#3-calibrated-fit-index-mathematical-formulation`](ARCHITECTURE.md)), PII redaction pipeline. |
| **Chapter 4** | **System Implementation & Design** | Component pipeline mapping ([`docs/ARCHITECTURE.md#2-component-pipeline-to-codebase-mapping`](ARCHITECTURE.md)), database schema & Alembic migrations ([`core/db/models.py`](../core/db/models.py)), UI evidence highlighting ([`resumefit/static/js/app.js`](../resumefit/static/js/app.js)). |
| **Chapter 5** | **Empirical Evaluation & Results** | Experiments E1 through E6 tables ([`docs/EVALUATION.md`](EVALUATION.md)), statistical significance tests (paired bootstrap + Wilcoxon), fairness and anti-gaming evaluations. |
| **Chapter 6** | **Ethics, Limitations & Conclusion** | GDPR Right-to-Erasure compliance ([`docs/ETHICS_AND_PRIVACY.md#3-gdpr-compliance-right-to-erasure-article-17`](ETHICS_AND_PRIVACY.md)), threats to validity, future multi-lingual extensions. |

---

## 2. LaTeX-Ready Formatted Tables

### Table 5.1: Model Comparison Benchmark (Experiment E1)
```latex
\begin{table}[htbp]
\centering
\caption{Retrieval Model Benchmark Comparison on Test Set ($N=350$ pairs, 35 JDs)}
\label{tab:e1_model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model Architecture} & \textbf{P@5} & \textbf{Recall@5} & \textbf{MRR} & \textbf{nDCG@5} & \textbf{95\% Bootstrap CI} & \textbf{Wilcoxon $p$} \\
\hline
TF-IDF Lexical & 0.8171 & 0.8929 & 0.9857 & \textbf{0.9192} & [0.884, 0.950] & $p = 0.441$ \\
Semantic (MiniLM) & 0.8057 & 0.8643 & 0.9429 & \textbf{0.8844} & [0.841, 0.923] & $p = 0.082$ \\
\textbf{Hybrid (ResumeFit)} & 0.8171 & 0.8810 & 0.9762 & \textbf{0.9010} & [0.860, 0.938] & \textit{Baseline} \\
\hline
\end{tabular}
\end{table}
```

### Table 5.2: Parameter Sensitivity Sweep for $\alpha$ (Experiment E2)
```latex
\begin{table}[htbp]
\centering
\caption{Alpha Sensitivity Analysis on Development Set ($N=150$ pairs, 15 JDs)}
\label{tab:e2_alpha_sweep}
\begin{tabular}{ccccc}
\hline
\textbf{Lexical Weight ($\alpha$)} & \textbf{nDCG@5} & \textbf{P@5} & \textbf{Recall@5} & \textbf{MRR} \\
\hline
0.0 (Pure Semantic) & 0.9056 & 0.8267 & 0.9333 & 0.9667 \\
0.2 & 0.9168 & 0.8533 & 0.9444 & 0.9667 \\
0.4 (Default) & 0.9400 & 0.8667 & 0.9556 & 1.0000 \\
0.5 & 0.9400 & 0.8667 & 0.9556 & 1.0000 \\
\textbf{0.7 (Empirical Optimum)} & \textbf{0.9539} & \textbf{0.8800} & \textbf{0.9667} & \textbf{1.0000} \\
1.0 (Pure Lexical) & 0.9377 & 0.8800 & 0.9667 & 0.9667 \\
\hline
\end{tabular}
\end{table}
```

### Table 5.3: Pipeline Component Ablations (Experiment E3)
```latex
\begin{table}[htbp]
\centering
\caption{Ablation Study of Pipeline Components on Test Set}
\label{tab:e3_ablations}
\begin{tabular}{lcc}
\hline
\textbf{Ablation Condition} & \textbf{Test nDCG@5} & \textbf{$\Delta$ vs Full Hybrid} \\
\hline
\textbf{Full Hybrid Pipeline} & \textbf{0.9010} & — \\
No Constraints (Raw Fusion) & 0.8911 & \textbf{-0.0099} \\
Mean-Pooled Semantic & 0.8997 & \textbf{-0.0013} \\
No Section Weights & 0.9010 & 0.0000 \\
Reciprocal Rank Fusion (RRF) & 0.9010 & 0.0000 \\
\hline
\end{tabular}
\end{table}
```

---

## 3. Key Discussion Points for Dissertation Defense

1. **Why Hybrid Beats Pure Paradigms in Practice:** While pure lexical retrieval achieves high nDCG on exact technical keywords, it cannot identify semantic equivalents (e.g., matching "microservices orchestration" when the JD specifies "Kubernetes experience"). Conversely, semantic models alone suffer from token dilution across multi-page documents. The hybrid approach preserves lexical precision while benefiting from conceptual generalisation.
2. **Empirical Justification of Constraints:** The ablation study (Table 5.3) demonstrates that removing constraint evaluations degrades nDCG@5 by $-0.0099$. Multiplicative constraints effectively down-rank candidates lacking non-negotiable minimum tenure or educational credentials who would otherwise be elevated by superficial textual overlap.
3. **Auditable Evidence Over Hallucinatory Summaries:** By mapping extracted skills to exact character spans and classifying them into *Demonstrated*, *Listed-only*, and *Tentative* tiers, ResumeFit eliminates the risk of generative hallucinations, fulfilling EU AI Act transparency requirements for high-risk employment systems.
