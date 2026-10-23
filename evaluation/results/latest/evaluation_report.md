# ResumeFit Benchmark Evaluation Report

Generated: 2026-10-23T16:44:46.777795+00:00

### Table 1: Retrieval Model Comparison on Held-Out Test Set (35 JDs, 350 Pairs)

| Retrieval Architecture | P@1 | P@3 | R@3 | MRR | nDCG@3 | nDCG@5 |
|:-----------------------|:---:|:---:|:---:|:---:|:------:|:------:|
| **Lexical (TF-IDF)** | 0.714 | 0.457 | 0.879 | 0.743 | 0.889 | 0.919 |
| **Semantic (MiniLM)** | 0.657 | 0.476 | 0.910 | 0.714 | 0.854 | 0.884 |
| **Hybrid Fusion (α=0.4, β=0.6)** | 0.657 | 0.486 | 0.924 | 0.714 | 0.885 | 0.901 |

- **Hybrid vs Lexical**: Diff=-0.018, 95% CI=[-0.057, 0.030], p=0.4230 (Significant: False)
- **Hybrid vs Semantic**: Diff=+0.016, 95% CI=[-0.012, 0.043], p=0.2370 (Significant: False)

---

### Table 2: Linear Fusion Weight Parameter Sweep (α·Lexical + (1-α)·Semantic) on Dev Set

| α (Lexical Weight) | β (Semantic Weight) | Dev Set nDCG@5 | Note |
|:------------------:|:-------------------:|:--------------:|:-----|
| 0.0 | 1.0 | 0.8751 | Sub-optimal |
| 0.1 | 0.9 | 0.9075 | Sub-optimal |
| 0.2 | 0.8 | 0.9415 | Sub-optimal |
| 0.3 | 0.7 | 0.9415 | Sub-optimal |
| 0.4 | 0.6 | 0.9419 | Sub-optimal |
| 0.5 | 0.5 | 0.9419 | Sub-optimal |
| 0.6 | 0.4 | 0.9431 | Sub-optimal |
| 0.7 | 0.3 | 0.9539 | **Optimal (Chosen)** |
| 0.8 | 0.2 | 0.9526 | Sub-optimal |
| 0.9 | 0.1 | 0.9281 | Sub-optimal |
| 1.0 | 0.0 | 0.9247 | Sub-optimal |

---

### Table 3: Component Ablation Analysis on Held-Out Test Set

| System Variant / Ablation | nDCG@5 | P@3 | Δ nDCG@5 |
|:--------------------------|:------:|:---:|:--------:|
| **Full System (Baseline)** | 0.9007 | 0.4857 | — (Ref) |
| **Ablation: No Constraints (Raw Fusion)** | 0.8908 | 0.4190 | -0.0099 |
| **Ablation: Mean-Pooled Only (No Req-to-Chunk)** | 0.9499 | 0.5048 | +0.0492 |

---

### Robustness (E4) & Fairness (E5)
- **Keyword Stuffed Penalty**: Candidate score reduced by 0.1464
- **Demographic Invariance (E5)**: Max score difference across identities = 0.046850 (Variance = 2.32e-04, Invariant: False)
