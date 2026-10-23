### Table 1: Retrieval Model Comparison on Held-Out Test Set (35 JDs, 350 Pairs)

| Retrieval Architecture | P@1 | P@3 | R@3 | MRR | nDCG@3 | nDCG@5 |
|:-----------------------|:---:|:---:|:---:|:---:|:------:|:------:|
| **Lexical (TF-IDF)** | 0.714 | 0.457 | 0.879 | 0.743 | 0.889 | 0.919 |
| **Semantic (MiniLM)** | 0.657 | 0.476 | 0.910 | 0.714 | 0.854 | 0.884 |
| **Hybrid Fusion (α=0.4, β=0.6)** | 0.657 | 0.486 | 0.924 | 0.714 | 0.885 | 0.901 |

- **Hybrid vs Lexical**: Diff=-0.018, 95% CI=[-0.057, 0.030], p=0.4230 (Significant: False)
- **Hybrid vs Semantic**: Diff=+0.016, 95% CI=[-0.012, 0.043], p=0.2370 (Significant: False)