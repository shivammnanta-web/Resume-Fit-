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