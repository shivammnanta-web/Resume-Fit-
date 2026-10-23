"""Statistical significance testing and bootstrap confidence intervals (Task P6.8).

Supports academic defense requirements:
- Paired bootstrap hypothesis testing (1000+ resamples, 95% CIs)
- Wilcoxon signed-rank test (non-parametric paired differences)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SignificanceResult:
    """Result of statistical hypothesis test comparing two ranking models."""

    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value_bootstrap: float
    wilcoxon_stat: float | None
    wilcoxon_p_value: float | None
    is_significant: bool  # at alpha = 0.05


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> SignificanceResult:
    """Perform paired bootstrap resampling to compare Model A vs Model B.

    Computes empirical difference of means, 95% confidence intervals, and p-value.
    """
    arr_a = np.array(scores_a, dtype=np.float64)
    arr_b = np.array(scores_b, dtype=np.float64)

    if len(arr_a) != len(arr_b) or len(arr_a) == 0:
        raise ValueError("Score arrays must be non-empty and of equal length.")

    n_samples = len(arr_a)
    observed_diff = float(np.mean(arr_a) - np.mean(arr_b))

    rng = np.random.default_rng(random_seed)
    diffs = np.zeros(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=n_samples)
        sample_a = arr_a[indices]
        sample_b = arr_b[indices]
        diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    # 95% Bootstrap Confidence Interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(diffs, lower_pct))
    ci_upper = float(np.percentile(diffs, upper_pct))

    # Empirical two-tailed p-value
    centered_diffs = diffs - observed_diff
    p_val_boot = float(np.mean(np.abs(centered_diffs) >= np.abs(observed_diff)))

    # Wilcoxon signed-rank test
    diff_vector = arr_a - arr_b
    wilc_stat = None
    wilc_p = None
    if not np.all(diff_vector == 0):
        try:
            res = stats.wilcoxon(arr_a, arr_b, alternative="two-sided")
            wilc_stat = float(res.statistic)
            wilc_p = float(res.pvalue)
        except Exception as exc:
            logger.debug("Wilcoxon test failed: %s", exc)

    # When N < 6, Wilcoxon cannot mathematically reach p < 0.05; bootstrap p-value is primary
    is_sig = bool(p_val_boot < alpha or (wilc_p is not None and wilc_p < alpha))

    return SignificanceResult(
        mean_a=float(np.mean(arr_a)),
        mean_b=float(np.mean(arr_b)),
        mean_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value_bootstrap=p_val_boot,
        wilcoxon_stat=wilc_stat,
        wilcoxon_p_value=wilc_p,
        is_significant=is_sig,
    )
