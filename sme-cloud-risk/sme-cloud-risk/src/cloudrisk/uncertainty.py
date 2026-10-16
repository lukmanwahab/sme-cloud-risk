"""
Bootstrap uncertainty for hold-out metrics.

Purpose : a point estimate of ROC-AUC on 750 firms is not enough to claim
          one model beats another. This module resamples the hold-out set
          with replacement, recomputes each metric, and reports percentile
          confidence intervals. It also bootstraps the paired difference in
          AUC between the selected model and each rival, which is the
          statistically correct comparison because both are scored on the
          same resampled firms.
Inputs  : predicted probabilities for one or more models, true labels.
Outputs : DataFrame of metric estimates with 95% CIs; DataFrame of paired
          AUC differences.
Depends : numpy, pandas, scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import SEED

METRICS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "brier": brier_score_loss,
}


def bootstrap_metrics(y: np.ndarray, p: np.ndarray, n_boot: int = 1000,
                      seed: int = SEED, alpha: float = 0.05) -> pd.DataFrame:
    """Percentile bootstrap CIs for each metric on the hold-out set."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    draws = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():       # degenerate resample, skip
            continue
        for m, fn in METRICS.items():
            draws[m].append(fn(y[idx], p[idx]))
    rows = []
    for m, fn in METRICS.items():
        arr = np.asarray(draws[m])
        rows.append({
            "metric": m,
            "estimate": fn(y, p),
            "ci_low": np.quantile(arr, alpha / 2),
            "ci_high": np.quantile(arr, 1 - alpha / 2),
            "n_boot": len(arr),
        })
    return pd.DataFrame(rows)


def paired_auc_difference(y: np.ndarray, p_ref: np.ndarray, others: dict[str, np.ndarray],
                          n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """
    Bootstrap the paired AUC difference (reference minus each rival).

    A CI that excludes zero is evidence that the ranking is not an artefact
    of which 750 firms happened to land in the hold-out set.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_ref = np.asarray(p_ref)
    rows = []
    for name, p_other in others.items():
        p_other = np.asarray(p_other)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() == y[idx].max():
                continue
            diffs.append(roc_auc_score(y[idx], p_ref[idx]) - roc_auc_score(y[idx], p_other[idx]))
        diffs = np.asarray(diffs)
        rows.append({
            "comparison": f"selected minus {name}",
            "auc_difference": roc_auc_score(y, p_ref) - roc_auc_score(y, p_other),
            "ci_low": np.quantile(diffs, 0.025),
            "ci_high": np.quantile(diffs, 0.975),
            "p_share_below_zero": float((diffs < 0).mean()),
        })
    return pd.DataFrame(rows)
