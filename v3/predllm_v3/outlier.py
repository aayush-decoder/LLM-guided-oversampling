"""Outlier-aware support bounding and anomaly detection primitives (v3 novelty).

Provides:
1. Outlier/boundary range computation per numeric feature (IQR, minority envelope, extreme quantiles).
2. Anomaly detector (Isolation Forest) for scoring and gating synthetic rows.
3. Outlier filtering to ensure generated samples fall within the target rare-event manifold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def compute_outlier_ranges(
    df: pd.DataFrame,
    num_cols: list[str],
    mode: str = "iqr",
    factor: float = 1.5,
    target_col: str | None = None,
    minority_classes: list | None = None,
) -> dict[str, tuple[float, float]]:
    """Computes the generation range for each numeric column.

    Modes:
      - 'iqr': [Q1 - factor*IQR, Q3 + factor*IQR] clipped to [min, max].
      - 'minority': empirical range [min, max] specifically over minority class rows.
      - 'quantile': [q_0.01, q_0.99] quantile range.
      - 'empirical': global [min, max].
    """
    ranges: dict[str, tuple[float, float]] = {}

    target_df = df
    if mode == "minority" and target_col is not None and minority_classes:
        min_mask = df[target_col].astype(str).isin([str(c) for c in minority_classes])
        if min_mask.sum() >= 2:
            target_df = df[min_mask]

    for col in num_cols:
        vals = pd.to_numeric(target_df[col], errors="coerce").dropna().to_numpy(float)
        all_vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            ranges[col] = (0.0, 1.0)
            continue

        global_min, global_max = float(np.min(all_vals)), float(np.max(all_vals))

        if mode == "iqr":
            q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q75 - q25
            lo = max(global_min, float(q25 - factor * iqr))
            hi = min(global_max, float(q75 + factor * iqr))
            if lo >= hi:
                lo, hi = global_min, global_max
            ranges[col] = (lo, hi)
        elif mode == "minority":
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if lo >= hi:
                lo, hi = global_min, global_max
            ranges[col] = (lo, hi)
        elif mode == "quantile":
            lo = float(np.percentile(vals, 1))
            hi = float(np.percentile(vals, 99))
            if lo >= hi:
                lo, hi = global_min, global_max
            ranges[col] = (lo, hi)
        else:  # empirical
            ranges[col] = (global_min, global_max)

    return ranges


class AnomalyScorer:
    """Isolation Forest based anomaly scorer for tabular candidate verification."""

    def __init__(self, contamination: float = 0.1, seed: int = 42):
        self.contamination = contamination
        self.seed = seed
        self.model = IsolationForest(
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        self.fitted = False
        self.threshold: float = 0.0

    def fit(self, X: np.ndarray | pd.DataFrame) -> "AnomalyScorer":
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(float)
        self.model.fit(X)
        self.fitted = True
        scores = self.model.decision_function(X)
        # Lower score = more anomalous
        self.threshold = float(np.percentile(scores, self.contamination * 100))
        return self

    def score(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(float)
        return self.model.decision_function(X)

    def is_outlier(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.score(X) <= self.threshold


def filter_by_anomaly_score(
    pool: pd.DataFrame,
    features: list[str],
    scorer: AnomalyScorer,
    target_quantile_low: float = 0.0,
    target_quantile_high: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Filters pool candidates to match target anomaly percentile envelope.

    Prevents majority mode collapse by dropping synthetic rows that fall deep into
    the dense majority inlier cluster (high decision score).
    """
    X = pool[features].to_numpy(float)
    scores = scorer.score(X)

    lo_val = float(np.quantile(scores, target_quantile_low))
    hi_val = float(np.quantile(scores, target_quantile_high))

    keep = (scores >= lo_val) & (scores <= hi_val)
    if keep.sum() < max(5, int(0.1 * len(pool))):
        # Fallback if filter is overly restrictive
        keep = np.ones(len(pool), dtype=bool)

    stats = {
        "anomaly_mean": round(float(scores.mean()), 4),
        "anomaly_kept_ratio": round(float(keep.mean()), 4),
        "anomaly_threshold_high": round(hi_val, 4),
    }
    return keep, stats
