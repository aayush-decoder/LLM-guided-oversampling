"""Post-hoc curation with outlier/anomaly gating and SIR (v3 novelty).

Pipeline:
1. Privacy gate (DCR): drops near-copies of real training records.
2. Anomaly gate: uses IsolationForest to drop synthetic samples that collapse into the
   dense majority inlier cluster, keeping candidates that match the target outlier envelope.
3. Discriminator SIR: density-ratio reweighting to match the empirical target distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from tabgen.eval.fidelity import dcr

from .outlier import AnomalyScorer, filter_by_anomaly_score


def privacy_gate(
    real: np.ndarray,
    pool: np.ndarray,
    alpha: float,
    max_drop: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real))
    n_hold = max(2, int(0.1 * len(real)))
    ref, hold = real[idx[n_hold:]], real[idx[:n_hold]]
    tau = float(np.quantile(dcr(ref, hold), alpha))
    d = dcr(ref, pool)
    keep = d >= tau
    if tau <= 0 or keep.mean() < 1 - max_drop:
        keep = d >= np.quantile(d, max_drop) if tau > 0 else np.ones(len(pool), bool)
    return keep, {"dcr_tau": round(tau, 5), "privacy_dropped": int((~keep).sum())}


def sir_weights(
    real: np.ndarray,
    pool: np.ndarray,
    beta: float,
    clip: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    X = np.vstack([real, pool])
    z = np.r_[np.ones(len(real)), np.zeros(len(pool))]
    oof = np.zeros(len(X))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, z):
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, n_jobs=-1)
        oof[te] = clf.fit(X[tr], z[tr]).predict_proba(X[te])[:, 1]
    d = np.clip(oof[len(real):], 1e-4, 1 - 1e-4)
    ratio = (len(pool) / len(real)) * d / (1 - d)
    w = np.clip(ratio, 1 / clip, clip) ** beta
    ess = w.sum() ** 2 / (w ** 2).sum()
    return w, {"sir_ess": round(float(ess), 1), "disc_mean_pool": round(float(d.mean()), 4)}


def curate(
    real_enc: pd.DataFrame,
    pool_enc: pd.DataFrame,
    n: int,
    features: list[str],
    cfg: dict,
    seed: int,
    scorer: AnomalyScorer | None = None,
):
    """Executes multi-stage curation: Privacy gate -> Anomaly filter -> Discriminator SIR."""
    c = cfg.get("curation", {})
    rng = np.random.default_rng(seed)
    stats = {"pool_size": len(pool_enc)}
    keep = np.ones(len(pool_enc), bool)

    # 1. Privacy gate
    if c.get("privacy_gate", True):
        keep_priv, st = privacy_gate(
            real_enc[features].to_numpy(float),
            pool_enc[features].to_numpy(float),
            c.get("privacy_alpha", 0.01),
            c.get("max_drop", 0.5),
            seed,
        )
        keep = keep & keep_priv
        stats.update(st)

    # 2. Anomaly gate (outlier filter)
    if c.get("anomaly_gate", True) and scorer is not None and scorer.fitted:
        pool_sub = pool_enc[keep].reset_index(drop=True)
        keep_anom, st_anom = filter_by_anomaly_score(
            pool_sub,
            features,
            scorer,
            target_quantile_low=c.get("anomaly_quantile_low", 0.0),
            target_quantile_high=c.get("anomaly_quantile_high", 0.90),
        )
        # Map back to full pool boolean mask
        idx_kept = np.where(keep)[0]
        for sub_i, keep_flag in enumerate(keep_anom):
            if not keep_flag:
                keep[idx_kept[sub_i]] = False
        stats.update(st_anom)

    pool = pool_enc[keep].reset_index(drop=True)
    if len(pool) == 0:
        # Fallback to unfiltered pool if over-filtered
        pool = pool_enc.reset_index(drop=True)
        keep = np.ones(len(pool_enc), bool)

    cols = list(pool.columns)
    if c.get("sir", True) and len(pool) > 5:
        w, st = sir_weights(
            real_enc[cols].to_numpy(float),
            pool[cols].to_numpy(float),
            c.get("beta", 1.0),
            c.get("clip", 20.0),
            c.get("folds", 5),
            seed,
        )
        stats.update(st)
    else:
        w = np.ones(len(pool))

    idx = rng.choice(len(pool), size=n, replace=n > len(pool), p=w / w.sum())
    from tabgen.data.preprocess import TARGET
    target_col = TARGET if TARGET in pool.columns else (cols[-1] if len(cols) > len(features) else None)
    if target_col and target_col in pool.columns:
        pool_classes = np.unique(pool[target_col])
        chosen_classes = set(np.unique(pool.iloc[idx][target_col]))
        missing = set(pool_classes) - chosen_classes
        for m in missing:
            candidates = np.where(pool[target_col].to_numpy() == m)[0]
            if len(candidates) > 0:
                replace_pos = int(rng.integers(0, len(idx)))
                idx[replace_pos] = candidates[0]

    return pool.iloc[idx].reset_index(drop=True), idx, keep, stats
