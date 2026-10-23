"""Post-hoc curation of an over-generated pool (v2 novelty N5). No LLM involved.

1. Privacy gate (DCR): tau = alpha-quantile of DCR(real holdout -> real reference). Pool rows closer
   than tau to a real training row are treated as near-copies and dropped (at most max_drop of the pool).
2. Discriminator-guided sampling-importance-resampling (SIR): a cross-fitted classifier D separates
   real (1) from synthetic (0). By the density-ratio trick
        p_real(x) / p_syn(x)  ~=  (n_syn / n_real) * D(x) / (1 - D(x))
   Weights w = clip(ratio, 1/c, c)^beta; n rows are drawn without replacement with prob ~ w, which
   moves the synthetic distribution towards the real one (beta=0 disables, beta=1 is exact SIR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from predLLM.common.tabgen.eval.fidelity import dcr


def privacy_gate(real: np.ndarray, pool: np.ndarray, alpha: float, max_drop: float, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real))
    n_hold = max(2, int(0.1 * len(real)))
    ref, hold = real[idx[n_hold:]], real[idx[:n_hold]]
    tau = float(np.quantile(dcr(ref, hold), alpha))
    d = dcr(ref, pool)
    keep = d >= tau
    if tau <= 0 or keep.mean() < 1 - max_drop:           # purely discrete data, or gate too aggressive
        keep = d >= np.quantile(d, max_drop) if tau > 0 else np.ones(len(pool), bool)
    return keep, {"dcr_tau": round(tau, 5), "privacy_dropped": int((~keep).sum())}


def sir_weights(real: np.ndarray, pool: np.ndarray, beta: float, clip: float, folds: int, seed: int) -> tuple[np.ndarray, dict]:
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


def curate(real_enc: pd.DataFrame, pool_enc: pd.DataFrame, n: int, features: list[str], cfg: dict, seed: int):
    """Returns (selected rows, stats). Distances use features only; the discriminator sees (x, y)."""
    c = cfg["curation"]
    rng = np.random.default_rng(seed)
    stats = {"pool_size": len(pool_enc)}
    keep = np.ones(len(pool_enc), bool)
    if c["privacy_gate"]:
        keep, st = privacy_gate(real_enc[features].to_numpy(float), pool_enc[features].to_numpy(float),
                                c["privacy_alpha"], c["max_drop"], seed)
        stats.update(st)
    pool = pool_enc[keep].reset_index(drop=True)
    cols = list(pool.columns)
    if c["sir"]:
        w, st = sir_weights(real_enc[cols].to_numpy(float), pool[cols].to_numpy(float), c["beta"], c["clip"],
                            c["folds"], seed)
        stats.update(st)
    else:
        w = np.ones(len(pool))
    idx = rng.choice(len(pool), size=n, replace=n > len(pool), p=w / w.sum())
    return pool.iloc[idx].reset_index(drop=True), idx, keep, stats
