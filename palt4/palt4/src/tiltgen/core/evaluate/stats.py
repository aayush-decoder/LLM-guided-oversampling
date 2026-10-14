"""Significance testing across datasets (Demsar, JMLR 2006) -- done carefully.

    friedman_nemenyi(table)   is ANY method different; which are within CD of the best
    paired(a, b)              Wilcoxon signed-rank + sign test + bootstrap CI
    holm(pvalues)             Holm step-down correction for several paired tests

FIXES vs palt2
  * q_alpha comes from scipy's studentized-range distribution for ANY k (palt2
    had a table up to k=15 and crashed on a full ablation report; it also used
    the alpha=0.10 table for every alpha != 0.05)
  * a method that FAILED on a dataset (no metric) gets the WORST rank there.
    palt2 dropped the dataset for every method, which rewards failing.
  * reports are per experiment family (see experiments/report.py); mixing the
    `original` upper bound or unrelated sweeps into one Friedman test is not
    meaningful and is no longer done.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata, wilcoxon

# Demsar 2006 Table 5(a), alpha = 0.05, used to cross-check the scipy value.
DEMSAR_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
              9: 3.102, 10: 3.164}


def q_alpha(k: int, alpha: float = 0.05) -> float:
    """Studentised range quantile / sqrt(2), infinite degrees of freedom."""
    from scipy.stats import studentized_range
    return float(studentized_range.ppf(1 - alpha, k, 1e6) / np.sqrt(2))


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    return q_alpha(k, alpha) * float(np.sqrt(k * (k + 1) / (6.0 * n)))


def mean_ranks(table: pd.DataFrame) -> pd.Series:
    """rows = datasets, cols = methods, higher is better; NaN = failed -> worst rank."""
    filled = table.astype(float).fillna(-np.inf)
    ranks = np.vstack([rankdata(-filled.iloc[i].to_numpy()) for i in range(len(filled))])
    return pd.Series(ranks.mean(0), index=table.columns)


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05) -> dict:
    table = table.dropna(axis=0, how="all")
    n, k = table.shape
    if n < 3 or k < 3:
        return {"error": f"need >=3 datasets and >=3 methods, got {n}x{k}"}
    filled = table.astype(float).fillna(-1e9)
    stat, p = friedmanchisquare(*[filled[c].to_numpy() for c in filled.columns])
    mr = mean_ranks(table)
    cd = critical_difference(k, n, alpha)
    best = mr.idxmin()
    return {
        "n_datasets": n, "n_methods": k, "friedman_chi2": float(stat), "friedman_p": float(p),
        "significant": bool(p < alpha), "mean_ranks": mr.round(3).to_dict(),
        "critical_difference": round(cd, 3), "best": best,
        "not_separable_from_best": sorted(m for m, r in mr.items() if r - mr[best] < cd),
        "failures": {c: int(table[c].isna().sum()) for c in table.columns
                     if table[c].isna().any()},
    }


def paired(a: dict, b: dict, *, n_boot: int = 5000, seed: int = 0) -> dict:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return {"error": f"only {len(common)} shared datasets"}
    x = np.array([a[d] for d in common], float)
    y = np.array([b[d] for d in common], float)
    d = x - y
    try:
        pw = float(wilcoxon(x, y).pvalue)
    except ValueError:
        pw = 1.0
    wins = int((d > 0).sum())
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return {"n": len(common), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "wins": wins, "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "wilcoxon_p": pw, "sign_p": float(binomtest(wins, len(common), .5).pvalue),
            "min_attainable_p": float(2 * 0.5 ** len(common))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def cd_text(res: dict, width: int = 60) -> str:
    if "mean_ranks" not in res:
        return res.get("error", "no result")
    mr, cd = res["mean_ranks"], res["critical_difference"]
    k = res["n_methods"]
    lines = [f"Friedman chi2={res['friedman_chi2']:.3f} p={res['friedman_p']:.2g}  "
             f"CD={cd:.3f} (k={k}, N={res['n_datasets']})", ""]
    best = min(mr.values())
    for m_, r in sorted(mr.items(), key=lambda kv: kv[1]):
        bar = [" "] * width
        bar[int(round((r - 1) / max(k - 1, 1) * (width - 1)))] = "@"
        tag = "" if r - best < cd else "   <- separable from best"
        lines.append(f"  {''.join(bar)} {r:5.2f}  {m_}{tag}")
    if res.get("failures"):
        lines.append(f"  failures (ranked last): {res['failures']}")
    return "\n".join(lines)
