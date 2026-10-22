"""Membership inference against the MINORITY training rows (DOMIAS-style).

Why. Utility-guided generators concentrate mass near the class boundary, where the
rarest records live, and rare records are what generators memorise (Carlini et al.
2022) and what attacks expose (Stadler et al. 2022). DCR cannot certify privacy
(Yao et al. 2025, "The DCR delusion"), so every oversampler is also scored with a
density-ratio membership attack (van Breugel et al., AISTATS 2023, DOMIAS):

    score(x) = log p̂_syn(x) - log p̂_ref(x)

Two density estimators are reported:
    mia_auc      (primary) nearest-neighbour densities, p̂ ∝ 1/d^M: the score is
                 log d_ref(x) - log d_syn(x), with d = distance to the nearest row.
                 Catches copies and near-copies at any scale.
    mia_auc_kde  Gaussian KDEs with Scott bandwidths: smoother, misses exact copies
                 when the bandwidth is wide (a tuned-bandwidth DOMIAS would sit between).

Members are the generator's training minority rows D_minor; non-members are held-out
minority rows D*_minor \\ D_minor, which the generator never saw. The attack is trained on nothing, so its AUC is directly

    mia_auc = P(score(member) > score(non-member))      0.5 = no leakage.

The reference density uses the held-out minority rows themselves (leave-one-out
for the scored non-member), i.e. an attacker with a sample from the population.
All rows live in the fixed metric space of core/evaluate/quality.Space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..data.protocol import Splits
from .quality import Space


def _kde_logpdf(ref: np.ndarray, q: np.ndarray, bw: float, exclude_self: bool = False
                ) -> np.ndarray:
    d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    k = -0.5 * d2 / bw ** 2
    if exclude_self:
        np.fill_diagonal(k, -np.inf)
    m = k.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(k - m).sum(1) + 1e-300))


def _bandwidth(X: np.ndarray) -> float:
    n, d = X.shape
    sd = float(np.sqrt(np.mean(X.var(0)))) if n > 1 else 0.1
    return max(1e-3, sd * (4.0 / ((d + 2) * max(n, 2))) ** (1.0 / (d + 4)))


def mia_report(synth: pd.DataFrame, splits: Splits, *, max_rows: int = 2000,
               seed: int = 0) -> dict:
    s = splits.schema
    mem, non = splits.train_minor, splits.minor_heldout
    if len(synth) < 5 or len(mem) < 5 or len(non) < 5:
        return {"mia_auc": float("nan"), "mia_note": "too few rows"}
    rng = np.random.default_rng(seed)
    sp = Space(s).fit(splits.imbalanced_train)
    S = sp.transform(synth[s.features])
    if len(S) > max_rows:
        S = S[rng.choice(len(S), max_rows, replace=False)]
    M = sp.transform(mem[s.features])
    N = sp.transform(non[s.features])
    y = np.r_[np.ones(len(M)), np.zeros(len(N))]
    # nearest-neighbour density ratio (primary)
    from sklearn.neighbors import NearestNeighbors
    nn_s = NearestNeighbors(n_neighbors=1).fit(S)
    nn_r = NearestNeighbors(n_neighbors=2).fit(N)
    d_syn = nn_s.kneighbors(np.vstack([M, N]))[0][:, 0]
    dr = nn_r.kneighbors(np.vstack([M, N]))[0]
    d_ref = np.r_[dr[:len(M), 0], dr[len(M):, 1]]          # leave-one-out for non-members
    delta = 1e-6 + 1e-3 * float(np.median(d_ref))
    auc = float(roc_auc_score(y, np.log(d_ref + delta) - np.log(d_syn + delta)))
    # KDE density ratio (secondary)
    bw, ref_bw = _bandwidth(S), _bandwidth(N)
    sc_m = _kde_logpdf(S, M, bw) - _kde_logpdf(N, M, ref_bw)
    sc_n = _kde_logpdf(S, N, bw) - _kde_logpdf(N, N, ref_bw, exclude_self=True)
    auc_kde = float(roc_auc_score(y, np.r_[sc_m, sc_n]))
    return {"mia_auc": auc, "mia_advantage": 2 * abs(auc - 0.5), "mia_auc_kde": auc_kde}
