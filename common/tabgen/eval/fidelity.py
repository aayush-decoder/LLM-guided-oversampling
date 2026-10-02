"""Quality / diversity / privacy metrics of Section IV-A2 and IV-C2, all in the encoded space.

  discriminator  (lower better)  XGBoost real-vs-synthetic on a held-out split. score = 2*max(AUC,.5)-1,
                                 so 0 = indistinguishable, 1 = perfectly separable (paper convention).
  inverse_kl     (higher better) 1/M * sum_i 1/(1+KL(p(X_i)||p(^X_i)))  (synthcity-style, 10-bin hist).
  density        (higher better) Naeem et al. 2020, k=2.
  coverage       (higher better) Naeem et al. 2020, k=2.
  dcr            distance of every real TRAINING sample to its closest synthetic sample (Fig. 6).

REFERENCE SETS (docs: palt3/docs/RESEARCH_REVIEW.md, 3.2). Fidelity is measured against
the HELD-OUT test rows, never against the rows the generator was fitted on: against the
training rows a generator that copies them scores perfectly, and v2's curation step (N5)
filters with an XGBoost train-vs-synthetic discriminator, so an XGBoost discriminator on
the training rows would grade v2 with its own filter. Privacy (DCR, exact copies) is,
by definition, measured against the training rows. A second discriminator family
(logistic regression, `discriminator_lr`) guards against a result that is specific to
XGBoost. `discriminator_train` keeps the old, train-referenced number for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier


def _subsample(a: np.ndarray, n: int, rng) -> np.ndarray:
    return a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]


def discriminator_score(real: np.ndarray, fake: np.ndarray, seed: int = 0, kind: str = "xgb") -> dict:
    n = min(len(real), len(fake))
    rng = np.random.default_rng(seed)
    X = np.vstack([_subsample(real, n, rng), _subsample(fake, n, rng)])
    y = np.r_[np.ones(n), np.zeros(n)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = (XGBClassifier(random_state=seed, n_jobs=-1) if kind == "xgb"
           else LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    return {"discriminator": round(2 * max(auc, 0.5) - 1, 4), "discriminator_auc": round(auc, 4),
            "discriminator_acc": round(float(((p > 0.5) == yte).mean()), 4)}


def inverse_kl(real: pd.DataFrame, fake: pd.DataFrame, cols: list[str], discrete: set[str],
               bins: int = 10) -> float:
    scores = []
    for c in cols:
        r, f = real[c].to_numpy(float), fake[c].to_numpy(float)
        if c in discrete or len(np.unique(r)) <= bins:
            support = np.unique(np.r_[r, f])
            pr = np.array([(r == v).sum() for v in support], float)
            pf = np.array([(f == v).sum() for v in support], float)
        else:
            edges = np.histogram_bin_edges(r, bins=bins)
            pr = np.histogram(r, edges)[0].astype(float)
            pf = np.histogram(np.clip(f, edges[0], edges[-1]), edges)[0].astype(float)
        pr, pf = pr / pr.sum() + 1e-8, pf / max(pf.sum(), 1) + 1e-8
        scores.append(1.0 / (1.0 + entropy(pr, pf)))
    return round(float(np.mean(scores)), 4)


def density_coverage(real: np.ndarray, fake: np.ndarray, k: int = 2) -> dict:
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real)
    radii = nn_real.kneighbors(real)[0][:, k]                  # distance to k-th NN (index 0 is itself)
    hits, covered = 0, 0
    for s in range(0, len(real), 1024):                        # chunked: O(1024 x n_fake) memory
        inside = euclidean_distances(real[s:s + 1024], fake) < radii[s:s + 1024, None]
        hits += int(inside.sum())
        covered += int(inside.any(axis=1).sum())
    density = hits / (k * len(fake))
    coverage = covered / len(real)
    return {"density": round(float(density), 4), "coverage": round(float(coverage), 4)}


def dcr(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """For every row of `query`, Euclidean distance to its nearest row in `reference`."""
    return NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(query)[0][:, 0]


def fidelity_report(real: pd.DataFrame, fake: pd.DataFrame, features: list[str], discrete: set[str],
                    seed: int = 0, max_rows: int = 5000, train: pd.DataFrame | None = None) -> dict:
    """`real` = held-out reference for fidelity; `train` = the generator's training rows,
    used for privacy (DCR, exact copies). `train=None` falls back to `real` for both."""
    rng = np.random.default_rng(seed)
    train = real if train is None else train
    R = _subsample(real[features].to_numpy(float), max_rows, rng)
    T = _subsample(train[features].to_numpy(float), max_rows, rng)
    Fk = _subsample(fake[features].to_numpy(float), max_rows, rng)
    out = {"inverse_kl": inverse_kl(real, fake, features, discrete)}
    out.update(discriminator_score(R, Fk, seed))
    out["discriminator_lr"] = discriminator_score(R, Fk, seed, kind="lr")["discriminator"]
    if train is not real:
        out["discriminator_train"] = discriminator_score(T, Fk, seed)["discriminator"]
    out.update(density_coverage(R, Fk, k=2))
    d = dcr(Fk, T)
    out.update({"dcr_mean": round(float(d.mean()), 4), "dcr_median": round(float(np.median(d)), 4),
                "exact_copy_rate": round(float((dcr(T, Fk) < 1e-9).mean()), 4),
                "fidelity_reference": "heldout_test" if train is not real else "same_as_privacy"})
    return out
