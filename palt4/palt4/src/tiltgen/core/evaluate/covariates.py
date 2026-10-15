"""Per-dataset covariates, so that dataset-conditional predictions can be tested.

docs/what_corrected.md predicts that each v2 flag helps on a KIND of dataset:
kNN geometry on multimodal minorities (B1), tau < 1 at high imbalance (B2), anchor
categoricals when there are many categoricals (B3). Without a measured covariate
per dataset those predictions cannot be checked (docs/RESEARCH_REVIEW.md, G8).
Every run stores these under `cov_*` in metrics.json; `tilt report` correlates the
per-dataset gain over the anchor with each of them (Spearman, over datasets).

    cov_imbalance_ratio     |largest class| / |D_minor| after the q-reduction
    cov_n_minor             |D_minor|
    cov_m_continuous        number of continuous features
    cov_m_categorical       number of categorical features
    cov_cat_share           m_categorical / M
    cov_minority_modes      BIC-selected number of Gaussian components (1..4) of the
                            z-scored continuous minority rows (NaN if < 20 rows or
                            no continuous feature)
    cov_class_overlap       fraction of minority rows whose 5 nearest neighbours in
                            D_train (z-scored continuous + one-hot/sqrt2) are mostly
                            majority rows: a standard "hardness" measure
"""

from __future__ import annotations

import numpy as np

from ..data.protocol import Splits


def _minority_modes(splits: Splits, seed: int) -> float:
    s = splits.schema
    if not s.continuous or len(splits.train_minor) < 20:
        return float("nan")
    from sklearn.mixture import GaussianMixture
    X = splits.train_minor[s.continuous].astype(float).to_numpy()
    sd = X.std(0)
    X = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
    best, best_k = np.inf, 1
    for k in range(1, min(4, len(X) // 10) + 1):
        gm = GaussianMixture(k, covariance_type="diag", random_state=seed, reg_covar=1e-3)
        gm.fit(X)
        b = gm.bic(X)
        if b < best:
            best, best_k = b, k
    return float(best_k)


def _overlap(splits: Splits) -> float:
    from sklearn.neighbors import NearestNeighbors
    s = splits.schema
    D = splits.imbalanced_train
    blocks = []
    if s.continuous:
        X = D[s.continuous].astype(float).to_numpy()
        sd = X.std(0)
        blocks.append((X - X.mean(0)) / np.where(sd > 0, sd, 1.0))
    for c in s.categorical:
        v = D[c].astype(str).to_numpy()
        u = np.unique(v)
        blocks.append((v[:, None] == u[None, :]).astype(float) / np.sqrt(2))
    if not blocks:
        return float("nan")
    X = np.hstack(blocks)
    is_min = (D[s.target].astype(str) == s.minority_label).to_numpy()
    if is_min.sum() < 2:
        return float("nan")
    k = min(6, len(X))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, idx = nn.kneighbors(X[is_min])
    frac_major = (~is_min[idx[:, 1:]]).mean(1)            # drop self
    return float((frac_major > 0.5).mean())


def compute(splits: Splits, seed: int = 0) -> dict:
    s = splits.schema
    m = len(s.features)
    return {"cov_imbalance_ratio": float(splits.imbalance_ratio),
            "cov_n_minor": int(len(splits.train_minor)),
            "cov_m_continuous": int(s.m_con), "cov_m_categorical": int(s.m_cat),
            "cov_cat_share": float(s.m_cat / m) if m else float("nan"),
            "cov_minority_modes": _minority_modes(splits, seed),
            "cov_class_overlap": _overlap(splits)}
