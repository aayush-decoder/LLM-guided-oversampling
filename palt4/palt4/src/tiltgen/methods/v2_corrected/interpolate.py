"""FIX 1 -- interpolation geometry (and FIX 3 -- categoricals from the ANCHOR).

    random                   Eq. (5) verbatim (delegates to v1)     Cov -> (2/3) Sigma
    knn                      partner = one of the k nearest minority rows,
                             searched in z-SCORED space (SMOTE geometry)
    variance_preserving      random pair, rescaled about the global mean: Cov = Sigma
                             exactly, but pushes mode-straddlers further off-manifold
    rose                     smoothed bootstrap (Menardi & Torelli 2014): Cov=(1+h^2)Sigma
    knn_variance_preserving  kNN pair rescaled about the pair midpoint (negative
                             result: indistinguishable from knn; kept to reproduce it)

palt2 searched neighbours on RAW values, so "nearest" meant nearest on the
largest-unit column. Here the search is standardised; the interpolation itself
happens in raw units.

CATEGORICALS (D3). v1 drops them from D_inter, so >=94% of the fine-tuning
corpus has no categorical at all. v2 gives each interpolated row the
categoricals of its ANCHOR x_i (the row it was interpolated from). palt2 used
random donor rows, which paired every continuous block with an unrelated
row's categoricals and destroyed the joint structure it meant to restore.

ORDINALS (flag `ordinal_interpolation`, core/data/kinds.py). When `ordinal` is given,
each ordered categorical joins the numeric block as its RANK: the neighbour search,
the partner and eps are shared with the continuous columns, and the interpolated
rank is rounded back to a level,  r' = r_i + eps (r_j - r_i),  level' = levels[round(r')].
A dataset with no continuous column but declared ordinals (car) thereby gets an
interpolated corpus at all. `random` (Eq. 5, delegated to v1) never touches ordinals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ...core.data import kinds
from ...core.data.protocol import Schema
from ..v1_paper import interpolate as v1i

MODES = ("random", "knn", "variance_preserving", "rose", "knn_variance_preserving")


def _knn_pairs(Z: np.ndarray, n: int, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    m = len(Z)
    i = rng.integers(0, m, size=n)
    if m <= 1:
        return i, i
    kk = max(1, min(k, m - 1))
    idx = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z)[1]
    return i, idx[i, rng.integers(1, kk + 1, size=n)]


def interpolate(minor: pd.DataFrame, schema: Schema, n: int, rng: np.random.Generator, *,
                mode: str = "knn", k: int = 5, rose_bandwidth: float | None = None,
                ordinal: dict | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if mode not in MODES:
        raise ValueError(f"unknown interpolation mode {mode!r}; have {MODES}")
    if mode == "random":
        return v1i.interpolate(minor, schema, n, rng)
    ordinal = ordinal or {}
    con = list(schema.continuous)
    cols = con + list(ordinal)                       # ordinals enter as ranks
    if not cols or len(minor) == 0 or n <= 0:
        z = np.zeros(0, int)
        return pd.DataFrame(columns=cols), z, z
    X = np.column_stack([minor[c].astype(float).to_numpy() for c in con]
                        + [kinds.to_rank(minor[c], lv) for c, lv in ordinal.items()])
    m, d = X.shape
    sd = X.std(0, ddof=1) if m > 1 else np.ones(d)
    sd = np.where(sd > 0, sd, 1.0)

    if mode == "rose":
        h = rose_bandwidth if rose_bandwidth is not None else \
            (4.0 / ((d + 2) * max(m, 2))) ** (1.0 / (d + 4))
        i = rng.integers(0, m, size=n)
        return _frame(X[i] + rng.normal(size=(n, d)) * h * sd, con, ordinal), i, i

    if mode in ("knn", "knn_variance_preserving"):
        i, j = _knn_pairs((X - X.mean(0)) / sd, n, k, rng)
    else:
        i = rng.integers(0, m, size=n)
        j = (i + rng.integers(1, m, size=n)) % m if m > 1 else i
    eps = rng.random((n, 1))
    if mode == "knn":
        Z = X[i] + eps * (X[j] - X[i])
    else:
        centre = X.mean(0, keepdims=True) if mode == "variance_preserving" else \
            0.5 * (X[i] + X[j])
        a, b = 1 - eps, eps
        Z = centre + (a * (X[i] - centre) + b * (X[j] - centre)) / np.sqrt(a ** 2 + b ** 2)
    return _frame(Z, con, ordinal), i, j


def _frame(Z: np.ndarray, con: list[str], ordinal: dict) -> pd.DataFrame:
    """Numeric block -> frame: continuous columns as is, ordinal ranks back to levels."""
    out = pd.DataFrame(Z[:, :len(con)], columns=con)
    for k, (c, lv) in enumerate(ordinal.items()):
        out[c] = kinds.from_rank(Z[:, len(con) + k], lv)
    return out


def with_anchor_categoricals(inter: pd.DataFrame, minor: pd.DataFrame, anchors: np.ndarray,
                             schema: Schema) -> pd.DataFrame:
    out = inter.copy()
    for c in schema.categorical:
        if c in out.columns:                 # an ordinal already interpolated on its rank
            continue
        out[c] = minor[c].astype(str).to_numpy()[anchors]
    return out


def off_manifold_fraction(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str],
                          k: int = 5) -> float:
    """Share of synthetic points farther from the real data than the 99th
    percentile of real-to-real k-NN distance (z-scored space)."""
    cols = [c for c in cols if c in synth.columns]
    if not cols or len(real) < k + 2 or len(synth) == 0:
        return float("nan")
    A = real[cols].astype(float).to_numpy()
    B = synth[cols].astype(float).to_numpy()
    mu, sd = A.mean(0), A.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    A, B = (A - mu) / sd, (B - mu) / sd
    nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
    thr = float(np.quantile(nn.kneighbors(A)[0][:, k], 0.99))
    return float((nn.kneighbors(B, n_neighbors=1)[0].ravel() > thr).mean())
