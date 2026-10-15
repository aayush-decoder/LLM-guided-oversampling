"""Quality, coverage and memorisation of a synthetic minority set.

ONE FIXED DISTANCE FOR EVERY METHOD
    The metric space is fitted on REAL training rows only (never on the
    synthetic set being scored):
        continuous   min-max scaled with the real training range
        categorical  one-hot / sqrt(2)      (a mismatch adds exactly 1 to d^2)
        d(x, x')     Euclidean / sqrt(M)    (in-range rows: d in [0, 1])
    so alpha = 0.2 means the same thing for every method and every dataset.

    palt2 divided each method's distances by THAT METHOD'S OWN maximum, so a
    uniformly closer generator could score lower (counterexample in
    tests/test_eval.py), and fitted the scaler on real + synthetic rows, so a
    generator's own outliers changed its metric space.

PAPER METRICS (Sec. IV-A2), against two references
    paper_*    reference D*_minor           -- what the paper computes; 20% of it
                                               IS the generator's training data
    heldout_*  reference D*_minor \\ D_minor -- genuinely unseen minority rows
    close_probability  mean 1[ min_j d(x_i, x^_j) <= alpha ],   alpha = 0.2
    coverage           Naeem et al. 2020, k = 2

MEMORISATION (Close probability cannot tell fidelity from copying)
    dup_minor / dup_major   fraction of synthetic rows that exactly equal a real
                            minority / majority TRAINING row
    nn_ratio                mean d(x^ -> D_minor) / mean d(held-out -> D_minor):
                            ~0 copying, ~1 as far as genuinely new rows
DCR (paper Fig. 7)          quantiles of d(held-out x -> nearest x^)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..data.protocol import Schema, Splits

ALPHA = 0.2
K_COV = 2


class Space:
    def __init__(self, schema: Schema):
        self.schema = schema

    def fit(self, real: pd.DataFrame) -> "Space":
        s = self.schema
        x = real[s.continuous].astype(float)
        self.lo = x.min().to_numpy() if s.continuous else np.zeros(0)
        rng = (x.max() - x.min()).to_numpy() if s.continuous else np.zeros(0)
        self.rng = np.where(rng > 0, rng, 1.0)
        self.cats = {c: sorted(real[c].astype(str).unique()) for c in s.categorical}
        self.m = max(len(s.features), 1)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        s = self.schema
        parts = []
        if s.continuous:
            parts.append((df[s.continuous].astype(float).to_numpy() - self.lo) / self.rng)
        for c in s.categorical:
            v = df[c].astype(str).to_numpy()
            parts.append((v[:, None] == np.array(self.cats[c])[None, :]) / np.sqrt(2))
        X = np.hstack(parts) if parts else np.zeros((len(df), 1))
        return X / np.sqrt(self.m)


def nn_dist(ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    if len(ref) == 0 or len(query) == 0:
        return np.array([])
    return NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(query)[0].ravel()


def close_probability(real: np.ndarray, synth: np.ndarray, alpha: float = ALPHA) -> float:
    d = nn_dist(synth, real)
    return float((d <= alpha).mean()) if len(d) else float("nan")


def coverage(real: np.ndarray, synth: np.ndarray, k: int = K_COV) -> float:
    if len(real) <= k or len(synth) == 0:
        return float("nan")
    radius = NearestNeighbors(n_neighbors=k + 1).fit(real).kneighbors(real)[0][:, k]
    return float((nn_dist(synth, real) <= radius).mean())


def _keys(df: pd.DataFrame, schema: Schema) -> set:
    d = df[schema.features].copy()
    for c in schema.continuous:
        d[c] = d[c].astype(float).round(6)
    for c in schema.categorical:
        d[c] = d[c].astype(str)
    return set(map(tuple, d.to_numpy().tolist()))


def dup_rate(synth: pd.DataFrame, ref: pd.DataFrame, schema: Schema) -> float:
    if len(synth) == 0:
        return float("nan")
    seen = _keys(ref, schema)
    d = synth[schema.features].copy()
    for c in schema.continuous:
        d[c] = d[c].astype(float).round(6)
    for c in schema.categorical:
        d[c] = d[c].astype(str)
    return float(np.mean([tuple(r) in seen for r in d.to_numpy().tolist()]))


def quality_report(synth: pd.DataFrame, splits: Splits, *, alpha: float = ALPHA,
                   k: int = K_COV) -> dict:
    s = splits.schema
    if len(synth) == 0:
        return {"error": "empty synthetic set", "n_synthetic": 0}
    space = Space(s).fit(splits.imbalanced_train)
    E = space.transform
    syn, minor = E(synth), E(splits.train_minor)
    out: dict = {"n_synthetic": int(len(synth)), "alpha": alpha, "k": k}
    for tag, ref_df in (("paper", splits.train_minor_full), ("heldout", splits.minor_heldout)):
        ref = E(ref_df)
        out[f"{tag}_close_probability"] = close_probability(ref, syn, alpha)
        out[f"{tag}_coverage"] = coverage(ref, syn, k)
    held = E(splits.minor_heldout)
    d = nn_dist(syn, held)
    if len(d):
        out.update({f"dcr_q{q}": float(np.quantile(d, q / 100)) for q in (10, 50, 90)})
    ds, dr = nn_dist(minor, syn), nn_dist(minor, held)
    out["nn_ratio"] = float(ds.mean() / dr.mean()) if len(dr) and dr.mean() > 0 else float("nan")
    out["dup_minor"] = dup_rate(synth, splits.train_minor, s)
    out["dup_major"] = dup_rate(synth, splits.train_major, s)
    if s.continuous and len(splits.minor_heldout) > 2 and len(synth) > 2:
        sd = splits.train_minor_full[s.continuous].astype(float).std().replace(0, 1).to_numpy()
        a = splits.train_minor_full[s.continuous].astype(float).to_numpy() / sd
        b = synth[s.continuous].astype(float).to_numpy() / sd
        out["variance_ratio"] = float(np.trace(np.atleast_2d(np.cov(b, rowvar=False)))
                                      / np.trace(np.atleast_2d(np.cov(a, rowvar=False))))
    # paper-comparable aliases
    out["close_probability"] = out["paper_close_probability"]
    out["coverage"] = out["paper_coverage"]
    return out


def dcr_values(synth: pd.DataFrame, splits: Splits) -> np.ndarray:
    """Per-row DCR of held-out minority rows (for the Fig. 7 histogram)."""
    space = Space(splits.schema).fit(splits.imbalanced_train)
    return nn_dist(space.transform(synth), space.transform(splits.minor_heldout))
