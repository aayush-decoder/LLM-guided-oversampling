"""FIX 2 -- the tau-bridge for condition_yx's conditioning values.

    p_tau(v)  proportional to  p(v | y_min)^(1 - tau) * p(v)^tau,   tau in [0, 1]

    tau = 1   EXACTLY v1's pooled reading (pooled empirical distribution)
    tau = 0   EXACTLY the minority's own empirical distribution
    between   a tunable boundary tilt (what ADASYN / Borderline-SMOTE do on
              purpose and v1 does by accident with the strength fixed at 1)

HOW EACH ENDPOINT IS MADE EXACT (palt2's tau=0 was not)
    categorical  the bridge is computed on category frequencies; the endpoints
                 use the raw empirical frequencies, interior tau uses Jeffreys
                 (+0.5) smoothing so a category unseen in the tiny minority
                 sample is down-weighted rather than impossible.
    continuous   tau = 1 draws a pooled row value, tau = 0 draws a minority
                 row value; interior tau importance-re-weights the POOLED sample
                 by (p_min / p)^(1 - tau), both densities by 1-D Gaussian KDE.
                 palt2 drew minority-weighted histogram BINS but then a POOLED
                 value inside the bin, so tau = 0 still carried the majority's
                 within-bin shape -- and its KL diagnostic compared histogram to
                 histogram, reporting 0 exactly when that defect was present.

DIAGNOSTIC: KL(bridge || minority) and KL(bridge || pooled) are MEASURED from
actual draws on common quantile bins, so they report what is sampled, not the
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.data.protocol import Schema, Splits
from ...core.serialize import clause, label_head, prompt_from_clauses


def _kde_logpdf(train: np.ndarray, at: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde
    if len(np.unique(train)) < 2:
        return np.where(np.isclose(at, train[0]), 0.0, -30.0)
    return gaussian_kde(train).logpdf(at)


class TauBridge:
    def __init__(self, schema: Schema, minority: pd.DataFrame, pooled: pd.DataFrame,
                 tau: float = 0.0):
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.schema, self.tau = schema, tau
        self.minority, self.pooled = minority, pooled
        self._support: dict[str, np.ndarray] = {}
        self._cum: dict[str, np.ndarray] = {}
        for c in schema.categorical:
            cats = np.array(sorted(set(minority[c].astype(str)) | set(pooled[c].astype(str))))
            cm = minority[c].astype(str).value_counts().reindex(cats).fillna(0).to_numpy(float)
            cp = pooled[c].astype(str).value_counts().reindex(cats).fillna(0).to_numpy(float)
            if tau in (0.0, 1.0):
                w = cm if tau == 0.0 else cp
            else:
                pm = (cm + .5) / (cm + .5).sum()
                pp = (cp + .5) / (cp + .5).sum()
                w = pm ** (1 - tau) * pp ** tau
            self._set(c, cats, w)
        for c in schema.continuous:
            mv = minority[c].astype(float).to_numpy()
            pv = pooled[c].astype(float).to_numpy()
            if tau == 0.0:
                self._set(c, mv, np.ones(len(mv)))
            elif tau == 1.0:
                self._set(c, pv, np.ones(len(pv)))
            else:
                logw = (1 - tau) * (_kde_logpdf(mv, pv) - _kde_logpdf(pv, pv))
                self._set(c, pv, np.exp(logw - logw.max()))

    def _set(self, c, support, w):
        w = np.asarray(w, float)
        self._support[c] = np.asarray(support)
        self._cum[c] = np.cumsum(w / w.sum())

    def sample(self, col: str, rng: np.random.Generator):
        k = int(np.searchsorted(self._cum[col], rng.random(), side="right"))
        v = self._support[col][min(k, len(self._support[col]) - 1)]
        return str(v) if col in self.schema.categorical else float(v)

    def draws(self, col: str, n: int, rng: np.random.Generator) -> np.ndarray:
        k = np.searchsorted(self._cum[col], rng.random(n), side="right")
        return self._support[col][np.minimum(k, len(self._support[col]) - 1)]

    def measured_kl(self, n: int = 4000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        out = {}
        for c in self.schema.features:
            d = self.draws(c, n, rng)
            if c in self.schema.categorical:
                cats = sorted(set(self._support[c]))
                h = lambda x: pd.Series(np.asarray(x).astype(str)).value_counts()\
                    .reindex(cats).fillna(0).to_numpy(float)  # noqa: E731
            else:
                ref = np.concatenate([self.minority[c].astype(float), self.pooled[c].astype(float)])
                e = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
                e[0], e[-1] = -np.inf, np.inf
                h = lambda x, e=e: np.histogram(np.asarray(x, float), bins=e)[0].astype(float)  # noqa: E731
            pb = (h(d) + .5) / (h(d) + .5).sum()
            pm = (h(self.minority[c]) + .5) / (h(self.minority[c]) + .5).sum()
            pp = (h(self.pooled[c]) + .5) / (h(self.pooled[c]) + .5).sum()
            out[c] = {"kl_to_minority": float((pb * np.log(pb / pm)).sum()),
                      "kl_to_pooled": float((pb * np.log(pb / pp)).sum())}
        return {"tau": self.tau,
                "mean_kl_to_minority": float(np.mean([v["kl_to_minority"] for v in out.values()])),
                "mean_kl_to_pooled": float(np.mean([v["kl_to_pooled"] for v in out.values()])),
                "per_feature": out}


def build_prompts(n: int, splits: Splits, rng: np.random.Generator, *, bridge: TauBridge,
                  n_clauses: int = 1, sig: int = 5) -> tuple[list[str], list]:
    s = splits.schema
    head = label_head(s)
    feats = s.features
    k = max(1, min(n_clauses, len(feats)))
    prompts, conds = [], []
    for _ in range(n):
        cols = [feats[i] for i in rng.choice(len(feats), size=k, replace=False)]
        vals = [bridge.sample(c, rng) for c in cols]
        prompts.append(prompt_from_clauses([head, *(clause(c, v, s, sig)
                                                    for c, v in zip(cols, vals))]))
        conds.append((cols[0], vals[0]))
    return prompts, conds
