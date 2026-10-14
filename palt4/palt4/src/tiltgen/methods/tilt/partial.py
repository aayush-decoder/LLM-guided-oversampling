"""Partial-row utility model: the TWIST of the field-wise SMC sampler.

The LLM writes a row one field at a time, in an order it chooses. After t fields
only a subset S of the features is known. Twisted SMC (Zhao et al., ICML 2024)
needs, at every step, a positive "look-ahead" potential

    ψ_t(x_S) ≈ E[ exp(λ z(x)) | x_S ]          (optimal twist, zero-variance weights)

We use  log ψ_t = λ · ẑ(x_S),  where ẑ is a regressor of the FULL-row standardised
utility z(x) on the observed subset x_S. It is FUDGE (Yang & Klein 2021) adapted to
table fields: trained on randomly MASKED copies of complete rows, so one model
answers for every subset. Missing features are NaN, which gradient-boosted trees
route natively. exp(E[λz]) is not E[exp(λz)] (Jensen), but ANY positive
intermediate twist leaves the SMC target exact; a better twist only lowers the
weight variance (docs/THEORY.md, Prop. 3a). The terminal potential is always the
exact exp(λ z(x_full)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.data.protocol import Schema


class PartialUtility:
    def __init__(self, schema: Schema, *, n_masks: int = 6, seed: int = 0):
        self.s = schema
        self.n_masks = n_masks
        self.seed = seed

    def _encode(self, rows: list[dict] | pd.DataFrame) -> np.ndarray:
        s = self.s
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        cols = []
        for c in s.continuous:
            v = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan,
                                                                                 index=df.index)
            cols.append(v.to_numpy(float))
        for c in s.categorical:
            lut = self.codes[c]
            v = df[c].map(lambda x: lut.get(str(x), np.nan) if pd.notna(x) else np.nan) \
                if c in df else pd.Series(np.nan, index=df.index)
            cols.append(v.to_numpy(float))
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def fit(self, rows: pd.DataFrame, z: np.ndarray) -> "PartialUtility":
        from sklearn.ensemble import HistGradientBoostingRegressor
        s = self.s
        rows = rows[s.features].reset_index(drop=True)
        self.codes = {c: {v: i for i, v in enumerate(sorted(rows[c].astype(str).unique()))}
                      for c in s.categorical}
        X = self._encode(rows)
        rng = np.random.default_rng(self.seed)
        Xs, ys = [X, np.full_like(X, np.nan)], [z, z]
        for _ in range(self.n_masks):
            keep = rng.random(X.shape) < rng.random((len(X), 1))     # per-row keep rate ~ U(0,1)
            Xs.append(np.where(keep, X, np.nan))
            ys.append(z)
        cat_mask = [False] * len(s.continuous) + [True] * len(s.categorical)
        self.model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=self.seed,
            categorical_features=cat_mask if any(cat_mask) else None)
        self.model.fit(np.vstack(Xs), np.concatenate(ys))
        self.mean = float(np.mean(z))
        return self

    def predict(self, partial_rows: list[dict]) -> np.ndarray:
        if not partial_rows:
            return np.zeros(0)
        return self.model.predict(self._encode(partial_rows))
