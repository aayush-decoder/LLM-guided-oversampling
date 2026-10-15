"""C3b -- interpolated fine-tuning rows.   Paper Sec. III-A3, Eq. (5), verbatim.

    x' = x_i^con + eps * (x_j^con - x_i^con),   eps ~ U[0,1],
    x_j "another RANDOM minority sample",       |D_inter| = r * |D_major|

Faithful consequences (measured, not assumed; see `tilt audit math`):
  (1) Cov(x') = E[(1-eps)^2 + eps^2] Sigma = (2/3) Sigma for independent i, j;
      without replacement from n rows: (2/3 - 1/(3(n-1))) Sigma. For a Gaussian
      that is -0.2027 nats per continuous feature -- the deleted Proposition 3
      claimed an entropy INCREASE.
  (2) on a multimodal minority class, random pairs straddle modes: ~35% of
      interpolants land where no minority row lives.
  (3) only continuous features are interpolated, so interpolated rows carry
      M_con of M features and are (1 + q/rho)^-1 >= ~94% of the corpus.

Returns the anchor/partner indices too, so v2 can reuse the exact pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.data.protocol import Schema


def interpolate(minor: pd.DataFrame, schema: Schema, n: int, rng: np.random.Generator
                ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    con = schema.continuous
    if not con or len(minor) == 0 or n <= 0:
        z = np.zeros(0, int)
        return pd.DataFrame(columns=con), z, z
    X = minor[con].astype(float).to_numpy()
    m = len(X)
    i = rng.integers(0, m, size=n)
    j = (i + rng.integers(1, m, size=n)) % m if m > 1 else i
    eps = rng.random((n, 1))
    return pd.DataFrame(X[i] + eps * (X[j] - X[i]), columns=con), i, j


def theoretical_covariance_ratio(n: int) -> float:
    return 2 / 3 - 1 / (3 * (n - 1)) if n > 1 else float("nan")


def covariance_ratio(real: pd.DataFrame, inter: pd.DataFrame, cols: list[str]) -> float:
    cols = [c for c in cols if c in inter.columns]
    if not cols or len(real) < 2 or len(inter) < 2:
        return float("nan")
    a = np.trace(np.atleast_2d(np.cov(real[cols].astype(float).to_numpy(), rowvar=False)))
    b = np.trace(np.atleast_2d(np.cov(inter[cols].astype(float).to_numpy(), rowvar=False)))
    return float(b / a) if a > 0 else float("nan")
