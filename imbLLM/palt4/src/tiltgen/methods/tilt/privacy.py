"""Memorisation gate for generated minority rows.

Rare minority rows are the ones generators memorise and membership attacks hit
hardest (Carlini et al. 2022; Stadler et al. 2022), and tilting toward boundary
regions amplifies exactly those records (docs/THEORY.md, Prop. 4). Two guards:

1. amplification cap A on λ (methods/tilt/tilt.py);
2. this DCR gate: a generated row closer to a real minority TRAINING row than
   τ = the α-quantile of "genuinely new" distances is dropped. "Genuinely new"
   distances come from an internal split of D_minor itself (20% vs the other
   80%), so the gate never looks at held-out, validation or test rows.

The gate is a guard, not a privacy guarantee: DCR can pass data that a
membership-inference attack still exposes (Yao et al. 2025, "The DCR delusion").
The honest privacy number is the DOMIAS-style attack in core/evaluate/privacy.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.data.protocol import Splits
from ...core.evaluate.quality import Space, nn_dist


def dcr_gate(rows: pd.DataFrame, splits: Splits, *, alpha: float = 0.05,
             seed: int = 0) -> tuple[np.ndarray, dict]:
    s = splits.schema
    minor = splits.train_minor.reset_index(drop=True)
    if len(minor) < 5 or len(rows) == 0:
        return np.ones(len(rows), bool), {"tau": 0.0, "dropped": 0, "note": "too few rows"}
    space = Space(s).fit(splits.imbalanced_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(minor))
    k = max(1, len(minor) // 5)
    hold, ref = minor.iloc[perm[:k]], minor.iloc[perm[k:]]
    tau = float(np.quantile(nn_dist(space.transform(ref), space.transform(hold)), alpha))
    d = nn_dist(space.transform(minor), space.transform(rows[s.features]))
    keep = d >= tau if tau > 0 else d > 0          # tau = 0 (discrete data): drop exact copies
    return keep, {"tau": tau, "alpha": alpha, "dropped": int((~keep).sum()),
                  "dropped_share": float((~keep).mean())}
