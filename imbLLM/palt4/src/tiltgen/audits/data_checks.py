"""Data audits: shapes vs the papers' tables, and leakage checks, per dataset.

    shape      N / M / #con / #cat / K after cleaning vs the paper's printed row
    overlap    exact duplicate feature rows shared by train and test
    leak       any SINGLE feature that predicts the minority label almost
               perfectly on the training split (AUC >= 0.99) -- the signature of
               a target leak (e.g. the Naive_Bayes_* columns of BankChurners)
    protocol   held-out minority, validation and test are disjoint from D_train
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..core.data import registry
from ..core.data import protocol
from ..core.data.fetch import raw_path


def check(name: str) -> dict:
    if not raw_path(name).exists():
        return {"dataset": name, "status": "not downloaded"}
    v = protocol.verify(name)
    if v["status"] == "error":
        return v
    s = protocol.prepare(name, seed=0)
    sch = s.schema
    key = lambda df: set(map(tuple, df[sch.features].astype(str).to_numpy().tolist()))  # noqa: E731
    overlap = len(key(s.imbalanced_train) & key(s.test))
    tr = s.train_minor_full.assign(_y=1)
    D = s.train_major.assign(_y=0)
    both = pd.concat([D, tr], ignore_index=True)
    leaks = {}
    for c in sch.features:
        x = both[c]
        if c in sch.continuous:
            a = roc_auc_score(both["_y"], x.astype(float))
            a = max(a, 1 - a)
        else:
            # out-of-fold category rate: in-sample target means make any
            # high-cardinality column (e.g. fuel/MODEL) look like a perfect leak
            from sklearn.model_selection import KFold
            rate = np.full(len(both), both["_y"].mean())
            xs = x.astype(str).to_numpy()
            for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(both):
                m = pd.Series(both["_y"].to_numpy()[tr_i]).groupby(xs[tr_i]).mean()
                rate[te_i] = pd.Series(xs[te_i]).map(m).fillna(both["_y"].mean()).to_numpy()
            a = roc_auc_score(both["_y"], rate)
        if a >= 0.99:
            leaks[c] = round(float(a), 4)
    disjoint = (len(s.minor_heldout) + len(s.train_minor) == len(s.train_minor_full))
    return {**{k: v[k] for k in ("dataset", "status", "expected", "actual", "delta")},
            "train_test_duplicate_rows": overlap, "single_feature_leaks": leaks,
            "protocol_disjoint": bool(disjoint), "notes": registry.get(name).notes}


def run(names) -> list[dict]:
    return [check(n) for n in registry.resolve(names)]


__all__ = ["run", "check", "np"]
