"""CLLM-style curation of generated minority rows (Seedat et al., ICML 2024).

Why (docs/RESEARCH_REVIEW.md, G5). Curating LLM-generated tabular rows by the
learning dynamics of a classifier trained on the REAL data is the published
state of the art for "LLM rows + small real set". It is the natural competitor
for v2's refinement steps, and it can be bolted onto ANY oversampler, so the
`curation` experiment asks: does curation help ImbLLM more than it helps SMOTE-NC?

Procedure (CLLM, adapted to oversampling one class):
    1. fit XGBoost on the real imbalanced training set D_train;
    2. for each generated row x (label = minority) record p_e(x) = P(minority | x)
       at checkpoints e = 10, 20, ..., n_estimators trees (the "training dynamics");
    3. confidence  c(x) = mean_e p_e(x);   aleatoric  a(x) = mean_e p_e(x)(1 - p_e(x));
    4. keep x if c(x) >= tau_c and a(x) <= tau_a.
CLLM uses fixed thresholds on a balanced problem. Under imbalance a real minority
row often has p < 0.5, so fixed thresholds would discard nearly everything.
Here tau_c / tau_a are CALIBRATED on the real minority rows, out of fold:
tau_c = 10th percentile of c, tau_a = 90th percentile of a over D_minor (5-fold).
This keeps generated rows that look at least as learnable as ~90% of real ones.
Rows are discarded, never replaced: the shortfall is reported, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.protocol import Splits
from ..evaluate.downstream import XGB_PARAMS, Encoder, labels


def _dynamics(model, X: np.ndarray, m_idx: int, n_trees: int, step: int = 10) -> np.ndarray:
    ps = []
    for e in range(step, n_trees + 1, step):
        p = model.predict_proba(X, iteration_range=(0, e))
        ps.append(p[:, m_idx])
    return np.stack(ps, 1)                                 # (n, checkpoints)


def _fit(D: pd.DataFrame, splits: Splits, seed: int):
    from xgboost import XGBClassifier
    s = splits.schema
    y = labels(D, s)
    present = np.unique(y)
    remap = {c: i for i, c in enumerate(present)}
    enc = Encoder(s).fit(D)
    model = XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(enc.transform(D), np.vectorize(remap.get)(y))
    m = s.classes.index(s.minority_label)
    return model, enc, remap.get(m)


def cllm_filter(synth: pd.DataFrame, splits: Splits, seed: int = 0, *,
                q_conf: float = 0.10, q_alea: float = 0.90) -> tuple[pd.DataFrame, dict]:
    from sklearn.model_selection import StratifiedKFold
    s = splits.schema
    if len(synth) == 0:
        return synth, {"kept": 0, "generated": 0}
    D = splits.imbalanced_train.reset_index(drop=True)
    n_trees = XGB_PARAMS["n_estimators"]
    # thresholds from real minority rows, out of fold
    y = labels(D, s)
    m = s.classes.index(s.minority_label)
    conf_real, alea_real = [], []
    n_splits = int(min(5, np.bincount(y)[np.bincount(y) > 0].min()))
    if n_splits >= 2:
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(D, y):
            model, enc, mi = _fit(D.iloc[tr], splits, seed)
            te_min = D.iloc[te][y[te] == m]
            if mi is None or len(te_min) == 0:
                continue
            P = _dynamics(model, enc.transform(te_min), mi, n_trees)
            conf_real.append(P.mean(1))
            alea_real.append((P * (1 - P)).mean(1))
    if not conf_real:
        return synth, {"kept": len(synth), "generated": len(synth),
                       "note": "too few minority rows to calibrate; nothing filtered"}
    tau_c = float(np.quantile(np.concatenate(conf_real), q_conf))
    tau_a = float(np.quantile(np.concatenate(alea_real), q_alea))
    model, enc, mi = _fit(D, splits, seed)
    P = _dynamics(model, enc.transform(synth[s.features]), mi, n_trees)
    conf, alea = P.mean(1), (P * (1 - P)).mean(1)
    keep = (conf >= tau_c) & (alea <= tau_a)
    return synth[keep].reset_index(drop=True), {
        "generated": int(len(synth)), "kept": int(keep.sum()),
        "kept_share": float(keep.mean()), "tau_confidence": tau_c, "tau_aleatoric": tau_a,
        "q_conf": q_conf, "q_alea": q_alea}


CURATORS = {"cllm": cllm_filter}
