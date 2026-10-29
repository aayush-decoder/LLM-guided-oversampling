"""Downstream classification: a classifier fit on a training frame, scored on D_test.

LEARNERS (G1 of docs/RESEARCH_REVIEW.md). XGBoost with default parameters is the
paper's learner and the source of every UNPREFIXED key (`f1`, `auc`, ...). The
same training frame is also scored with the extra learners in `EXTRA_LEARNERS`
(prefix `<learner>_`, e.g. `lr_f1`, `rf_auprc`), because rebalancing matters
far more for weak or uncalibrated learners than for boosted trees (Elor &
Averbuch-Elor 2022): a conclusion such as "a tuned threshold matches LLM
oversampling" must be shown per learner, not for XGBoost alone.
    xgb   XGBoost (paper)                       native NaN handling
    lr    standardised logistic regression      class_weight=None (rules do the work)
    rf    random forest, 300 trees
    mlp   standardised 1-hidden-layer MLP       (no sample weights -> NaN for
                                                 scale_pos_weight)
    lgbm  LightGBM, only if installed
Set IMBLLM_LEARNERS=xgb to restrict (e.g. for quick sweeps).

EVERY METHOD REPORTS ITS OWN DECISION RULE UNDER `f1`
    oversamplers / imbalance / original / scale_pos_weight : argmax
    threshold_only : minority threshold tuned on the REAL validation fold
    prior_shift    : argmax of p(y|x) / p_train(y)   (Elkan 2001, Thm 2;
                     Menon et al. 2021 logit adjustment) -- no val fold needed
palt2 scored threshold_only at 0.5, which made it numerically identical to
imbalance, so the "missing baseline" was never actually compared.

`f1_tuned` is reported for every method as a secondary number: the threshold is
chosen on the real validation fold (protocol "clean"); NaN under protocol
"paper", which has no fold. palt2 tuned it on a slice of the REBALANCED set,
i.e. against ~94% synthetic positives.

MULTICLASS (iris, balance, waveform, cardio, car): the classifier is K-way.
Minority metrics are one-vs-rest on the minority class; `macro_f1` is added.
The paper-comparable keys are `f1` (minority F1) and `auc` (minority AUC).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from ..data.protocol import Schema, Splits

XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0,
                  colsample_bytree=1.0, tree_method="hist", verbosity=0, n_jobs=4)


class Encoder:
    """One-hot categoricals with the vocabulary of the TRAINING frame; continuous
    passthrough. Unknown categories at test time -> all-zero block."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.cats: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "Encoder":
        self.cats = {c: sorted(df[c].astype(str).unique()) for c in self.schema.categorical}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = [df[self.schema.continuous].apply(pd.to_numeric, errors="coerce")
                  .to_numpy(np.float32)] if self.schema.continuous else []
        for c in self.schema.categorical:
            v = df[c].astype(str).to_numpy()
            blocks.append((v[:, None] == np.array(self.cats[c])[None, :]).astype(np.float32))
        return np.hstack(blocks) if blocks else np.zeros((len(df), 1), np.float32)


def labels(df: pd.DataFrame, schema: Schema) -> np.ndarray:
    lut = {c: i for i, c in enumerate(schema.classes)}
    return df[schema.target].astype(str).map(lut).to_numpy()


PRIMARY = "xgb"
EXTRA_LEARNERS_DEFAULT = ("lr", "rf", "mlp", "lgbm")


def extra_learners() -> tuple[str, ...]:
    env = os.environ.get("IMBLLM_LEARNERS")
    names = tuple(x.strip() for x in env.split(",")) if env else (PRIMARY,
                                                                  *EXTRA_LEARNERS_DEFAULT)
    out = []
    for n in names:
        if n == PRIMARY:
            continue
        if n == "lgbm":
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                continue
        if n not in ("lr", "rf", "mlp", "lgbm"):
            raise ValueError(f"unknown learner {n!r}; have xgb, lr, rf, mlp, lgbm")
        out.append(n)
    return tuple(out)


def _model(learner: str, seed: int):
    if learner == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS, random_state=seed)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if learner == "lr":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if learner == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=seed)
    if learner == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=seed))
    if learner == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
    raise ValueError(learner)


def _fit(train: pd.DataFrame, schema: Schema, seed: int, weights=None, learner: str = "xgb"):
    y = labels(train, schema)
    present = np.unique(y)
    # contiguous labels 0..k-1 (xgboost requires it); probabilities are mapped back later
    remap = {c: i for i, c in enumerate(present)}
    enc = Encoder(schema).fit(train)
    X = enc.transform(train)
    if learner != "xgb":
        X = np.nan_to_num(X)
    model = _model(learner, seed)
    yy = np.vectorize(remap.get)(y)
    if weights is None:
        model.fit(X, yy)
    elif learner in ("lr", "mlp"):
        if learner == "mlp":
            raise TypeError("mlp does not take sample weights")
        model.fit(X, yy, logisticregression__sample_weight=weights)
    else:
        model.fit(X, yy, sample_weight=weights)
    model._imbllm_learner = learner
    return model, enc, present


def _proba(model, enc, present, df: pd.DataFrame, K: int) -> np.ndarray:
    P = np.zeros((len(df), K))
    if len(df) == 0:
        return P
    X = enc.transform(df)
    if getattr(model, "_imbllm_learner", "xgb") != "xgb":
        X = np.nan_to_num(X)
    p = model.predict_proba(X)
    P[:, present] = p
    return P


def _decide(P: np.ndarray, m: int, rule: str, thr: float | None = None,
            prior: np.ndarray | None = None) -> np.ndarray:
    if rule == "argmax":
        return P.argmax(1)
    if rule == "prior":
        return (P / np.clip(prior, 1e-12, None)).argmax(1)
    if rule == "threshold":
        rest = P.copy()
        rest[:, m] = -1
        return np.where(P[:, m] >= thr, m, rest.argmax(1))
    raise ValueError(rule)


def _tune(p_min: np.ndarray, is_min: np.ndarray) -> float:
    if is_min.sum() == 0 or is_min.all():
        return 0.5
    cand = np.unique(np.clip(p_min, 1e-6, 1 - 1e-6))
    best, best_t = -1.0, 0.5
    for t in cand:
        f = f1_score(is_min, p_min >= t, zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def _metrics(y: np.ndarray, P: np.ndarray, pred: np.ndarray, m: int) -> dict:
    is_min = (y == m).astype(int)
    two = len(np.unique(is_min)) == 2
    return {
        "f1": float(f1_score(is_min, (pred == m).astype(int), zero_division=0)),
        "auc": float(roc_auc_score(is_min, P[:, m])) if two else float("nan"),
        "auprc": float(average_precision_score(is_min, P[:, m])) if two else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "bacc": float(balanced_accuracy_score(y, pred)),
    }


def evaluate(train: pd.DataFrame, splits: Splits, *, seed: int = 0, rule: str = "argmax",
             weights: np.ndarray | None = None, learners: tuple[str, ...] | None = None) -> dict:
    """Fit XGBoost on `train`, score on test with `rule`; add val-tuned f1_tuned.
    Then repeat with every extra learner under the prefix `<learner>_`."""
    out = _evaluate_one(train, splits, seed=seed, rule=rule, weights=weights, learner=PRIMARY)
    if "error" in out:
        return out
    for ln in (extra_learners() if learners is None else learners):
        try:
            r = _evaluate_one(train, splits, seed=seed, rule=rule, weights=weights, learner=ln)
        except TypeError as exc:                          # learner cannot take weights
            r = {"error": str(exc)}
        for k in ("f1", "auc", "auprc", "macro_f1", "bacc", "f1_tuned"):
            out[f"{ln}_{k}"] = r.get(k, float("nan"))
    return out


def _evaluate_one(train: pd.DataFrame, splits: Splits, *, seed: int, rule: str,
                  weights: np.ndarray | None, learner: str) -> dict:
    s = splits.schema
    K, m = len(s.classes), s.classes.index(s.minority_label)
    y_tr = labels(train, s)
    if len(np.unique(y_tr)) < 2:
        return {"error": "single-class training set"}
    model, enc, present = _fit(train, s, seed, weights, learner)
    P_te = _proba(model, enc, present, splits.test, K)
    y_te = labels(splits.test, s)

    thr = None
    has_val = len(splits.val) > 0
    if has_val:
        P_va = _proba(model, enc, present, splits.val, K)
        thr = _tune(P_va[:, m], (labels(splits.val, s) == m).astype(int))
    prior = np.bincount(y_tr, minlength=K) / len(y_tr)

    if rule == "threshold" and not has_val:
        return {"error": "threshold rule needs a validation fold (protocol 'clean')"}
    out = _metrics(y_te, P_te, _decide(P_te, m, rule, thr, prior), m)
    out["f1_tuned"] = float(f1_score((y_te == m).astype(int),
                                     (_decide(P_te, m, "threshold", thr) == m).astype(int),
                                     zero_division=0)) if has_val else float("nan")
    out.update({"rule": rule, "threshold": thr, "n_train": int(len(train)),
                "n_minority_train": int((y_tr == m).sum()), "learner": learner})
    return out


# --------------------------------------------------------------------------- references

def _real_train(splits: Splits) -> pd.DataFrame:
    return pd.concat([splits.train_major, splits.train_minor, splits.minor_heldout],
                     ignore_index=True)


REFERENCES = ("original", "imbalance", "threshold_only", "prior_shift", "scale_pos_weight")


def evaluate_reference(name: str, splits: Splits, seed: int = 0) -> dict:
    s = splits.schema
    if name == "original":        # D_major U D*_minor: nothing thrown away (upper ref.)
        return evaluate(_real_train(splits), splits, seed=seed)
    D = splits.imbalanced_train
    if name == "imbalance":
        return evaluate(D, splits, seed=seed)
    if name == "threshold_only":
        return evaluate(D, splits, seed=seed, rule="threshold")
    if name == "prior_shift":
        return evaluate(D, splits, seed=seed, rule="prior")
    if name == "scale_pos_weight":
        y = D[s.target].astype(str).to_numpy()
        w = np.where(y == s.minority_label, splits.n_target / max(len(splits.train_minor), 1),
                     1.0)
        return evaluate(D, splits, seed=seed, weights=w)
    raise KeyError(f"unknown reference {name!r}; have {REFERENCES}")


def evaluate_synthetic(synth: pd.DataFrame, splits: Splits, seed: int = 0) -> dict:
    """Score an oversampler BOTH ways (no extra generation needed):
        paper     : D_major U D^_minor               (paper, verbatim: real minority dropped)
        with_real : D_major U D_minor U D^_minor     (v2 protocol)
    Keys of the first are unprefixed (paper-comparable); the second is `real_*`."""
    s = splits.schema
    cols = [*s.features, s.target]
    syn = synth.copy()
    syn[s.target] = s.minority_label
    paper = pd.concat([splits.train_major[cols], syn[cols]], ignore_index=True)
    real = pd.concat([splits.train_major[cols], splits.train_minor[cols], syn[cols]],
                     ignore_index=True)
    a = evaluate(paper, splits, seed=seed)
    b = evaluate(real, splits, seed=seed)
    return {**a, **{f"real_{k}": v for k, v in b.items()}}
