"""Classical oversamplers (paper baselines AdaSyn / SMOTE / SMOTE-NC, plus ROS).

All produce exactly `splits.n_target` minority rows (|largest class|), fitted on
D_train only, minority-vs-rest (multiclass datasets keep their other classes
untouched downstream).

scale=True (default): continuous columns are z-scored before the k-NN search
and un-scaled afterwards. Unscaled k-NN on mixed-scale tables is "nearest on
the largest-unit column" (e.g. credit_amount), which is a strawman SMOTE.
scale=False reproduces a naive imblearn call on raw values, for comparison.

SMOTE / ADASYN need numeric input, so categoricals are one-hot encoded and
decoded by argmax -- the information loss the paper attributes to them.
SMOTE-NC handles mixed types natively and is the fair classical reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.protocol import Splits


def _xy(splits: Splits):
    s = splits.schema
    D = splits.imbalanced_train
    y = (D[s.target].astype(str) == s.minority_label).astype(int).to_numpy()
    return D[s.features].copy(), y


def _scale(X: pd.DataFrame, cols: list[str]):
    mu = X[cols].astype(float).mean()
    sd = X[cols].astype(float).std().replace(0, 1.0)
    Z = X.copy()
    Z[cols] = (X[cols].astype(float) - mu) / sd
    return Z, mu, sd


def _run(sampler_fn, splits: Splits, *, one_hot: bool, scale: bool, seed: int) -> pd.DataFrame:
    s = splits.schema
    X, y = _xy(splits)
    n_new = splits.n_target
    strategy = {1: int((y == 1).sum()) + n_new, 0: int((y == 0).sum())}
    k = max(1, min(5, int((y == 1).sum()) - 1))
    mu = sd = None
    if scale and s.continuous:
        X, mu, sd = _scale(X, s.continuous)
    if one_hot:
        D = pd.get_dummies(X, columns=s.categorical, dtype=float)
        Xr, yr = sampler_fn(strategy, k, seed).fit_resample(D.to_numpy(float), y)
        new = pd.DataFrame(Xr[len(D):], columns=D.columns)
        out = pd.DataFrame(index=new.index)
        for c in s.continuous:
            out[c] = new[c].astype(float)
        for c in s.categorical:
            cols = [f"{c}_{v}" for v in sorted(X[c].astype(str).unique())]
            cols = [x for x in cols if x in new.columns]
            out[c] = [cols[i][len(c) + 1:] for i in new[cols].to_numpy().argmax(1)]
        out = out[s.features]
    else:
        Xr, yr = sampler_fn(strategy, k, seed).fit_resample(X, y)
        out = pd.DataFrame(Xr, columns=s.features).iloc[len(X):].reset_index(drop=True)
    out = out[np.asarray(yr[len(X):]) == 1].reset_index(drop=True)
    if mu is not None:
        for c in s.continuous:
            out[c] = out[c].astype(float) * sd[c] + mu[c]
    return out


def ros(splits, seed=0, scale=True):
    from imblearn.over_sampling import RandomOverSampler
    return _run(lambda st, k, sd: RandomOverSampler(sampling_strategy=st, random_state=sd),
                splits, one_hot=False, scale=False, seed=seed)


def smote(splits, seed=0, scale=True):
    from imblearn.over_sampling import SMOTE
    return _run(lambda st, k, sd: SMOTE(sampling_strategy=st, k_neighbors=k, random_state=sd),
                splits, one_hot=True, scale=scale, seed=seed)


def adasyn(splits, seed=0, scale=True):
    from imblearn.over_sampling import ADASYN
    return _run(lambda st, k, sd: ADASYN(sampling_strategy=st, n_neighbors=k, random_state=sd),
                splits, one_hot=True, scale=scale, seed=seed)


def smote_nc(splits, seed=0, scale=True):
    from imblearn.over_sampling import SMOTENC
    s = splits.schema
    if not s.categorical:
        return smote(splits, seed, scale)
    if not s.continuous:                      # SMOTE-NC needs >= 1 continuous column
        from imblearn.over_sampling import SMOTEN
        return _run(lambda st, k, sd: SMOTEN(sampling_strategy=st, k_neighbors=k,
                                             random_state=sd),
                    splits, one_hot=False, scale=False, seed=seed)
    idx = [s.features.index(c) for c in s.categorical]
    return _run(lambda st, k, sd: SMOTENC(categorical_features=idx, sampling_strategy=st,
                                          k_neighbors=k, random_state=sd),
                splits, one_hot=False, scale=scale, seed=seed)


REGISTRY = {"ros": ros, "smote": smote, "adasyn": adasyn, "smote_nc": smote_nc}
PAPER_BASELINES = ("adasyn", "smote", "smote_nc")


def run(name: str, splits: Splits, seed: int = 0, scale: bool = True) -> pd.DataFrame:
    if name not in REGISTRY:
        raise KeyError(f"unknown classical baseline {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](splits, seed, scale)
