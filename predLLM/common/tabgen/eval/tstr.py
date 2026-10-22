# Train-on-Synthetic / Test-on-Real (Fig. 1 of the paper): fit XGBoost (default hyper-parameters,
# random_state=seed) on D_fake and score it on the real held-out D_test.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from ..data.preprocess import TARGET


def xgb_model(task: str, seed: int):
    if task == "classification":
        return XGBClassifier(random_state=seed, n_jobs=-1)
    return XGBRegressor(random_state=seed, n_jobs=-1)


def tstr(train: pd.DataFrame, test: pd.DataFrame, features: list[str], task: str, n_classes: int,
         seed: int) -> dict:
    Xtr, ytr = train[features].to_numpy(float), train[TARGET].to_numpy()
    Xte, yte = test[features].to_numpy(float), test[TARGET].to_numpy()
    if task == "classification":
        ytr = np.asarray(ytr).astype(int)
        present = np.unique(ytr)
        # Paper rule: "Accuracy=0 means the method cannot generate a valid D_fake (e.g. only one class)".
        if len(present) != n_classes or not np.array_equal(present, np.arange(n_classes)):
            return {"accuracy": 0.0, "f1_macro": 0.0, "valid": False,
                    "note": f"synthetic labels {present.tolist()} != {n_classes} classes"}
        clf = xgb_model(task, seed).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return {"accuracy": round(float(accuracy_score(yte, pred)), 4),
                "f1_macro": round(float(f1_score(yte, pred, average="macro")), 4), "valid": True}
    reg = xgb_model(task, seed).fit(Xtr, ytr.astype(float))
    pred = reg.predict(Xte)
    mse = float(mean_squared_error(yte, pred))
    return {"mse": round(mse, 4), "rmse": round(mse ** 0.5, 4), "mae": round(float(mean_absolute_error(yte, pred)), 4),
            "r2": round(float(r2_score(yte, pred)), 4), "valid": True}


def primary_metric(task: str) -> tuple[str, bool]:
    """(metric name, higher_is_better)"""
    return ("accuracy", True) if task == "classification" else ("mse", False)
