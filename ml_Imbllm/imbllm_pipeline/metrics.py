from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, pairwise_distances,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp


def evaluate_classifier(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    return {
        "Method": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probability),
        "PR-AUC": average_precision_score(y_test, probability),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "_probability": probability,
    }


def quality_metrics(real, synthetic, features, k=2):
    """Return DCR plus documented radius-based Close Probability/Coverage approximations."""
    scaler = StandardScaler().fit(pd.concat([real[features], synthetic[features]]))
    real_scaled = scaler.transform(real[features])
    synthetic_scaled = scaler.transform(synthetic[features])
    distances = pairwise_distances(real_scaled, synthetic_scaled)
    dcr = distances.min(axis=1)
    real_to_real = pairwise_distances(real_scaled, real_scaled)
    np.fill_diagonal(real_to_real, np.inf)
    radius = float(np.median(real_to_real.min(axis=1)))
    close_probability = float((dcr <= radius).mean())
    coverage = float((distances <= radius).sum(axis=1).ge(k).mean())
    ks = pd.DataFrame(
        {
            "feature": column,
            "ks_statistic": ks_2samp(real[column], synthetic[column]).statistic,
            "ks_pvalue": ks_2samp(real[column], synthetic[column]).pvalue,
            "mean_difference": abs(real[column].mean() - synthetic[column].mean()),
            "std_difference": abs(real[column].std() - synthetic[column].std()),
        }
        for column in features
    ).sort_values("ks_statistic", ascending=False)
    return {
        "close_probability_approx": close_probability,
        "coverage_approx": coverage,
        "mean_ks": float(ks["ks_statistic"].mean()),
        "median_dcr": float(np.median(dcr)),
        "mean_dcr": float(dcr.mean()),
        "std_dcr": float(dcr.std()),
    }, ks, dcr
