"""Compare classifiers trained on original versus synthetic-augmented data.

Run this after the notebook has produced results/synthetic_tabular.csv:
    python compare_original_vs_synthetic.py
The test set is recreated with the notebook's fixed seed and is never used
while constructing the synthetic training data.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from predLLM.ml_Imbllm.imbllm_pipeline.data import load_and_split_credit_card


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "creditcard.csv" / "creditcard.csv"
SYNTHETIC = ROOT / "results" / "synthetic_tabular.csv"
SEED = 42


def make_model():
    return XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=SEED, n_jobs=-1,
    )


def evaluate(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "Method": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def main():
    if not SYNTHETIC.exists():
        raise FileNotFoundError(
            f"{SYNTHETIC} does not exist. Run the notebook through the generation cell first."
        )
    original, x_train, x_test, y_train, y_test, features = load_and_split_credit_card(
        DATA, target="Class", test_size=0.2, seed=SEED
    )
    synthetic = pd.read_csv(SYNTHETIC)
    required = set(features) | {"Class"}
    missing = required - set(synthetic.columns)
    if missing:
        raise ValueError(f"Synthetic file is missing columns: {sorted(missing)}")
    synthetic = synthetic[features + ["Class"]].dropna()
    synthetic["Class"] = synthetic["Class"].astype(int)
    if not synthetic["Class"].eq(1).all():
        raise ValueError("Synthetic training data must contain only Class=1 rows.")

    augmented_x = pd.concat([x_train, synthetic[features]], ignore_index=True)
    augmented_y = pd.concat([y_train, synthetic["Class"]], ignore_index=True)
    rows = [
        evaluate("Original training", make_model(), x_train, y_train, x_test, y_test),
        evaluate("Original + synthetic", make_model(), augmented_x, augmented_y, x_test, y_test),
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "results" / "original_vs_synthetic.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nInterpretation: compare F1 and PR-AUC first; use the same untouched test set for both rows.")


if __name__ == "__main__":
    main()
