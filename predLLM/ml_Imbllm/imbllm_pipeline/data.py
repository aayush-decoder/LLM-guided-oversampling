from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_credit_card(path, target="Class", test_size=0.2, seed=42):
    """Load the dataset and perform the only train/test split used by the study."""
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} is missing.")
    if frame[target].isna().any():
        raise ValueError("The target contains missing values.")
    features = [column for column in frame.columns if column != target]
    x = frame[features].astype(float)
    y = frame[target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=seed
    )
    return frame, x_train, x_test, y_train, y_test, features


def build_minority_partitions(x_train, y_train, q=0.2, seed=42, target="Class"):
    train = pd.concat([x_train, y_train.rename(target)], axis=1)
    majority = train[train[target].eq(0)].copy()
    all_minority = train[train[target].eq(1)].copy()
    visible = all_minority.sample(frac=q, random_state=seed)
    unseen = all_minority.drop(visible.index)
    return majority, all_minority, visible, unseen


def interpolate_minority(visible, count, features, seed=42, target="Class"):
    """Create numerical convex combinations; the target is never interpolated."""
    if visible.empty:
        raise ValueError("Cannot interpolate an empty visible minority set.")
    rng = np.random.default_rng(seed)
    values = visible[features].to_numpy(dtype=float)
    rows = []
    for _ in range(count):
        left, right = rng.integers(0, len(values), size=2)
        alpha = rng.uniform()
        rows.append(alpha * values[left] + (1 - alpha) * values[right])
    result = pd.DataFrame(rows, columns=features, index=None)
    result[target] = 1
    return result
