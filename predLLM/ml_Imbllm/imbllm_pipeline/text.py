from __future__ import annotations

import re
import numpy as np


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def row_to_text(row, features, seed=None, fix_y=True):
    """Serialize a row with Class first and optional per-row feature permutation."""
    columns = list(features)
    if fix_y:
        rng = np.random.default_rng(seed)
        rng.shuffle(columns)
    return "Class is 1, " + ", ".join(
        f"{column} is {float(row[column]):.8g}" for column in columns
    ) + "."


def parse_generated_text(text, features, reference, target="Class"):
    """Parse one generation and return values, validity, and an explicit reason."""
    values = {
        key: float(value)
        for key, value in re.findall(
            rf"([A-Za-z][A-Za-z0-9]*)\s+is\s+({NUMBER_PATTERN})", text
        )
        if key in set(features) | {target}
    }
    missing = [column for column in [target, *features] if column not in values]
    if missing:
        return values, False, "missing columns: " + ", ".join(missing)
    if values[target] != 1:
        return values, False, "Class is not 1"
    if not np.isfinite(list(values.values())).all():
        return values, False, "non-finite value"
    for column in features:
        low, high = reference[column].quantile([0.001, 0.999])
        margin = 10 * max(1.0, abs(high - low))
        if not (low - margin <= values[column] <= high + margin):
            return values, False, f"{column} is outside the broad training range"
    return values, True, "valid"
