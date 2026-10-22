"""Textual encoding (Sec. III-B1a), permutation (III-B1b) and parsing of generated text.

    row [X1=v1, ..., XM=vM, Y=y]  ->  "X1 is v1, ..., XM is vM, target is y"
"""
from __future__ import annotations

import numpy as np

from predLLM.common.tabgen.data.preprocess import TARGET
from predLLM.common.tabgen.utils import format_number


def row_fields(row: dict, features: list[str], task: str, decimals: int = 2, target_decimals: int = 4) -> dict:
    """Formats every value of one encoded row as text (features rounded like the paper)."""
    out = {f: format_number(row[f], decimals) for f in features}
    if TARGET in row:
        y = row[TARGET]
        out[TARGET] = str(int(y)) if task == "classification" else format_number(y, target_decimals)
    return out


def column_order(features: list[str], strategy: str, rng: np.random.Generator, original: bool = False,
                 with_target: bool = True) -> list[str]:
    """permute_x: shuffle features, target last. permute_xy: shuffle everything. original: no shuffle."""
    cols = list(features)
    if original:
        return cols + ([TARGET] if with_target else [])
    if strategy == "x":
        return [cols[i] for i in rng.permutation(len(cols))] + ([TARGET] if with_target else [])
    if strategy == "xy":
        allc = cols + ([TARGET] if with_target else [])
        return [allc[i] for i in rng.permutation(len(allc))]
    raise ValueError(strategy)


def encode(fields: dict, order: list[str]) -> str:
    return ", ".join(f"{c} is {fields[c]}" for c in order if c in fields)


def parse(text: str, columns: list[str]) -> dict[str, float]:
    """Inverse of `encode` for generated text. Keeps the first numeric value per known column and
    ignores everything else (unknown names, repeated columns, non-numeric values)."""
    colset = set(columns)
    out: dict[str, float] = {}
    for piece in text.split(","):
        name, sep, val = piece.strip().partition(" is ")
        if not sep or name not in colset or name in out:
            continue
        try:
            v = float(val.strip().split(" ")[0])
        except ValueError:
            continue
        if np.isfinite(v):
            out[name] = v
    return out
