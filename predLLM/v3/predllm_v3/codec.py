"""Schema + canonical piece-wise tokenization with outlier support bounds (v3 novelty).

A row is serialized as a sequence of PIECES that are tokenized independently and concatenated:
    [BOS] key(c1) val(c1) SEP  key(c2) val(c2) SEP ... key(y) val(y) EOS
    key(c) = "c is" (first field) / " c is" (later fields),  val = " <value>",  SEP = ","

v3 additions:
1. Support for both global empirical ranges and targeted outlier/minority ranges (outlier_ranges).
2. Explicit detection and registry of minority vs. majority classes for negative alignment.
3. Numeric spec computation tailored to the generation range mode (outlier vs empirical).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predLLM.common.tabgen.data.preprocess import TARGET, TabularSplit
from predLLM.common.tabgen.utils import format_number

from .outlier import compute_outlier_ranges


@dataclass
class Schema:
    features: list[str]
    target: str                                   # name used in text
    task: str
    kinds: dict[str, str]                         # col -> "num" | "cat"   (target: "cat" iff classification)
    decimals: dict[str, int] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    outlier_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    cats: dict[str, list[str]] = field(default_factory=dict)
    minority_classes: list[str] = field(default_factory=list)
    majority_classes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return self.features + [self.target]

    def fmt(self, col: str, v) -> str:
        return format_number(v, self.decimals[col]) if self.kinds[col] == "num" else str(v)

    def numeric_spec(self, col: str, range_mode: str = "outlier") -> tuple[int, int, bool]:
        target_ranges = self.outlier_ranges if (range_mode == "outlier" and col in self.outlier_ranges) else self.ranges
        lo, hi = target_ranges[col]
        max_int = max(1, len(str(int(max(abs(lo), abs(hi))))))
        return max_int, self.decimals[col], lo < 0

    def active_range(self, col: str, range_mode: str = "outlier") -> tuple[float, float]:
        if range_mode == "outlier" and col in self.outlier_ranges:
            return self.outlier_ranges[col]
        return self.ranges[col]


def _infer_decimals(s: pd.Series, cap: int) -> int:
    v = s.to_numpy(float)
    for d in range(cap + 1):
        if np.mean(np.abs(v - np.round(v, d)) < 1e-9) >= 0.99:
            return d
    return cap


def build_schema(
    split: TabularSplit,
    serialization: str = "raw",
    max_decimals: int = 4,
    outlier_mode: str = "iqr",
    outlier_factor: float = 1.5,
) -> Schema:
    """Builds v3 Schema with empirical ranges, outlier envelopes, and class partitions."""
    df, target = (split.raw_train, split.target) if serialization == "raw" else (split.train, TARGET)
    kinds = {c: ("cat" if c in split.cat_cols else "num") for c in split.features}
    kinds[target] = "cat" if split.task == "classification" else "num"
    sch = Schema(list(split.features), target, split.task, kinds)

    # Class partition for negative sample detection
    if split.task == "classification":
        counts = df[target].value_counts().astype(float)
        min_count = counts.min()
        max_count = counts.max()
        # Only partition into majority/minority if genuine class imbalance exists (ratio >= 1.5)
        if max_count / max(min_count, 1) >= 1.5:
            median_count = counts.median()
            sch.minority_classes = [str(c) for c in counts[counts <= median_count].index.tolist()]
            sch.majority_classes = [str(c) for c in counts[counts > median_count].index.tolist()]
        else:
            # Balanced classes: all classes are positive targets
            sch.minority_classes = [str(c) for c in counts.index.tolist()]
            sch.majority_classes = []

    num_cols = [c for c in sch.columns if kinds[c] == "num"]
    for c in sch.columns:
        if kinds[c] == "num":
            dec = 2 if (serialization == "encoded" and c != target) else _infer_decimals(df[c], max_decimals)
            sch.decimals[c] = dec
            sch.ranges[c] = (float(df[c].min()), float(df[c].max()))
        elif c == target:
            sch.cats[c] = [str(x) for x in (split.pre.classes if serialization == "raw" else range(split.n_classes))]
        else:
            vals = df[c] if serialization == "raw" else df[c].map(lambda v: format_number(v, 2))
            sch.cats[c] = sorted(vals.astype(str).unique().tolist())

    # Outlier ranges for numerical features
    sch.outlier_ranges = compute_outlier_ranges(
        df,
        [c for c in split.features if kinds[c] == "num"],
        mode=outlier_mode,
        factor=outlier_factor,
        target_col=target if split.task == "classification" else None,
        minority_classes=sch.minority_classes,
    )
    if kinds[target] == "num":
        sch.outlier_ranges[target] = sch.ranges[target]

    return sch


def rows_as_text(df: pd.DataFrame, schema: Schema) -> list[dict[str, str]]:
    cols = [c for c in schema.columns if c in df.columns]
    return [{c: schema.fmt(c, r[c]) for c in cols} for r in df[cols].to_dict("records")]


class FieldCodec:
    def __init__(self, schema: Schema, tok):
        self.schema, self.tok = schema, tok
        self.bos = tok("", add_special_tokens=True)["input_ids"]
        sep = tok(",", add_special_tokens=False)["input_ids"]
        if len(sep) != 1:
            raise ValueError(f"v3 needs ',' to be a single token; {tok.name_or_path} gives {sep}")
        self.sep, self.eos = sep[0], tok.eos_token_id
        self._key: dict[tuple[str, bool], list[int]] = {}
        self._val: dict[str, list[int]] = {}

    def key_ids(self, col: str, first: bool) -> list[int]:
        k = (col, first)
        if k not in self._key:
            self._key[k] = self.tok(("" if first else " ") + f"{col} is", add_special_tokens=False)["input_ids"]
        return self._key[k]

    def val_ids(self, s: str) -> list[int]:
        if s not in self._val:
            self._val[s] = self.tok(" " + s, add_special_tokens=False)["input_ids"]
        return self._val[s]

    def terminator(self, col: str) -> int:
        return self.eos if col == self.schema.target else self.sep

    def encode(
        self,
        row: dict[str, str],
        order: list[str],
        lam: float = 1.0,
        w_scaffold: float = 1.0,
    ) -> tuple[list[int], list[float]]:
        ids, w = list(self.bos), [0.0] * len(self.bos)
        for j, c in enumerate(order):
            k = self.key_ids(c, j == 0)
            ids += k
            w += [w_scaffold] * len(k)
            v = self.val_ids(row[c]) + [self.terminator(c)]
            ids += v
            w += [lam if c == self.schema.target else 1.0] * len(v)
        return ids, w

    def prefix(self, row: dict[str, str], order: list[str], then_key: str | None = None) -> list[int]:
        ids = list(self.bos)
        for j, c in enumerate(order):
            ids += self.key_ids(c, j == 0) + self.val_ids(row[c]) + [self.sep]
        if then_key is not None:
            ids += self.key_ids(then_key, len(order) == 0)
        return ids
