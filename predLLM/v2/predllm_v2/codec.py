"""Schema + canonical piece-wise tokenization (v2 novelty N1, N3).

A row is serialized as a sequence of PIECES that are tokenized independently and concatenated:

    [BOS] key(c1) val(c1) SEP  key(c2) val(c2) SEP ... key(y) val(y) EOS
    key(c) = "c is" (first field) / " c is" (later fields),  val = " <value>",  SEP = ","

Because training (encode) and generation (sampler/posterior) build token ids from the SAME cached
pieces, the token sequence of a value never depends on its neighbours. That is what lets the decoder
(a) score every categorical value exactly and (b) write keys/separators itself.

Per-token loss weights (N3): key tokens -> w_scaffold (default 0, since the decoder writes them),
value tokens + their terminator (SEP/EOS) -> 1 for features and lambda for the target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predLLM.common.tabgen.data.preprocess import TARGET, TabularSplit
from predLLM.common.tabgen.utils import format_number


@dataclass
class Schema:
    features: list[str]
    target: str                                   # name used in text
    task: str
    kinds: dict[str, str]                         # col -> "num" | "cat"   (target: "cat" iff classification)
    decimals: dict[str, int] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    cats: dict[str, list[str]] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.features + [self.target]

    def fmt(self, col: str, v) -> str:
        return format_number(v, self.decimals[col]) if self.kinds[col] == "num" else str(v)

    def numeric_spec(self, col: str) -> tuple[int, int, bool]:
        lo, hi = self.ranges[col]
        max_int = max(1, len(str(int(max(abs(lo), abs(hi))))))
        return max_int, self.decimals[col], lo < 0


def _infer_decimals(s: pd.Series, cap: int) -> int:
    v = s.to_numpy(float)
    for d in range(cap + 1):
        if np.mean(np.abs(v - np.round(v, d)) < 1e-9) >= 0.99:
            return d
    return cap


def build_schema(split: TabularSplit, serialization: str, max_decimals: int = 4) -> Schema:
    """serialization='raw'     : real category strings and real units (v2 default, N1)
       serialization='encoded' : the paper's MinMax/code space (ablation)"""
    df, target = (split.raw_train, split.target) if serialization == "raw" else (split.train, TARGET)
    kinds = {c: ("cat" if c in split.cat_cols else "num") for c in split.features}
    kinds[target] = "cat" if split.task == "classification" else "num"
    sch = Schema(list(split.features), target, split.task, kinds)
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
            raise ValueError(f"v2 needs ',' to be a single token; {tok.name_or_path} gives {sep}")
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

    def encode(self, row: dict[str, str], order: list[str], lam: float = 1.0,
               w_scaffold: float = 1.0) -> tuple[list[int], list[float]]:
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
        """BOS + complete fields of `order` (each ended by SEP) [+ key of `then_key`]."""
        ids = list(self.bos)
        for j, c in enumerate(order):
            ids += self.key_ids(c, j == 0) + self.val_ids(row[c]) + [self.sep]
        if then_key is not None:
            ids += self.key_ids(then_key, len(order) == 0)
        return ids
