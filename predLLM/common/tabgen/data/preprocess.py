"""Three data stages, used by every generator and by the visualisations:

  raw      exactly as downloaded                                  (download.read_raw)
  clean    selected features, stripped strings, NaN rows dropped,
           categorical -> str, numerical -> float, target first    (clean_table)
  encoded  paper protocol (Pred-LLM read_data.py): categories -> integer codes, then MinMax
           scaling of every feature, rounded to 2 decimals; class labels -> 0..K-1    (TablePreprocessor)

TSTR evaluation always happens in the *encoded* space, so every method (v1, v2, SDV) is scored on
identical features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import get_logger
from .download import read_raw
from .registry import get_spec

log = get_logger("tabgen.data")
TARGET = "target"          # target column name in the encoded space (the paper serializes "target is y")


@dataclass
class CleanTable:
    name: str
    task: str
    df: pd.DataFrame                    # features ... + target (original target name)
    features: list[str]                 # feature order used for serialization ("original order")
    num_cols: list[str]
    cat_cols: list[str]
    target: str
    report: dict = field(default_factory=dict)


def _natural_sort(values):
    vals = list(values)
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def _label_str(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def clean_table(name: str, binarize: str | None = None) -> CleanTable:
    spec = get_spec(name)
    df, meta = read_raw(name)
    report = {"raw_shape": list(df.shape)}
    target = spec.target if spec.target != "auto" else meta["target"]

    drop = set(spec.drop) | {c for c in df.columns for p in spec.drop_prefix if c.startswith(p)}
    df = df.drop(columns=[c for c in drop if c in df.columns])
    num = spec.numerical if spec.numerical is not None else meta.get("numerical")
    cat = spec.categorical if spec.categorical is not None else meta.get("categorical")
    paper_order = spec.numerical is not None or spec.categorical is not None
    feats_all = [c for c in df.columns if c != target]
    if num is None and cat is None:
        num = [c for c in feats_all if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in feats_all if c not in num]
    num = [c for c in (num or []) if c in df.columns and c != target]
    cat = [c for c in (cat or []) if c in df.columns and c != target]
    feats = num + cat if paper_order else [c for c in feats_all if c in set(num) | set(cat)]
    df = df[feats + [target]].copy()

    # strings: strip, '?' -> NaN, commas are the serialization separator so replace them
    for c in df.columns:
        if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    report["missing_per_column"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    n0 = len(df)
    if spec.dropna:
        df = df.dropna()
    report["rows_dropped_na"] = n0 - len(df)
    for c in cat:
        df[c] = df[c].map(_label_str).str.replace(",", ";", regex=False)
    df[num] = df[num].astype(float)

    task = spec.task
    binarize = binarize if binarize is not None else spec.binarize
    if task == "regression":
        df[target] = pd.to_numeric(df[target], errors="coerce").astype(float)
        df = df.dropna(subset=[target])
        if binarize == "median":
            med = df[target].median()
            df[target] = (df[target] > med).astype(int).astype(str)
            task = "classification"
            report["binarized_at"] = float(med)
    else:
        df[target] = df[target].map(_label_str)
    df = df.reset_index(drop=True)
    report["clean_shape"] = list(df.shape)
    return CleanTable(name, task, df, feats, num, cat, target, report)


class TablePreprocessor:
    """Paper-faithful encoder: categorical -> code, every feature MinMax-scaled to [0, 1] and rounded."""

    def __init__(self, num_cols, cat_cols, features, target, task, decimals: int = 2):
        self.num_cols, self.cat_cols, self.features = list(num_cols), list(cat_cols), list(features)
        self.target, self.task, self.decimals = target, task, decimals
        self.cat_vocab: dict[str, list[str]] = {}
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}
        self.classes: list[str] = []

    # ---------------------------------------------------------------- fit / transform
    def fit(self, df: pd.DataFrame) -> "TablePreprocessor":
        for c in self.cat_cols:
            self.cat_vocab[c] = _natural_sort(df[c].astype(str).unique())
        codes = self._codes(df)
        for c in self.features:
            self.mins[c], self.maxs[c] = float(codes[c].min()), float(codes[c].max())
        if self.task == "classification":
            self.classes = _natural_sort(df[self.target].astype(str).unique())
        return self

    def _codes(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.features:
            if c in self.cat_cols:
                lut = {v: i for i, v in enumerate(self.cat_vocab[c])}
                codes = df[c].astype(str).map(lut)
                if codes.isna().any():   # unseen category (only possible for synthetic data)
                    codes = codes.fillna(codes.mode().iloc[0] if codes.notna().any() else 0)
                out[c] = codes.astype(float)
            else:
                out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """clean-space -> encoded-space. Rows whose class label is unknown are dropped."""
        codes = self._codes(df)
        enc = pd.DataFrame(index=df.index)
        for c in self.features:
            rng = self.maxs[c] - self.mins[c]
            enc[c] = ((codes[c] - self.mins[c]) / rng if rng > 0 else codes[c] * 0.0).round(self.decimals)
        if self.target in df.columns:
            if self.task == "classification":
                lut = {v: i for i, v in enumerate(self.classes)}
                enc[TARGET] = df[self.target].astype(str).map(lut)
            else:
                enc[TARGET] = pd.to_numeric(df[self.target], errors="coerce")
            enc = enc.dropna(subset=[TARGET])
            if self.task == "classification":
                enc[TARGET] = enc[TARGET].astype(int)
        return enc.reset_index(drop=True)

    def inverse_transform(self, enc: pd.DataFrame) -> pd.DataFrame:
        """encoded-space -> clean-space (codes rounded to the nearest category)."""
        out = pd.DataFrame(index=enc.index)
        for c in self.features:
            v = enc[c].astype(float) * (self.maxs[c] - self.mins[c]) + self.mins[c]
            if c in self.cat_cols:
                idx = v.round().clip(0, len(self.cat_vocab[c]) - 1).astype(int)
                out[c] = [self.cat_vocab[c][i] for i in idx]
            else:
                out[c] = v
        if TARGET in enc.columns:
            if self.task == "classification":
                idx = enc[TARGET].round().clip(0, len(self.classes) - 1).astype(int)
                out[self.target] = [self.classes[i] for i in idx]
            else:
                out[self.target] = enc[TARGET].astype(float)
        return out

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "TablePreprocessor":
        obj = cls(d["num_cols"], d["cat_cols"], d["features"], d["target"], d["task"], d["decimals"])
        obj.__dict__.update(d)
        return obj


@dataclass
class TabularSplit:
    """One train/test split of one dataset for one seed, in both clean and encoded space."""
    name: str
    task: str
    seed: int
    features: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    target: str                    # original target name (clean space)
    raw_train: pd.DataFrame        # clean space
    raw_test: pd.DataFrame
    train: pd.DataFrame            # encoded space (features + "target")
    test: pd.DataFrame
    pre: TablePreprocessor
    report: dict

    @property
    def n_classes(self) -> int:
        return len(self.pre.classes) if self.task == "classification" else 0

    def summary(self) -> dict:
        d = {"dataset": self.name, "task": self.task, "seed": self.seed, "n_train": len(self.train),
             "n_test": len(self.test), "n_features": len(self.features), "n_num": len(self.num_cols),
             "n_cat": len(self.cat_cols), "n_classes": self.n_classes, "target": self.target}
        if self.task == "classification":
            d["class_names"] = self.pre.classes
            d["train_class_counts"] = self.train[TARGET].value_counts().sort_index().tolist()
        return {**d, **self.report}


def make_split(name: str, seed: int = 0, test_size: float = 0.2, train_size: float = 1.0,
               fit_scaler_on: str = "full", binarize: str | None = None, max_rows: int | None = None,
               decimals: int = 2) -> TabularSplit:
    tbl = clean_table(name, binarize=binarize)
    df = tbl.df
    strat = df[tbl.target] if tbl.task == "classification" else None
    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, random_state=seed, stratify=strat)
        df = df.reset_index(drop=True)
        strat = df[tbl.target] if tbl.task == "classification" else None
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=strat)
    if train_size < 1.0:
        s = tr[tbl.target] if tbl.task == "classification" else None
        tr, _ = train_test_split(tr, train_size=train_size, random_state=seed, stratify=s)
    feats = tbl.features
    pre = TablePreprocessor(tbl.num_cols, tbl.cat_cols, feats, tbl.target, tbl.task, decimals)
    pre.fit(df if fit_scaler_on == "full" else tr)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    return TabularSplit(name, tbl.task, seed, feats, tbl.num_cols, tbl.cat_cols, tbl.target,
                        tr, te, pre.transform(tr), pre.transform(te), pre, tbl.report)
