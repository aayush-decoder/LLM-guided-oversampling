"""Visualise a dataset at each preprocessing stage:
    raw     - as downloaded: missing values per column, dtypes, first features' distributions
    clean   - after column selection / NaN drop: distributions, class balance
    encoded - after encoding + MinMax (what v1's LLM and XGBoost see): distributions, correlation
    v2text  - example serialized rows exactly as the v1 and v2 LLMs see them
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..data.download import read_raw  # noqa: E402
from ..data.preprocess import TARGET, clean_table, make_split  # noqa: E402
from ..paths import REPORTS  # noqa: E402


def _grid(df: pd.DataFrame, cols: list[str], title: str, path: Path, max_cols: int = 16):
    cols = cols[:max_cols]
    n = len(cols)
    nc = min(4, n)
    nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.8 * nr), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
            ax.hist(s.dropna().astype(float), bins=30, color="#4C72B0")
        else:
            vc = s.astype(str).value_counts().head(12)
            ax.barh(vc.index[::-1], vc.values[::-1], color="#55A868")
            ax.tick_params(axis="y", labelsize=7)
        ax.set_title(str(c)[:30], fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_dataset(name: str, stages=("raw", "clean", "encoded"), out: Path | None = None) -> list[Path]:
    out = out or (REPORTS / "data_viz" / name)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if "raw" in stages:
        raw, _ = read_raw(name)
        miss = raw.isna().sum() + (raw.astype(str).apply(lambda s: s.str.strip()) == "?").sum()
        fig, ax = plt.subplots(figsize=(8, max(3, 0.22 * len(miss))))
        ax.barh(miss.index.astype(str), miss.values, color="#C44E52")
        ax.set_title(f"{name} RAW: missing values per column ({raw.shape[0]} rows x {raw.shape[1]} cols)")
        fig.tight_layout()
        fig.savefig(out / "raw_missing.png", dpi=110)
        plt.close(fig)
        _grid(raw, list(raw.columns), f"{name} RAW distributions", out / "raw_distributions.png")
        written += [out / "raw_missing.png", out / "raw_distributions.png"]
    if "clean" in stages:
        tbl = clean_table(name)
        _grid(tbl.df, tbl.features + [tbl.target], f"{name} CLEAN ({len(tbl.df)} rows)", out / "clean_distributions.png")
        written.append(out / "clean_distributions.png")
    if "encoded" in stages:
        sp = make_split(name, seed=0)
        enc = pd.concat([sp.train, sp.test])
        _grid(enc, sp.features + [TARGET], f"{name} ENCODED (MinMax, 2 decimals)", out / "encoded_distributions.png")
        corr = enc.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(0.35 * len(corr) + 3, 0.35 * len(corr) + 2))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
        fig.colorbar(im)
        ax.set_title(f"{name} encoded correlation")
        fig.tight_layout()
        fig.savefig(out / "encoded_correlation.png", dpi=110)
        plt.close(fig)
        written += [out / "encoded_distributions.png", out / "encoded_correlation.png"]
        # what the LLM actually reads
        lines = ["# v1 (paper) serialization, encoded space:"]
        for _, r in sp.train.head(3).iterrows():
            lines.append(", ".join(f"{c} is {r[c]:g}" for c in sp.features) + f", target is {r[TARGET]:g}")
        lines.append("\n# v2 serialization, clean space (real categories and units):")
        for _, r in sp.raw_train.head(3).iterrows():
            lines.append(", ".join(f"{c} is {r[c]}" for c in sp.features) + f", {sp.target} is {r[sp.target]}")
        (out / "serialized_examples.txt").write_text("\n".join(lines), encoding="utf-8")
        written.append(out / "serialized_examples.txt")
    return written
