# Collects every metrics.json under <version>/runs and builds Table II / Table III style tables
# (markdown + csv) next to the paper's reported numbers.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import CONFIGS, runs_root
from ..utils import load_json, load_yaml

FIDELITY_KEYS = ("discriminator", "discriminator_lr", "inverse_kl", "density", "coverage", "dcr_mean",
                 "exact_copy_rate")


def collect(versions: list[str], subdir: str | None = None) -> pd.DataFrame:
    rows = []
    for v in versions:
        root = runs_root(v) / subdir if subdir else runs_root(v)
        for p in root.glob("*/*/seed*/metrics.json"):
            m = load_json(p)
            key = "accuracy" if m["task"] == "classification" else "mse"
            row = {"version": v, "dataset": p.parts[-4], "method": m["method"], "llm": m["llm"], "seed": m["seed"],
                   "task": m["task"], "metric": key, "score": m["tstr"].get(key), "original": m["original"].get(key)}
            row.update({k: m.get("fidelity", {}).get(k) for k in FIDELITY_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt(s: pd.Series) -> str:
    return f"{s.mean():.4f} ({s.std(ddof=0):.3f})" if len(s) else ""


def build_tables(versions: list[str], out_dir: Path, subdir: str | None = None) -> dict[str, Path]:
    df = collect(versions, subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return {}
    df.to_csv(out_dir / "all_runs.csv", index=False)
    df["col"] = df["version"] + ":" + df["method"] + "@" + df["llm"]
    ref = load_yaml(CONFIGS / "paper_reference.yaml")
    written = {}
    for metric in ("accuracy", "mse"):
        sub = df[df.metric == metric]
        if sub.empty:
            continue
        table = sub.groupby(["dataset", "col"])["score"].apply(_fmt).unstack("col")
        table.insert(0, "Original(ours)", sub.groupby("dataset")["original"].apply(_fmt))
        for i, name in enumerate(ref["columns"]):
            table[f"paper:{name}"] = [ref[metric].get(d, [np.nan] * 7)[i] for d in table.index]
        means = sub.groupby("col")["score"].mean()
        path = out_dir / f"table_{metric}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# TSTR {metric} (XGBoost)  mean (std) over seeds\n\n")
            f.write(table.to_markdown() + "\n\n## Average over datasets\n\n" + means.to_frame("mean").to_markdown())
        table.to_csv(out_dir / f"table_{metric}.csv")
        written[metric] = path
    fid = df.groupby("col")[list(FIDELITY_KEYS)].mean().round(4)
    note = ("Fidelity (discriminator, discriminator_lr, inverse_kl, density, coverage) is measured against "
            "the HELD-OUT test rows; privacy (dcr_mean, exact_copy_rate) against the training rows. "
            "Discriminators: lower is better.\n\n")
    (out_dir / "fidelity.md").write_text("# Quality/diversity (Fig. 5)\n\n" + note + fid.to_markdown(),
                                         encoding="utf-8")
    written["fidelity"] = out_dir / "fidelity.md"
    return written
