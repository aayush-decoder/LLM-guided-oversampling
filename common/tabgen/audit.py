# Audits. Each writes a JSON report and prints PASS/WARN/FAIL lines.

#   env   python/torch/CUDA/VRAM, key packages, which registry LLMs fit this GPU
#   data  per dataset: availability, size vs registry expectation, missing values, duplicates,
#         constant columns, class imbalance, train/test overlap, single-feature target leakage
#   run   per finished run dir: required artifacts, synthetic validity (NaN / range / class coverage),
#         memorisation (exact copies of training rows), label-distribution shift, metric sanity
from __future__ import annotations

import importlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data.preprocess import TARGET, make_split
from .data.download import is_available
from .utils import load_json, save_json

REQUIRED_RUN_FILES = ["config.json", "split.json", "preprocessor.json", "synthetic_encoded.csv",
                      "synthetic_clean.csv", "generation_stats.json", "metrics.json"]


class Report:
    def __init__(self, title):
        self.title, self.items = title, []

    def add(self, level, msg):
        self.items.append({"level": level, "msg": msg})
        print(f"  [{level}] {msg}")

    def ok(self, cond, msg_ok, msg_bad, bad="WARN"):
        self.add("PASS" if cond else bad, msg_ok if cond else msg_bad)

    @property
    def status(self):
        lv = [i["level"] for i in self.items]
        return "FAIL" if "FAIL" in lv else "WARN" if "WARN" in lv else "PASS"


def audit_env() -> dict:
    r = Report("env")
    print("== environment audit")
    r.add("INFO", f"python {platform.python_version()} on {platform.system()}")
    for pkg in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "xgboost", "sklearn", "pandas", "sdv"):
        try:
            m = importlib.import_module(pkg)
            r.add("PASS", f"{pkg} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            r.add("WARN" if pkg in ("bitsandbytes", "sdv") else "FAIL", f"{pkg} not importable ({type(e).__name__})")
    import torch
    from .llm.loader import vram_gb
    from .llm.registry import list_llms
    cuda = torch.cuda.is_available()
    r.ok(cuda, f"CUDA available: {torch.cuda.get_device_name(0) if cuda else ''} {vram_gb():.1f} GB",
         "CUDA not available: training runs on CPU (only tiny / distilgpt2 are practical)")
    fits = [s.key for s in list_llms() if s.min_vram_gb <= vram_gb() or s.min_vram_gb == 0]
    r.add("INFO", f"LLMs that fit this GPU: {', '.join(fits)}")
    return {"status": r.status, "items": r.items}


def audit_data(name: str) -> dict:
    from .data.registry import get_spec
    r = Report(name)
    print(f"== data audit: {name}")
    if not is_available(name):
        try:
            from .data.download import fetch
            fetch(name)
        except Exception as e:  # noqa: BLE001
            r.add("FAIL", f"not available: {str(e).splitlines()[0]}")
            return {"dataset": name, "status": r.status, "items": r.items}
    sp = make_split(name, seed=0)
    exp = get_spec(name).expected
    n = len(sp.train) + len(sp.test)
    r.ok(not exp.get("n") or abs(n - exp["n"]) <= 0.02 * exp["n"], f"rows {n} (expected {exp.get('n')})",
         f"rows {n} differ from expected {exp.get('n')}")
    r.ok(not exp.get("m") or len(sp.features) == exp["m"], f"features {len(sp.features)}",
         f"features {len(sp.features)} != expected {exp.get('m')}")
    if sp.task == "classification":
        r.ok(not exp.get("k") or sp.n_classes == exp["k"], f"classes {sp.n_classes}",
             f"classes {sp.n_classes} != expected {exp.get('k')}")
        counts = sp.train[TARGET].value_counts(normalize=True)
        r.ok(counts.min() >= 0.05, f"class balance ok (min {counts.min():.3f})",
             f"imbalanced: smallest class {counts.min():.3f}")
    full = pd.concat([sp.train, sp.test])
    const = [c for c in sp.features if full[c].nunique() <= 1]
    r.ok(not const, "no constant columns", f"constant columns: {const}")
    dup = full.duplicated().mean()
    r.ok(dup < 0.05, f"duplicate rows {dup:.3%}", f"duplicate rows {dup:.3%}")
    overlap = len(pd.merge(sp.train.drop_duplicates(), sp.test.drop_duplicates(), how="inner")) / max(len(sp.test), 1)
    r.ok(overlap < 0.05, f"train/test exact overlap {overlap:.2%}", f"train/test exact overlap {overlap:.2%}")
    if len(sp.train) > 20:
        corr = sp.train[sp.features].corrwith(sp.train[TARGET]).abs()
        leak = corr[corr > 0.98].index.tolist()
        # correlation with the class CODE misses multiclass indicators (cardio: V26-V35 one-hot
        # encode the 10 classes but correlate weakly with the code), so also test each feature
        # as a one-vs-rest score for each class: AUC >= 0.99 means it alone identifies a class
        separators = []
        if sp.task == "classification":
            from sklearn.metrics import roc_auc_score
            y = sp.train[TARGET].to_numpy()
            for c in sp.features:
                x = sp.train[c].to_numpy(float)
                nu = np.unique(x).size
                if nu < 2:
                    continue
                for k in np.unique(y):
                    pos = y == k
                    if 0.01 <= pos.mean() <= 0.99 and max(auc := roc_auc_score(pos, x), 1 - auc) >= 0.99:
                        # a 2-valued column that identifies a class = an indicator of the label;
                        # a continuous one may be genuine signal (iris petal length -> setosa)
                        (leak if nu <= 2 else separators).append(f"{c}->class {k:g}")
                        break
        r.ok(not leak, "no single-feature target leakage (|corr|<=0.98; no binary class indicator)",
             f"possible leakage: {leak}")
        if separators:
            r.add("INFO", f"features that alone separate a class (check they are not label-derived): "
                          f"{separators}")
    r.add("INFO", f"missing before cleaning: {sp.report.get('missing_per_column', {})}")
    return {"dataset": name, "status": r.status, "items": r.items, "summary": sp.summary()}


def audit_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    r = Report(str(run_dir))
    print(f"== run audit: {run_dir}")
    missing = [f for f in REQUIRED_RUN_FILES if not (run_dir / f).exists()]
    r.ok(not missing, "all required artifacts present", f"missing artifacts: {missing}", bad="FAIL")
    if missing:
        return {"run": str(run_dir), "status": r.status, "items": r.items}
    cfg, split_info, m = load_json(run_dir / "config.json"), load_json(run_dir / "split.json"), load_json(run_dir / "metrics.json")
    syn = pd.read_csv(run_dir / "synthetic_encoded.csv")
    r.ok(not syn.isna().any().any(), "no NaN in synthetic data", "NaN values in synthetic data", bad="FAIL")
    n_train = split_info["n_train"]
    r.ok(abs(len(syn) - cfg["data"]["gen_size"] * n_train) <= 1, f"|D_fake| = {len(syn)}",
         f"|D_fake| = {len(syn)} but expected {cfg['data']['gen_size'] * n_train:.0f}")
    feats = [c for c in syn.columns if c != TARGET]
    out_of_range = float(((syn[feats] < -0.05) | (syn[feats] > 1.05)).mean().mean())
    r.ok(out_of_range < 0.01, f"feature values within [0,1] ({out_of_range:.2%} outside)",
         f"{out_of_range:.2%} of feature values outside [0,1]")
    if split_info["task"] == "classification":
        k = split_info["n_classes"]
        present = sorted(syn[TARGET].astype(int).unique().tolist())
        r.ok(present == list(range(k)), f"all {k} classes generated", f"classes generated {present}, expected 0..{k-1}", bad="FAIL")
        real = np.array(split_info["train_class_counts"], float)
        real /= real.sum()
        fake = syn[TARGET].astype(int).value_counts(normalize=True).reindex(range(k), fill_value=0).to_numpy()
        tv = 0.5 * np.abs(real - fake).sum()
        r.ok(tv < 0.15, f"label TV distance {tv:.3f}", f"label distribution shift, TV {tv:.3f}")
    copy = m.get("fidelity", {}).get("exact_copy_rate")
    if copy is not None:
        r.ok(copy < 0.2, f"exact copies of training rows {copy:.2%}", f"memorisation: {copy:.2%} exact copies")
    r.ok(m["tstr"].get("valid", False), f"TSTR valid: {m['tstr']}", f"TSTR invalid: {m['tstr']}", bad="FAIL")
    gen = m.get("generation", {})
    if "validity_rate" in gen:
        r.ok(gen["validity_rate"] > 0.5, f"generation validity {gen['validity_rate']:.2%}",
             f"low generation validity {gen['validity_rate']:.2%}")
    rep = {"run": str(run_dir), "status": r.status, "items": r.items}
    save_json(rep, run_dir / "audit.json")
    return rep
