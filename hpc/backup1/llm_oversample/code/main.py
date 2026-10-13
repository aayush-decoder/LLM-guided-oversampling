#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImbLLM (+ importance-aware fix_y, "IAFP") benchmark over every classification dataset in data/raw.

For each dataset and each seed:
  * stratified 80/20 split -> the TEST SET is created once and is shared by every method
  * minority class of the TRAIN part is reduced to q=0.2 (the paper's setting); the removed
    minority rows are only used as an unseen *reference* for synthetic-quality metrics
  * methods: Original, Imbalanced, RandomOverSampler, SMOTE / ADASYN (numeric-only datasets),
    SMOTE-NC (mixed datasets), ImbLLM-random (paper's random fix_y) and ImbLLM-IAFP (proposal)
  * XGBoost classifier, metrics on the identical test set

Everything is resumable: finished stages leave artifacts on disk and are skipped on re-run, and every
dataset directory is self-contained (summary, charts, logs), so a job killed at dataset 6 leaves
datasets 1-5 fully finished.

Usage examples
  python main.py --data_dir data/raw --run_name exp1
  python main.py --smoke                      # tiny/fast sanity run
  python main.py --list                       # show which datasets would run / be skipped
  python main.py --aggregate_only --run_name exp1
"""
import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler


# =====================================================================================
# GLOBAL CONFIG  (edit here, or override a few of these from the command line)
# =====================================================================================
class C:
    SEEDS = [42, 43, 44]            # each seed = its own split, reduction, fine-tuning
    Q = 0.20                        # fraction of the TRAIN minority rows kept (paper: q = 0.2)
    TEST_SIZE = 0.20                # stratified 80/20 split
    THRESHOLD = 0.5                 # decision threshold for all hard-label metrics

    # --- LLM (paper: distilgpt2, batch 32, 50 epochs, temperature 0.7) ---
    LLM_MODEL_NAME = "distilgpt2"   # HF hub name or local path (--model_name)
    LLM_EPOCHS = 5                  # paper uses 50; kept small for HPC time/GPU limits
    LLM_BATCH_SIZE = 32             # effective batch size (OOM -> micro-batches + grad accumulation)
    LLM_LR = 5e-5
    LLM_WEIGHT_DECAY = 0.01
    LLM_WARMUP_FRAC = 0.05
    TEMPERATURE = 0.7
    TOP_P = 0.95
    GEN_BATCH_SIZE = 128            # halves automatically on CUDA OOM
    VALUE_SIG_DIGITS = 5            # numbers are written with this many significant digits
    INTERP_FRACTION = 0.25          # interpolated rows = 25% of |majority| ...
    MAX_INTERP_ROWS = 5000          # ... capped (time)
    MIN_INTERP_ROWS = 100
    MAX_SYNTH_ROWS = 20000          # cap on generated rows (time). Larger need => partial balancing (logged)
    IAFP_ORDER_TEMPERATURE = 0.25   # lower = stricter "important features first"
    LABEL_TOKEN = "target_class"    # the label is always written first: "target_class is 1, f1 is ..."

    # --- dataset eligibility ---
    CLASSIFICATION_MAX_UNIQUE = 20  # numeric target with more unique values => regression => skipped
    MIN_MINORITY_TOTAL = 15         # fewer minority rows than this => skipped
    KNOWN_REGRESSION = {"california", "housing", "insurance", "gem_price", "fuel"}

    # --- plots ---
    TSNE_MAX_MAJORITY = 1500
    TSNE_MAX_PER_GROUP = 1000

    CAT_MISSING = "__missing__"
    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))


METRICS = ["f1", "accuracy", "g_score", "roc_auc", "pr_auc", "precision", "recall"]
METHOD_ORDER = ["Original", "Imbalanced", "RandomOverSampler", "SMOTE", "ADASYN", "SMOTE-NC",
                "ImbLLM-random", "ImbLLM-IAFP"]
SEED_STAGES = ["split", "importance"] + METHOD_ORDER + ["plots"]
METHOD_COLORS = {"Original": "#7f7f7f", "Imbalanced": "#bcbd22", "RandomOverSampler": "#1f77b4",
                 "SMOTE": "#17becf", "ADASYN": "#9467bd", "SMOTE-NC": "#8c564b",
                 "ImbLLM-random": "#ff7f0e", "ImbLLM-IAFP": "#2ca02c"}

log = logging.getLogger("imbllm")


# =====================================================================================
# logging / status helpers
# =====================================================================================
def setup_logging(run_dir: Path):
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(run_dir / "main.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)


@contextlib.contextmanager
def file_logging(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)
    try:
        yield
    finally:
        log.removeHandler(fh)
        fh.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    return str(o)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def fmt_secs(s):
    s = int(max(0, s))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


TAG = {"pending": "[PEND]", "running": "[RUN ]", "done": "[DONE]", "failed": "[FAIL]",
       "skipped": "[SKIP]", "interrupted": "[INTR]", "partial": "[PART]"}


class Interrupted(BaseException):
    """Raised from the SIGTERM/SIGUSR1 handler (Slurm time limit / scancel)."""


class StatusBoard:
    """Run-level status of every dataset / seed / stage. Written to status.json + status.txt."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "status.json"
        self.d = {"run": run_dir.name, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
        if self.path.exists():
            try:
                old = json.load(open(self.path))
                self.d["datasets"] = old.get("datasets", {})
                for ds in self.d["datasets"].values():       # leftovers of a killed job
                    if ds.get("state") == "running":
                        ds["state"] = "interrupted"
                    for sd in ds.get("seeds", {}).values():
                        if sd.get("state") == "running":
                            sd["state"] = "interrupted"
                        for st in sd.get("stages", {}).values():
                            if st.get("state") == "running":
                                st["state"] = "interrupted"
            except Exception:
                pass
        self.current = None  # (ds, seed, stage)

    # -- mutation ---------------------------------------------------------------
    def init_dataset(self, name, order, n_rows, state="pending", reason=""):
        ds = self.d["datasets"].setdefault(name, {"seeds": {}})
        ds.update({"order": order, "n_rows": n_rows, "reason": reason})
        if state == "skipped" or ds.get("state") in (None, "pending", "skipped"):
            ds["state"] = state

    def set_ds(self, name, state, reason=None):
        ds = self.d["datasets"][name]
        ds["state"] = state
        if reason is not None:
            ds["reason"] = reason
        self.save()

    def plan_seed(self, name, seed):
        ds = self.d["datasets"][name]
        sd = ds["seeds"].setdefault(str(seed), {"stages": {}})
        sd["state"] = "running"
        for s in SEED_STAGES:
            sd["stages"][s] = {"state": "pending"}
        self.save()

    def set_stage(self, name, seed, stage, state, **info):
        sd = self.d["datasets"][name]["seeds"][str(seed)]
        sd["stages"][stage] = {"state": state, **info}
        self.current = (name, seed, stage) if state == "running" else self.current
        self.save()

    def finish_seed(self, name, seed):
        sd = self.d["datasets"][name]["seeds"][str(seed)]
        states = [v["state"] for v in sd["stages"].values()]
        sd["state"] = "partial" if "failed" in states else "done"
        self.save()

    # -- views ------------------------------------------------------------------
    def counts(self):
        c = {}
        for ds in self.d["datasets"].values():
            c[ds["state"]] = c.get(ds["state"], 0) + 1
        return c

    def render(self):
        lines = []
        c = self.counts()
        n = len(self.d["datasets"])
        lines.append(f"STATUS  datasets: {n} total | " + " | ".join(f"{k}={v}" for k, v in sorted(c.items())))
        for name, ds in sorted(self.d["datasets"].items(), key=lambda kv: kv[1].get("order", 1e9)):
            seeds = ds.get("seeds", {})
            done = sum(1 for s in seeds.values() if s.get("state") == "done")
            extra = ""
            if ds["state"] in ("running", "interrupted", "partial"):
                for sk, sv in seeds.items():
                    if sv.get("state") in ("running", "interrupted", "partial"):
                        stg = [f"{k}:{v['state']}" for k, v in sv["stages"].items()
                               if v["state"] in ("running", "failed", "interrupted")]
                        extra += f" seed {sk}" + (f" [{', '.join(stg)}]" if stg else "")
            reason = f"  ({ds['reason']})" if ds.get("reason") else ""
            lines.append(f"  {TAG.get(ds['state'], '[????]')} {name:<16} seeds_done={done}/{len(C.SEEDS)}{extra}{reason}")
        return "\n".join(lines)

    def save(self):
        self.d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(self.path, self.d)
        try:
            (self.run_dir / "status.txt").write_text(self.render() + "\n", encoding="utf-8")
        except Exception:
            pass

    def mark_interrupted(self):
        for ds in self.d["datasets"].values():
            if ds.get("state") == "running":
                ds["state"] = "interrupted"
            for sd in ds.get("seeds", {}).values():
                if sd.get("state") == "running":
                    sd["state"] = "interrupted"
                for st in sd.get("stages", {}).values():
                    if st.get("state") == "running":
                        st["state"] = "interrupted"
        self.save()


@contextlib.contextmanager
def stage(board, ds, seed, name, progress=""):
    """Run one resumable stage. Exceptions are logged and recorded, never fatal for the whole job."""
    board.set_stage(ds, seed, name, "running")
    log.info("%s | seed %s | stage %-18s | RUNNING %s", ds, seed, name, progress)
    t0 = time.time()
    try:
        yield
        board.set_stage(ds, seed, name, "done", seconds=round(time.time() - t0, 1))
        log.info("%s | seed %s | stage %-18s | DONE in %s", ds, seed, name, fmt_secs(time.time() - t0))
    except Exception as e:  # noqa
        log.error("%s | seed %s | stage %-18s | FAILED after %s: %s\n%s", ds, seed, name,
                  fmt_secs(time.time() - t0), e, traceback.format_exc())
        board.set_stage(ds, seed, name, "failed", error=f"{type(e).__name__}: {e}",
                        seconds=round(time.time() - t0, 1))


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =====================================================================================
# dataset discovery / loading
# =====================================================================================
class DatasetSpec:
    def __init__(self, name, csv, meta_path):
        self.name, self.csv, self.meta_path = name, csv, meta_path
        self.n_rows = 0
        self.skip_reason = None
        self.meta = {}


def inspect_target(y: pd.Series, meta: dict):
    y = y.dropna()
    if meta.get("task", "").lower().startswith("reg"):
        return "meta says regression"
    n = y.nunique()
    if n < 2:
        return "target has a single class"
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        if n > C.CLASSIFICATION_MAX_UNIQUE:
            return f"target looks continuous ({n} unique values) -> regression"
    elif n > 50:
        return f"target has {n} unique string values"
    vc = y.value_counts()
    if vc.min() < C.MIN_MINORITY_TOTAL:
        return f"smallest class has only {vc.min()} rows (< {C.MIN_MINORITY_TOTAL})"
    return None


def discover_datasets(data_dir: Path, include, exclude):
    specs = []
    for csv in sorted(data_dir.glob("*.csv")):
        name = csv.stem
        if include and name not in include:
            continue
        spec = DatasetSpec(name, csv, csv.with_suffix(".meta.json"))
        if name in exclude:
            spec.skip_reason = "excluded by --exclude"
        elif name in C.KNOWN_REGRESSION:
            spec.skip_reason = "known regression dataset"
        elif not spec.meta_path.exists():
            spec.skip_reason = "no .meta.json next to the csv"
        else:
            try:
                spec.meta = json.load(open(spec.meta_path))
                df = pd.read_csv(csv, usecols=lambda c: True)
                spec.n_rows = len(df)
                tgt = spec.meta.get("target")
                if tgt not in df.columns:
                    spec.skip_reason = f"target column {tgt!r} not in csv"
                else:
                    spec.skip_reason = inspect_target(df[tgt], spec.meta)
            except Exception as e:
                spec.skip_reason = f"unreadable: {type(e).__name__}: {e}"
        specs.append(spec)
    return specs


class DatasetData:
    pass


def load_dataset(spec: DatasetSpec):
    """Returns DatasetData with: df (features only), y (binary np array, 1 = minority), schema + info."""
    meta = spec.meta
    raw = pd.read_csv(spec.csv)
    tgt = meta["target"]
    raw = raw[raw[tgt].notna()].reset_index(drop=True)

    num_cols = [c for c in meta.get("numerical", []) if c in raw.columns and c != tgt]
    cat_cols = [c for c in meta.get("categorical", []) if c in raw.columns and c != tgt]
    if not num_cols and not cat_cols:   # no schema in meta -> infer
        for c in raw.columns:
            if c == tgt:
                continue
            (num_cols if pd.api.types.is_numeric_dtype(raw[c]) else cat_cols).append(c)
    dropped = [c for c in raw.columns if c not in num_cols + cat_cols + [tgt]]
    missing_meta = [c for c in meta.get("numerical", []) + meta.get("categorical", []) if c not in raw.columns]

    vc = raw[tgt].astype(str).value_counts()
    minority_label = sorted(vc[vc == vc.min()].index)[0]       # deterministic tie-break
    y = (raw[tgt].astype(str) == minority_label).astype(int).to_numpy()

    d = DatasetData()
    d.name = spec.name
    d.df = raw[num_cols + cat_cols].copy()
    d.y = y
    d.num_cols, d.cat_cols = num_cols, cat_cols
    d.features = num_cols + cat_cols
    d.label_token = C.LABEL_TOKEN
    while d.label_token in d.features:
        d.label_token += "_"
    d.info = {
        "dataset": spec.name, "rows": int(len(raw)), "n_numerical": len(num_cols),
        "n_categorical": len(cat_cols), "target": tgt, "class_counts": vc.to_dict(),
        "n_classes": int(len(vc)), "minority_label": minority_label,
        "minority_count": int(y.sum()), "majority_count": int((1 - y).sum()),
        "minority_fraction": float(y.mean()),
        "multiclass_merged": bool(len(vc) > 2),
        "csv_columns_not_in_meta_schema(ignored)": dropped,
        "meta_columns_missing_in_csv": missing_meta,
    }
    return d


# =====================================================================================
# preprocessing (fitted ONLY on the reduced training data)
# =====================================================================================
class Prep:
    def __init__(self, num_cols, cat_cols, fit_df):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.features = self.num_cols + self.cat_cols
        num = fit_df[self.num_cols].apply(pd.to_numeric, errors="coerce") if self.num_cols else pd.DataFrame(index=fit_df.index)
        self.medians = num.median().fillna(0.0) if self.num_cols else pd.Series(dtype=float)
        self.lo = num.min().fillna(0.0) if self.num_cols else pd.Series(dtype=float)
        self.hi = num.max().fillna(0.0) if self.num_cols else pd.Series(dtype=float)
        self.int_cols = [c for c in self.num_cols
                         if num[c].dropna().size and np.all(np.isclose(np.mod(num[c].dropna().to_numpy(), 1.0), 0.0))]
        self.categories, self.maps = {}, {}
        for c in self.cat_cols:
            s = self._cat(fit_df[c])
            cats = sorted(s.unique())
            self.categories[c] = cats
            self.maps[c] = {v: i for i, v in enumerate(cats)}
        self.cat_idx = list(range(len(self.num_cols), len(self.features)))

    @staticmethod
    def _cat(s):
        return s.where(s.notna(), C.CAT_MISSING).astype(str)

    def clean(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians[c]).astype(float)
        for c in self.cat_cols:
            out[c] = self._cat(df[c])
        return out[self.features]

    def encode(self, clean_df):
        cols = []
        for c in self.num_cols:
            cols.append(clean_df[c].to_numpy(dtype=float))
        for c in self.cat_cols:
            cols.append(clean_df[c].map(self.maps[c]).fillna(-1).to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.zeros((len(clean_df), 0))


# =====================================================================================
# classifier + metrics
# =====================================================================================
def make_classifier(seed):
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, eval_metric="logloss", random_state=seed,
                         n_jobs=C.N_CPUS, tree_method="hist", verbosity=0)


def compute_metrics(y_true, prob):
    pred = (prob >= C.THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    m = {
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "g_score": math.sqrt(sens * spec),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": spec,
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y_true, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def evaluate_table(name, Xtr, ytr, Xte, yte, seed, test_fp, out_path, dataset, extra=None):
    t0 = time.time()
    clf = make_classifier(seed)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    m = compute_metrics(yte, prob)
    res = {"dataset": dataset, "seed": seed, "method": name, **m,
           "n_train": int(len(ytr)), "train_minority": int(ytr.sum()), "train_majority": int((1 - ytr).sum()),
           "train_minority_fraction": float(ytr.mean()), "n_test": int(len(yte)), "test_minority": int(yte.sum()),
           "test_fingerprint": test_fp, "threshold": C.THRESHOLD,
           "eval_seconds": round(time.time() - t0, 1)}
    if extra:
        res.update(extra)
    write_json(out_path, res)
    log.info("%s | seed %s | %-18s | train=%d (min %d / maj %d) | F1=%.4f Acc=%.4f G=%.4f AUC=%.4f PR-AUC=%.4f",
             dataset, seed, name, len(ytr), ytr.sum(), (1 - ytr).sum(), m["f1"], m["accuracy"],
             m["g_score"], m["roc_auc"], m["pr_auc"])
    return res


# =====================================================================================
# feature importance (IAFP)
# =====================================================================================
def compute_importance(X, y, features, seed):
    scores = None
    try:
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        clf = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                            eval_metric="logloss", random_state=seed, n_jobs=C.N_CPUS, tree_method="hist", verbosity=0)
        clf.fit(Xa, ya)
        pr = permutation_importance(clf, Xb, yb, scoring="average_precision", n_repeats=5,
                                    random_state=seed, n_jobs=C.N_CPUS)
        scores = pd.Series(np.maximum(pr.importances_mean, 0.0), index=features)
        if float(scores.sum()) <= 1e-12:
            gain = clf.get_booster().get_score(importance_type="gain")
            scores = pd.Series([max(float(gain.get(f"f{i}", 0.0)), 0.0) for i in range(len(features))], index=features)
    except Exception as e:
        log.warning("importance estimation failed (%s) -> uniform order", e)
    if scores is None or float(scores.sum()) <= 1e-12:
        scores = pd.Series(1.0, index=features)
    return scores.sort_values(ascending=False)


def order_weights(imp: pd.Series, features):
    v = np.array([max(float(imp.get(f, 0.0)), 0.0) for f in features], dtype=float)
    if not np.any(v > 0):
        return np.ones(len(features))
    v = v / (v.max() + 1e-12)
    logits = v / max(C.IAFP_ORDER_TEMPERATURE, 1e-3)
    return np.exp(logits - logits.max())


def random_order(rng, n, weights=None):
    """Permutation of range(n). weights=None -> uniform (paper); else weighted-without-replacement (IAFP)."""
    if weights is None:
        return rng.permutation(n)
    keys = -np.log(rng.random(n)) / weights      # Efraimidis-Spirakis
    return np.argsort(keys)


# =====================================================================================
# text serialization / parsing
# =====================================================================================
def fmt_value(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        f = float(v)
        if np.isfinite(f) and f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return f"{f:.{C.VALUE_SIG_DIGITS}g}"
    return str(v)


def make_parser(features, num_cols, cat_categories):
    alt = "|".join(re.escape(f) for f in sorted(features, key=len, reverse=True))
    pat = re.compile(r"(?:^|,\s*)(" + alt + r") is ")
    numpat = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    num_set = set(num_cols)

    def parse(text):
        ms = list(pat.finditer(text))
        out = {}
        for k, m in enumerate(ms):
            name = m.group(1)
            if name in out:
                continue
            end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
            val = text[m.end():end].strip().rstrip(",").strip()
            if name in num_set:
                mm = numpat.match(val)
                if mm:
                    try:
                        out[name] = float(mm.group(0))
                    except ValueError:
                        pass
            elif val in cat_categories[name]:
                out[name] = val
        return out
    return parse


class PermutedTextDataset:
    """Map-style dataset. Every access builds the sentence with a NEW feature permutation (label first)."""

    def __init__(self, rows_parts, label_part, weights, tokenizer, rng, max_len=None):
        self.rows, self.label_part, self.weights = rows_parts, label_part, weights
        self.tok, self.rng, self.max_len = tokenizer, rng, max_len
        self.n_feat = len(rows_parts[0]) if rows_parts else 0

    def __len__(self):
        return len(self.rows)

    def text(self, i, order=None):
        if order is None:
            order = random_order(self.rng, self.n_feat, self.weights)
        parts = self.rows[i]
        return ", ".join([self.label_part] + [parts[j] for j in order])

    def __getitem__(self, i):
        ids = self.tok(self.text(i), truncation=True, max_length=(self.max_len or 1024) - 1)["input_ids"]
        return ids + [self.tok.eos_token_id]


def rows_to_parts(clean_df, features):
    cols = [clean_df[c].to_numpy() for c in features]
    return [[f"{f} is {fmt_value(col[i])}" for f, col in zip(features, cols)] for i in range(len(clean_df))]


def make_interpolated(minority_clean, prep, n, seed):
    """Paper's D_inter: random pairs of retained minority rows. Numeric -> linear interpolation,
    categorical -> value of one parent (chosen by the interpolation weight)."""
    rng = np.random.default_rng(seed)
    m = len(minority_clean)
    i = rng.integers(0, m, size=n)
    j = (i + rng.integers(1, m, size=n)) % m          # j != i
    eps = rng.random(n)
    out = {}
    for c in prep.num_cols:
        a, b = minority_clean[c].to_numpy(float)[i], minority_clean[c].to_numpy(float)[j]
        v = a + eps * (b - a)
        if c in prep.int_cols:
            v = np.round(v)
        out[c] = v
    for c in prep.cat_cols:
        a, b = minority_clean[c].to_numpy()[i], minority_clean[c].to_numpy()[j]
        out[c] = np.where(eps < 0.5, a, b)
    return pd.DataFrame(out, columns=prep.features)


# =====================================================================================
# LLM: fine-tuning + generation
# =====================================================================================
def _collate(batch_ids, pad_id, torch):
    B, L = len(batch_ids), max(len(x) for x in batch_ids)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    for k, x in enumerate(batch_ids):
        t = torch.tensor(x, dtype=torch.long)
        ids[k, :len(x)], att[k, :len(x)], lab[k, :len(x)] = t, 1, t     # EOS keeps its label (stop learning)
    return ids, att, lab


def _is_oom(e):
    return "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"


def finetune_llm(model_name, rows_parts, label_part, weights, seed, epochs, out_dir: Path, tag):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_pos = getattr(AutoConfig.from_pretrained(model_name), "n_positions", 1024)

    rng = np.random.default_rng(seed)
    ds = PermutedTextDataset(rows_parts, label_part, weights, tok, rng)
    probe = [len(tok(ds.text(i, np.arange(ds.n_feat)))["input_ids"]) for i in range(min(len(ds), 300))]
    ds.max_len = int(min(n_pos, max(probe) * 1.15 + 8))
    log.info("[%s] fine-tuning rows=%d | tokens/row: mean=%.0f max(probe)=%d -> max_len=%d | device=%s",
             tag, len(ds), float(np.mean(probe)), max(probe), ds.max_len, device)
    if max(probe) * 1.15 + 8 > n_pos:
        log.warning("[%s] rows may exceed the model context (%d) and get truncated", tag, n_pos)

    use_cuda = device == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    micro_candidates = [b for b in [C.LLM_BATCH_SIZE, 16, 8, 4, 2, 1] if b <= C.LLM_BATCH_SIZE]
    last_err = None
    for mb in micro_candidates:
        try:
            set_all_seeds(seed)
            rng = np.random.default_rng(seed)
            ds.rng = rng
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            model.config.pad_token_id = tok.pad_token_id
            model.train()
            accum = int(math.ceil(C.LLM_BATCH_SIZE / mb))
            steps_per_epoch = int(math.ceil(len(ds) / C.LLM_BATCH_SIZE))
            total_steps = steps_per_epoch * epochs
            opt = torch.optim.AdamW(model.parameters(), lr=C.LLM_LR, weight_decay=C.LLM_WEIGHT_DECAY)
            sched = get_linear_schedule_with_warmup(opt, int(C.LLM_WARMUP_FRAC * total_steps), total_steps)
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
            log.info("[%s] training: epochs=%d | steps/epoch=%d | micro-batch=%d x accum=%d (effective %d) | amp=%s | lr=%g",
                     tag, epochs, steps_per_epoch, mb, accum, C.LLM_BATCH_SIZE,
                     "bf16" if use_bf16 else ("fp16" if use_fp16 else "off"), C.LLM_LR)
            step_rows, epoch_rows, t_start, step = [], [], time.time(), 0
            log_every = max(1, steps_per_epoch // 4)
            for ep in range(1, epochs + 1):
                t_ep = time.time()
                perm = rng.permutation(len(ds))
                ep_losses = []
                for b in range(steps_per_epoch):
                    idxs = perm[b * C.LLM_BATCH_SIZE:(b + 1) * C.LLM_BATCH_SIZE]
                    if len(idxs) == 0:
                        continue
                    loss_acc = 0.0
                    chunks = [idxs[k:k + mb] for k in range(0, len(idxs), mb)]
                    for ch in chunks:
                        ids, att, lab = _collate([ds[int(i)] for i in ch], tok.pad_token_id, torch)
                        ids, att, lab = ids.to(device), att.to(device), lab.to(device)
                        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                        scaler.scale(loss * (len(ch) / len(idxs))).backward()
                        loss_acc += float(loss.detach()) * (len(ch) / len(idxs))
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    ep_losses.append(loss_acc)
                    step_rows.append({"step": step, "epoch": ep, "loss": loss_acc, "lr": sched.get_last_lr()[0]})
                    if (b + 1) % log_every == 0 and (b + 1) != steps_per_epoch:
                        log.info("[%s] epoch %d/%d | step %d/%d | loss=%.4f", tag, ep, epochs, step, total_steps,
                                 float(np.mean(ep_losses[-log_every:])))
                el = time.time() - t_start
                eta = el / ep * (epochs - ep)
                epoch_rows.append({"epoch": ep, "mean_loss": float(np.mean(ep_losses)),
                                   "last_loss": ep_losses[-1], "epoch_seconds": round(time.time() - t_ep, 1)})
                log.info("[%s] EPOCH %d/%d DONE | mean_loss=%.4f | epoch_time=%s | elapsed=%s | ETA=%s",
                         tag, ep, epochs, epoch_rows[-1]["mean_loss"], fmt_secs(time.time() - t_ep),
                         fmt_secs(el), fmt_secs(eta))
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(step_rows).to_csv(out_dir / "training_loss_steps.csv", index=False)
            pd.DataFrame(epoch_rows).to_csv(out_dir / "training_loss_epochs.csv", index=False)
            model.eval()
            return model, tok, ds.max_len, {"micro_batch": mb, "grad_accum": accum, "epochs": epochs,
                                            "train_seconds": round(time.time() - t_start, 1),
                                            "total_steps": total_steps, "max_len": ds.max_len}
        except Exception as e:      # noqa
            if _is_oom(e) and mb > 1:
                log.warning("[%s] CUDA OOM at micro-batch %d -> retrying with smaller micro-batch", tag, mb)
                last_err = e
                try:
                    del model, opt
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"fine-tuning kept running out of memory: {last_err}")


def run_generation(model, tok, prep, minority_clean, n_target, variant, imp, seed, max_len, label_token, tag):
    """condition_yx generation: 'label is 1, <random feature> is <value>,' then the LLM completes the row."""
    import torch
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    rng = random.Random(seed)
    tok.padding_side = "left"
    features = prep.features
    parse = make_parser(features, prep.num_cols, prep.categories)
    med = minority_clean[prep.num_cols].median() if prep.num_cols else pd.Series(dtype=float)
    modes = {c: minority_clean[c].mode().iloc[0] for c in prep.cat_cols}
    cond_w = None
    if variant == "IAFP":
        cond_w = np.array([max(float(imp.get(f, 0.0)), 0.0) for f in features], dtype=float)
        if not np.any(cond_w > 0):
            cond_w = None
    gen_bs = C.GEN_BATCH_SIZE
    accepted, rejected, n_gen, n_valid = [], [], 0, 0
    t0 = time.time()
    last_log = t0
    rounds = 0
    while len(accepted) < n_target and rounds < 4:
        rounds += 1
        need = n_target - len(accepted)
        to_gen = int(math.ceil(need * (1.10 if rounds == 1 else 1.3)))
        done = 0
        while done < to_gen:
            bs = min(gen_bs, to_gen - done)
            prompts, conds = [], []
            for _ in range(bs):
                f = rng.choices(features, weights=cond_w, k=1)[0] if cond_w is not None else rng.choice(features)
                v = minority_clean[f].iloc[rng.randrange(len(minority_clean))]
                prompts.append(f"{label_token} is 1, {f} is {fmt_value(v)},")
                conds.append((f, v))
            try:
                enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    out = model.generate(**enc, do_sample=True, temperature=C.TEMPERATURE, top_p=C.TOP_P,
                                         max_new_tokens=max_len, pad_token_id=tok.eos_token_id,
                                         eos_token_id=tok.eos_token_id)
            except Exception as e:  # noqa
                if _is_oom(e) and gen_bs > 4:
                    gen_bs = max(4, gen_bs // 2)
                    log.warning("[%s] generation OOM -> batch size %d", tag, gen_bs)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
            texts = tok.batch_decode(out, skip_special_tokens=True)
            for text, (cf, cv) in zip(texts, conds):
                p = parse(text)
                p[cf] = cv if cf in prep.cat_cols else float(cv)
                valid = len(p) >= max(2, int(0.5 * len(features)))
                row = {}
                for c in prep.num_cols:
                    row[c] = p.get(c, np.nan)
                for c in prep.cat_cols:
                    row[c] = p.get(c, np.nan)
                row["_text"] = text
                row["_n_parsed"] = len(p)
                (accepted if valid else rejected).append(row)
                n_valid += int(valid)
            n_gen += bs
            done += bs
            if time.time() - last_log > 30:
                last_log = time.time()
                log.info("[%s] generated %d/%d (round %d) | valid parse %.1f%% | accepted %d/%d | elapsed %s",
                         tag, done, to_gen, rounds, 100.0 * n_valid / max(1, n_gen), len(accepted), n_target,
                         fmt_secs(time.time() - t0))
    rows = accepted[:n_target]
    if len(rows) < n_target:       # last resort: top up with rows that parsed poorly (missing values get imputed)
        log.warning("[%s] only %d valid rows after %d rounds; topping up with %d low-quality rows",
                    tag, len(rows), rounds, min(n_target - len(rows), len(rejected)))
        rows += rejected[:n_target - len(rows)]
    df = pd.DataFrame(rows)
    meta = df[["_text", "_n_parsed"]].copy()
    df = df.drop(columns=["_text", "_n_parsed"])
    # impute omitted values with retained-minority stats; clip numeric to the observed training range
    for c in prep.num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med[c]).clip(prep.lo[c], prep.hi[c])
        if c in prep.int_cols:
            df[c] = df[c].round()
    for c in prep.cat_cols:
        df[c] = df[c].fillna(modes[c]).astype(str)
    df = df[features].reset_index(drop=True)
    stats = {"generated_total": int(n_gen), "valid_parse_rate": float(n_valid / max(1, n_gen)),
             "rounds": rounds, "rows_returned": int(len(df)), "gen_batch_size_final": gen_bs,
             "gen_seconds": round(time.time() - t0, 1)}
    return df, meta, stats


# =====================================================================================
# synthetic-quality metrics + plots
# =====================================================================================
def quality_metrics(ref, syn, train_min, seed):
    rng = np.random.default_rng(seed)

    def cap(a, k=5000):
        return a[rng.choice(len(a), k, replace=False)] if len(a) > k else a
    ref, syn, train_min = cap(ref), cap(syn), cap(train_min)
    scaler = StandardScaler().fit(ref if len(ref) >= 3 else train_min)
    ref_s, syn_s, tr_s = scaler.transform(ref), scaler.transform(syn), scaler.transform(train_min)
    d_ref_syn = NearestNeighbors(n_neighbors=1).fit(syn_s).kneighbors(ref_s)[0].ravel()
    q95 = np.quantile(d_ref_syn, 0.95)
    q95 = q95 if q95 > 0 else 1.0
    out = {"close_probability": float(np.mean(np.clip(d_ref_syn / q95, 0, 1) <= 0.2)),
           "mean_dist_heldout_real_to_nearest_synth": float(d_ref_syn.mean())}
    if len(ref_s) >= 3:
        rad = NearestNeighbors(n_neighbors=3).fit(ref_s).kneighbors(ref_s)[0][:, 2]
        out["coverage_k2"] = float(np.mean(d_ref_syn <= rad))
    d_syn_tr = NearestNeighbors(n_neighbors=1).fit(tr_s).kneighbors(syn_s)[0].ravel()
    out["mean_dist_synth_to_train_minority(DCR)"] = float(d_syn_tr.mean())
    out["frac_synth_exact_copy_of_train_minority"] = float(np.mean(d_syn_tr < 1e-8))
    return out


def make_tsne_plot(path, title, seed, majority, retained, heldout, synth: dict):
    rng = np.random.default_rng(seed)

    def cap(a, k):
        return a[rng.choice(len(a), k, replace=False)] if len(a) > k else a
    groups = [("majority (real)", cap(majority, C.TSNE_MAX_MAJORITY)),
              ("minority train-retained (real)", retained),
              ("minority held-out (real, unseen by generator)", cap(heldout, C.TSNE_MAX_PER_GROUP))]
    for k, v in synth.items():
        groups.append((f"synthetic minority: {k}", cap(v, C.TSNE_MAX_PER_GROUP)))
    allx = np.vstack([g[1] for g in groups if len(g[1])])
    z = StandardScaler().fit_transform(allx)
    if z.shape[1] > 30:
        z = PCA(n_components=30, random_state=seed).fit_transform(z)
    perp = max(5, min(30, (len(z) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed, n_jobs=C.N_CPUS).fit_transform(z)

    style = {
        "majority (real)": dict(c="#bbbbbb", marker=".", s=8, alpha=0.35, lw=0),
        "minority train-retained (real)": dict(c="#1f4e9c", marker="o", s=55, alpha=0.95, edgecolors="black", lw=0.6),
        "minority held-out (real, unseen by generator)": dict(c="#4fc3f7", marker="o", s=28, alpha=0.8, edgecolors="#1f4e9c", lw=0.5),
        "synthetic minority: ImbLLM-random": dict(c="#ff7f0e", marker="x", s=28, alpha=0.8, lw=1.2),
        "synthetic minority: ImbLLM-IAFP": dict(c="#2ca02c", marker="x", s=28, alpha=0.8, lw=1.2),
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = 0
    for name, arr in groups:
        if not len(arr):
            continue
        e = emb[pos:pos + len(arr)]
        pos += len(arr)
        kw = dict(style.get(name, dict(c="k", marker="x", s=20)))
        ax.scatter(e[:, 0], e[:, 1], label=f"{name} (n={len(arr)})", **kw)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_loss(sdir: Path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    drawn = False
    for variant, col in [("random", METHOD_COLORS["ImbLLM-random"]), ("IAFP", METHOD_COLORS["ImbLLM-IAFP"])]:
        sp, ep = sdir / f"llm_{variant}" / "training_loss_steps.csv", sdir / f"llm_{variant}" / "training_loss_epochs.csv"
        if sp.exists():
            d = pd.read_csv(sp)
            axes[0].plot(d["step"], d["loss"], color=col, alpha=0.8, label=f"ImbLLM-{variant}")
            drawn = True
        if ep.exists():
            d = pd.read_csv(ep)
            axes[1].plot(d["epoch"], d["mean_loss"], marker="o", color=col, label=f"ImbLLM-{variant}")
    if not drawn:
        plt.close(fig)
        return
    axes[0].set(title="loss per optimizer step", xlabel="step", ylabel="loss")
    axes[1].set(title="mean loss per epoch", xlabel="epoch", ylabel="loss")
    for a in axes:
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(sdir / "llm_training_loss.png", dpi=140)
    plt.close(fig)


# =====================================================================================
# per-seed pipeline
# =====================================================================================
def read_synth(path: Path, prep):
    d = pd.read_csv(path, dtype={c: str for c in prep.cat_cols}, keep_default_na=False)
    for c in prep.num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[prep.features]


def run_seed(D, spec, seed, seed_pos, ds_pos, ds_total, ds_dir, board, args, llm_ok, llm_skip_reason):
    sdir = ds_dir / f"seed_{seed}"
    mdir, syndir = sdir / "metrics", sdir / "synthetic"
    mdir.mkdir(parents=True, exist_ok=True)
    syndir.mkdir(parents=True, exist_ok=True)
    name = D.name
    tag = f"[{ds_pos}/{ds_total}] {name} seed {seed} ({seed_pos}/{len(C.SEEDS)})"

    with file_logging(sdir / "train.log"):
        log.info("=" * 100)
        log.info("%s | START", tag)
        board.plan_seed(name, seed)

        # ---------------- split / reduction (deterministic -> recomputed on resume) ----------------
        with stage(board, name, seed, "split", tag):
            idx = np.arange(len(D.df))
            tr_idx, te_idx = train_test_split(idx, test_size=C.TEST_SIZE, stratify=D.y, random_state=seed)
            assert len(set(tr_idx) & set(te_idx)) == 0, "train/test overlap!"
            y_tr, y_te = D.y[tr_idx], D.y[te_idx]
            rng = np.random.default_rng(seed)
            min_pos = np.where(y_tr == 1)[0]
            maj_pos = np.where(y_tr == 0)[0]
            n_keep = max(2, int(round(C.Q * len(min_pos))))
            keep_min = rng.choice(min_pos, size=n_keep, replace=False)
            ref_pos = np.setdiff1d(min_pos, keep_min)
            imb_pos = rng.permutation(np.concatenate([maj_pos, keep_min]))
            full_train = D.df.iloc[tr_idx].reset_index(drop=True)
            test_df = D.df.iloc[te_idx].reset_index(drop=True)
            imb_df = full_train.iloc[imb_pos].reset_index(drop=True)
            y_imb = y_tr[imb_pos]
            ref_df = full_train.iloc[ref_pos].reset_index(drop=True)
            test_fp = hashlib.sha1(np.sort(te_idx).tobytes()).hexdigest()[:12]

            prep = Prep(D.num_cols, D.cat_cols, imb_df)       # fitted on REDUCED TRAIN ONLY (no test, no removed rows)
            Xte = prep.encode(prep.clean(test_df))
            X_full = prep.encode(prep.clean(full_train))
            X_imb = prep.encode(prep.clean(imb_df))
            X_ref = prep.encode(prep.clean(ref_df))
            X_maj = X_imb[y_imb == 0]
            X_ret = X_imb[y_imb == 1]
            n_major, n_minor = int((y_imb == 0).sum()), int((y_imb == 1).sum())
            split_info = {
                "seed": seed, "train_rows": int(len(tr_idx)), "test_rows": int(len(te_idx)),
                "test_fingerprint(sha1 of test row ids)": test_fp,
                "full_train_minority": int(len(min_pos)), "full_train_majority": int(len(maj_pos)),
                "retained_minority(q=%.2f)" % C.Q: int(n_keep), "heldout_reference_minority": int(len(ref_pos)),
                "reduced_train_imbalance_ratio(maj:min)": round(n_major / max(1, n_minor), 2),
                "full_train_imbalance_ratio(maj:min)": round(len(maj_pos) / max(1, len(min_pos)), 2),
                "test_minority": int(y_te.sum()), "test_majority": int((1 - y_te).sum()),
            }
            write_json(sdir / "split_info.json", split_info)
            log.info("%s | split: train=%d test=%d | full-train minority=%d -> retained=%d (held-out ref=%d) | "
                     "reduced imbalance %.1f:1 | test minority=%d | test_fingerprint=%s",
                     tag, len(tr_idx), len(te_idx), len(min_pos), n_keep, len(ref_pos), n_major / max(1, n_minor),
                     int(y_te.sum()), test_fp)
        if board.d["datasets"][name]["seeds"][str(seed)]["stages"]["split"]["state"] != "done":
            board.finish_seed(name, seed)
            return

        # ---------------- importance (needed by IAFP) ----------------
        imp_path = sdir / "feature_importance.csv"
        if imp_path.exists() and not args.force:
            imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0]
            board.set_stage(name, seed, "importance", "done", cached=True)
            log.info("%s | stage importance | cached", tag)
        else:
            with stage(board, name, seed, "importance", tag):
                imp = compute_importance(X_imb, y_imb, prep.features, seed)
                imp.to_frame("importance").to_csv(imp_path)
                log.info("%s | top features: %s", tag, ", ".join(f"{k}={v:.4f}" for k, v in imp.head(5).items()))
            if not imp_path.exists():
                imp = pd.Series(1.0, index=prep.features)

        def metrics_path(m):
            return mdir / f"{m}.json"

        def cached(m):
            if metrics_path(m).exists() and not args.force:
                board.set_stage(name, seed, m, "done", cached=True)
                log.info("%s | stage %-18s | cached (metrics.json exists)", tag, m)
                return True
            return False

        def ev(m, Xtr, ytr, extra=None):
            return evaluate_table(m, Xtr, ytr, Xte, y_te, seed, test_fp, metrics_path(m), name, extra)

        # ---------------- baselines ----------------
        if not cached("Original"):
            with stage(board, name, seed, "Original", tag):
                ev("Original", X_full, y_tr, {"note": "all minority rows of the train split (reference upper bound)"})
        if not cached("Imbalanced"):
            with stage(board, name, seed, "Imbalanced", tag):
                ev("Imbalanced", X_imb, y_imb)
        if not cached("RandomOverSampler"):
            with stage(board, name, seed, "RandomOverSampler", tag):
                Xr, yr = RandomOverSampler(random_state=seed).fit_resample(X_imb, y_imb)
                ev("RandomOverSampler", Xr, yr)

        k = max(1, min(5, n_minor - 1))
        has_cat = len(D.cat_cols) > 0
        for m, applicable, builder in [
            ("SMOTE", not has_cat, lambda: SMOTE(random_state=seed, k_neighbors=k)),
            ("ADASYN", not has_cat, lambda: ADASYN(random_state=seed, n_neighbors=k)),
            ("SMOTE-NC", has_cat, lambda: SMOTENC(categorical_features=prep.cat_idx, random_state=seed, k_neighbors=k)),
        ]:
            if not applicable:
                why = "dataset has categorical features (SMOTE/ADASYN need numeric-only)" if m != "SMOTE-NC" \
                    else "dataset is numeric-only (SMOTE-NC is for mixed data)"
                board.set_stage(name, seed, m, "skipped", reason=why)
                log.info("%s | stage %-18s | SKIPPED: %s", tag, m, why)
                continue
            if cached(m):
                continue
            with stage(board, name, seed, m, tag):
                Xr, yr = builder().fit_resample(X_imb, y_imb)
                if m == "SMOTE-NC":
                    Xr = Xr.copy()
                    Xr[:, prep.cat_idx] = np.round(Xr[:, prep.cat_idx])
                ev(m, Xr, yr)

        # ---------------- ImbLLM (random fix_y  vs  importance-aware fix_y) ----------------
        n_needed_full = max(0, n_major - n_minor)
        n_synth = min(n_needed_full, C.MAX_SYNTH_ROWS)
        n_interp = max(C.MIN_INTERP_ROWS, min(int(C.INTERP_FRACTION * n_major), C.MAX_INTERP_ROWS))
        for variant in ["random", "IAFP"]:
            m = f"ImbLLM-{variant}"
            if cached(m):
                continue
            if not llm_ok:
                board.set_stage(name, seed, m, "skipped", reason=llm_skip_reason)
                log.info("%s | stage %-18s | SKIPPED: %s", tag, m, llm_skip_reason)
                continue
            with stage(board, name, seed, m, tag):
                set_all_seeds(seed)
                vdir = sdir / f"llm_{variant}"
                syn_csv = syndir / f"{variant}.csv"
                label_part = f"{D.label_token} is 1"
                info = {}
                minority_clean = prep.clean(imb_df[y_imb == 1])
                if syn_csv.exists() and not args.force:
                    log.info("%s | %s | synthetic csv already exists -> skipping fine-tune + generation", tag, m)
                    info = json.load(open(vdir / "info.json")) if (vdir / "info.json").exists() else {}
                else:
                    import torch
                    model_dir = vdir / "model"
                    if (model_dir / "config.json").exists() and (vdir / "training_loss_epochs.csv").exists():
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        log.info("%s | %s | found saved fine-tuned model -> skipping fine-tuning", tag, m)
                        tok = AutoTokenizer.from_pretrained(model_dir)
                        model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
                        model.eval()
                        info = json.load(open(vdir / "info.json")) if (vdir / "info.json").exists() else {}
                        max_len = info.get("max_len", 512)
                    else:
                        interp = make_interpolated(minority_clean, prep, n_interp, seed)    # same for both variants
                        rows = rows_to_parts(pd.concat([minority_clean, interp], ignore_index=True), prep.features)
                        log.info("%s | %s | fine-tune set = %d retained minority + %d interpolated rows | "
                                 "synthetic to generate = %d%s", tag, m, len(minority_clean), len(interp), n_synth,
                                 "" if n_synth == n_needed_full else
                                 f" (CAPPED from {n_needed_full}; classes will not be perfectly balanced)")
                        w = order_weights(imp, prep.features) if variant == "IAFP" else None
                        model, tok, max_len, info = finetune_llm(
                            C.LLM_MODEL_NAME, rows, label_part, w, seed, C.LLM_EPOCHS, vdir, f"{name}/s{seed}/{m}")
                        vdir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(vdir / "model")
                        tok.save_pretrained(vdir / "model")
                        write_json(vdir / "info.json", info)
                    log.info("%s | %s | generating %d rows (condition_yx, temperature %.2f)", tag, m, n_synth, C.TEMPERATURE)
                    syn_df, gmeta, gstats = run_generation(model, tok, prep, minority_clean, n_synth, variant, imp,
                                                           seed, max_len, D.label_token, f"{name}/s{seed}/{m}")
                    syn_df.to_csv(syn_csv, index=False)
                    gmeta.head(2000).to_csv(vdir / "generation_samples.csv", index=False)
                    info.update(gstats)
                    write_json(vdir / "info.json", info)
                    log.info("%s | %s | generation done: %s", tag, m, gstats)
                    del model
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    shutil.rmtree(vdir / "model", ignore_errors=True)      # keep disk small; csv + logs are kept

                syn_clean = read_synth(syn_csv, prep)
                X_syn = prep.encode(syn_clean)
                qm = quality_metrics(X_ref, X_syn, X_ret, seed)
                write_json(sdir / f"quality_{variant}.json", qm)
                log.info("%s | %s | synthetic quality: %s", tag, m, {k_: round(v, 4) for k_, v in qm.items()})
                Xtr = np.vstack([X_imb, X_syn])
                ytr = np.concatenate([y_imb, np.ones(len(X_syn), dtype=int)])
                ev(m, Xtr, ytr, {"synthetic_rows": int(len(X_syn)), "synthetic_needed_for_full_balance": int(n_needed_full),
                                 "fully_balanced": bool(len(X_syn) >= n_needed_full), "quality": qm,
                                 "llm_info": info, "llm_epochs": C.LLM_EPOCHS})

        # ---------------- plots (one t-SNE with everything overlaid) ----------------
        with stage(board, name, seed, "plots", tag):
            synth = {}
            for variant in ["random", "IAFP"]:
                p = syndir / f"{variant}.csv"
                if p.exists():
                    synth[f"ImbLLM-{variant}"] = prep.encode(read_synth(p, prep))
            if synth and len(X_ref) > 0:
                make_tsne_plot(sdir / "tsne_real_vs_synthetic.png", f"{name} | seed {seed} | real vs synthetic minority",
                               seed, X_maj, X_ret, X_ref, synth)
                log.info("%s | wrote tsne_real_vs_synthetic.png", tag)
            else:
                log.info("%s | no synthetic data -> t-SNE skipped", tag)
            plot_training_loss(sdir, f"{name} | seed {seed} | DistilGPT2 fine-tuning loss")

        res = {}
        for m in METHOD_ORDER:
            if metrics_path(m).exists():
                res[m] = json.load(open(metrics_path(m)))
        write_json(sdir / "seed_results.json", res)
        board.finish_seed(name, seed)
        log.info("%s | FINISHED (%d/%d methods have metrics)", tag, len(res), len(METHOD_ORDER))


# =====================================================================================
# aggregation (per dataset + across datasets)
# =====================================================================================
def load_long(root: Path, dataset_dirs):
    rows = []
    for dd in dataset_dirs:
        for f in sorted(dd.glob("seed_*/metrics/*.json")):
            try:
                r = json.load(open(f))
                r.pop("quality", None)
                r.pop("llm_info", None)
                rows.append(r)
            except Exception:
                pass
    return pd.DataFrame(rows)


def ordered_methods(present):
    return [m for m in METHOD_ORDER if m in present]


def bar_chart(summary, metric, title, path):
    ms = ordered_methods(summary.index)
    mean = summary.loc[ms, (metric, "mean")]
    std = summary.loc[ms, (metric, "std")].fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(ms, mean, yerr=std, capsize=4, color=[METHOD_COLORS[m] for m in ms], edgecolor="black", lw=0.6)
    for b, v in zip(bars, mean):
        ax.text(b.get_x() + b.get_width() / 2, min(v + 0.01, 1.0), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_dataset(ds_dir: Path):
    long = load_long(ds_dir.parent, [ds_dir])
    if long.empty:
        return None
    long.to_csv(ds_dir / "results_per_seed.csv", index=False)
    summary = long.groupby("method")[METRICS].agg(["mean", "std"])
    summary["n_seeds"] = long.groupby("method")["seed"].nunique()
    flat = summary.copy()
    flat.columns = [f"{a}_{b}" if b else a for a, b in flat.columns]
    flat = flat.loc[ordered_methods(flat.index)]
    flat.to_csv(ds_dir / "summary.csv")
    write_json(ds_dir / "summary.json", json.loads(flat.to_json(orient="index")))
    cdir = ds_dir / "charts"
    cdir.mkdir(exist_ok=True)
    for m in METRICS:
        bar_chart(summary, m, f"{ds_dir.name}: {m} (mean ± std over seeds, same test set per seed)", cdir / f"{m}.png")
    return summary


def aggregate_run(run_dir: Path):
    dsdirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name != "aggregate" and (p / "dataset_info.json").exists()])
    long = load_long(run_dir, dsdirs)
    if long.empty:
        return
    adir = run_dir / "aggregate"
    adir.mkdir(exist_ok=True)
    long.to_csv(adir / "all_results_long.csv", index=False)
    per_ds = long.groupby(["dataset", "method"])[METRICS].mean().reset_index()
    per_ds.to_csv(adir / "mean_per_dataset_method.csv", index=False)
    write_json(adir / "mean_per_dataset_method.json", json.loads(per_ds.to_json(orient="records")))
    methods = ordered_methods(long["method"].unique())
    n_ds = long["dataset"].nunique()
    overall = {}
    for metric in METRICS:
        piv = per_ds.pivot(index="dataset", columns="method", values=metric)[methods]
        piv.to_csv(adir / f"table_{metric}.csv")
        # heatmap datasets x methods
        fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 3, 0.42 * len(piv) + 2))
        im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.6 else "black")
        ax.set_title(f"{metric}: mean over seeds (rows = datasets)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(adir / f"heatmap_{metric}.png", dpi=150)
        plt.close(fig)
        # mean over datasets, with std across datasets
        mm, ss = piv.mean(), piv.std()
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(methods, mm[methods], yerr=ss[methods].fillna(0), capsize=4,
               color=[METHOD_COLORS[m] for m in methods], edgecolor="black", lw=0.6)
        for i, m in enumerate(methods):
            ax.text(i, min(mm[m] + 0.01, 1.0), f"{mm[m]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{metric}: average over {n_ds} datasets (error bar = std across datasets)")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(adir / f"mean_over_datasets_{metric}.png", dpi=150)
        plt.close(fig)
        # average rank (1 = best)
        ranks = piv.rank(axis=1, ascending=False, method="average").mean()
        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.bar(methods, ranks[methods], color=[METHOD_COLORS[m] for m in methods], edgecolor="black", lw=0.6)
        for i, m in enumerate(methods):
            ax.text(i, ranks[m] + 0.05, f"{ranks[m]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{metric}: average rank over datasets (lower = better)")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(adir / f"avg_rank_{metric}.png", dpi=150)
        plt.close(fig)
        overall[metric] = {"mean_over_datasets": mm.to_dict(), "avg_rank": ranks.to_dict()}
        # IAFP vs random paired comparison
        if {"ImbLLM-IAFP", "ImbLLM-random"} <= set(piv.columns):
            dlt = (piv["ImbLLM-IAFP"] - piv["ImbLLM-random"]).dropna().sort_values()
            if len(dlt):
                fig, ax = plt.subplots(figsize=(8, 0.32 * len(dlt) + 2))
                ax.barh(dlt.index, dlt.values, color=["#2ca02c" if v > 0 else "#d62728" for v in dlt.values])
                ax.axvline(0, color="k", lw=0.8)
                ax.set_title(f"{metric}: ImbLLM-IAFP minus ImbLLM-random (>0 => IAFP better)")
                fig.tight_layout()
                fig.savefig(adir / f"iafp_minus_random_{metric}.png", dpi=150)
                plt.close(fig)
                overall[metric]["iafp_vs_random"] = {"wins": int((dlt > 1e-9).sum()), "ties": int((dlt.abs() <= 1e-9).sum()),
                                                     "losses": int((dlt < -1e-9).sum()), "mean_delta": float(dlt.mean())}
    write_json(adir / "overall_summary.json", {"n_datasets": n_ds, "datasets": sorted(long["dataset"].unique()),
                                               "methods": methods, "metrics": overall})
    # one markdown table (F1 / G-score / AUC means) for quick pasting
    lines = ["| method | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for m in methods:
        lines.append(f"| {m} | " + " | ".join(f"{overall[x]['mean_over_datasets'][m]:.4f}" for x in METRICS) + " |")
    (adir / "overall_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =====================================================================================
# main
# =====================================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_name", default=None, help="re-use the same name to RESUME a run")
    p.add_argument("--datasets", nargs="*", default=None, help="only these dataset names")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help=f"LLM epochs (default {C.LLM_EPOCHS})")
    p.add_argument("--model_name", default=None, help="HF name or local path of the LM")
    p.add_argument("--max_synth", type=int, default=None)
    p.add_argument("--max_interp", type=int, default=None)
    p.add_argument("--no_llm", action="store_true", help="skip both ImbLLM variants (baselines only)")
    p.add_argument("--allow_cpu_llm", action="store_true", help="run the LLM even without a GPU (very slow)")
    p.add_argument("--smoke", action="store_true", help="tiny/fast sanity run: 1 seed, 1 epoch, small caps, 2 smallest datasets")
    p.add_argument("--list", action="store_true", help="list datasets that would run, then exit")
    p.add_argument("--aggregate_only", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute everything, ignoring cached artifacts")
    return p.parse_args()


def apply_overrides(a):
    if a.smoke:
        C.SEEDS, C.LLM_EPOCHS, C.MAX_INTERP_ROWS, C.MIN_INTERP_ROWS, C.MAX_SYNTH_ROWS = [42], 1, 200, 50, 200
        C.GEN_BATCH_SIZE = 64
    if a.seeds:
        C.SEEDS = a.seeds
    if a.epochs is not None:
        C.LLM_EPOCHS = a.epochs
    if a.model_name:
        C.LLM_MODEL_NAME = a.model_name
    if a.max_synth is not None:
        C.MAX_SYNTH_ROWS = a.max_synth
    if a.max_interp is not None:
        C.MAX_INTERP_ROWS = a.max_interp


def log_environment():
    log.info("python %s | pandas %s | numpy %s", sys.version.split()[0], pd.__version__, np.__version__)
    try:
        import sklearn, xgboost, imblearn
        log.info("sklearn %s | xgboost %s | imblearn %s", sklearn.__version__, xgboost.__version__, imblearn.__version__)
    except Exception:
        pass
    for k in ["SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"]:
        if os.environ.get(k):
            log.info("env %s=%s", k, os.environ[k])
    try:
        import torch
        import transformers
        log.info("torch %s | transformers %s | CUDA available: %s", torch.__version__, transformers.__version__, torch.cuda.is_available())
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            log.info("GPU: %s | %.1f GB visible", pr.name, pr.total_memory / 1e9)
    except Exception as e:
        log.warning("torch/transformers not importable: %s", e)


def _signal_handler(signum, frame):
    raise Interrupted(signal.Signals(signum).name)


def main():
    args = parse_args()
    apply_overrides(args)
    run_name = args.run_name or ("smoke" if args.smoke else datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    for s in (signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)):
        signal.signal(s, _signal_handler)

    if args.aggregate_only:
        aggregate_run(run_dir)
        log.info("aggregated -> %s", run_dir / "aggregate")
        return 0

    log.info("#" * 100)
    log.info("RUN %s  ->  %s", run_name, run_dir.resolve())
    log_environment()
    log.info("config: seeds=%s q=%.2f epochs=%d batch=%d temp=%.2f max_interp=%d max_synth=%d cpus=%d model=%s",
             C.SEEDS, C.Q, C.LLM_EPOCHS, C.LLM_BATCH_SIZE, C.TEMPERATURE, C.MAX_INTERP_ROWS, C.MAX_SYNTH_ROWS,
             C.N_CPUS, C.LLM_MODEL_NAME)

    specs = discover_datasets(Path(args.data_dir), set(args.datasets or []), set(args.exclude))
    eligible = sorted([s for s in specs if s.skip_reason is None], key=lambda s: s.n_rows)
    if args.smoke and not args.datasets:
        eligible = eligible[:2]
    skipped = [s for s in specs if s.skip_reason is not None]

    log.info("DATASET PLAN (smallest first, so more datasets finish early):")
    for i, s in enumerate(eligible, 1):
        log.info("  %2d. %-16s rows=%d", i, s.name, s.n_rows)
    for s in skipped:
        log.info("  --  %-16s SKIPPED: %s", s.name, s.skip_reason)
    if args.list:
        return 0

    board = StatusBoard(run_dir)
    for i, s in enumerate(eligible, 1):
        board.init_dataset(s.name, i, s.n_rows, "pending")
    for s in skipped:
        board.init_dataset(s.name, 1000, s.n_rows, "skipped", s.skip_reason)
    board.save()
    write_json(run_dir / "run_config.json", {
        "args": vars(args), "config": {k: v for k, v in vars(C).items() if k.isupper()},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})

    # LLM availability
    llm_ok, llm_reason = (not args.no_llm), "--no_llm"
    if llm_ok:
        try:
            import torch
            import transformers  # noqa
            if not torch.cuda.is_available() and not args.allow_cpu_llm:
                llm_ok, llm_reason = False, "no GPU visible (use --allow_cpu_llm to force CPU)"
        except Exception as e:
            llm_ok, llm_reason = False, f"torch/transformers unavailable: {e}"
    if not llm_ok:
        log.warning("ImbLLM stages will be SKIPPED: %s", llm_reason)

    t_run = time.time()
    try:
        for pos, spec in enumerate(eligible, 1):
            ds_dir = run_dir / spec.name
            if args.force and ds_dir.exists():
                shutil.rmtree(ds_dir)
            if (ds_dir / "DATASET_DONE").exists():
                board.set_ds(spec.name, "done", "cached from a previous job")
                log.info("[%d/%d] %s | already complete (DATASET_DONE) -> skipping", pos, len(eligible), spec.name)
                continue
            ds_dir.mkdir(parents=True, exist_ok=True)
            board.set_ds(spec.name, "running")
            t_ds = time.time()
            with file_logging(ds_dir / "train.log"):
                log.info("=" * 100)
                log.info("[%d/%d] DATASET %s | elapsed so far %s", pos, len(eligible), spec.name, fmt_secs(time.time() - t_run))
                log.info("\n%s", board.render())
                try:
                    D = load_dataset(spec)
                    write_json(ds_dir / "dataset_info.json", D.info)
                    log.info("%s | rows=%d | %d numeric + %d categorical features | classes=%d%s | minority=%r: %d rows (%.2f%%)",
                             spec.name, D.info["rows"], len(D.num_cols), len(D.cat_cols), D.info["n_classes"],
                             " (smallest class = minority, others merged)" if D.info["multiclass_merged"] else "",
                             D.info["minority_label"], D.info["minority_count"], 100 * D.info["minority_fraction"])
                    for si, seed in enumerate(C.SEEDS, 1):
                        run_seed(D, spec, seed, si, pos, len(eligible), ds_dir, board, args, llm_ok, llm_reason)
                    summarize_dataset(ds_dir)
                    bad = [(s_, st) for s_, sd in board.d["datasets"][spec.name]["seeds"].items()
                           for st, v in sd["stages"].items() if v["state"] == "failed"]
                    if bad:
                        board.set_ds(spec.name, "partial", f"failed stages: {bad}")
                        log.error("%s | finished WITH FAILED STAGES: %s (re-run to retry only those)", spec.name, bad)
                    else:
                        (ds_dir / "DATASET_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
                        board.set_ds(spec.name, "done", "")
                        log.info("%s | DATASET COMPLETE in %s -> %s", spec.name, fmt_secs(time.time() - t_ds), ds_dir)
                except Exception as e:  # noqa
                    log.error("%s | DATASET FAILED: %s\n%s", spec.name, e, traceback.format_exc())
                    board.set_ds(spec.name, "failed", f"{type(e).__name__}: {e}")
            try:
                aggregate_run(run_dir)
            except Exception as e:  # noqa
                log.error("aggregate failed: %s", e)
            log.info("\n%s", board.render())
    except (Interrupted, KeyboardInterrupt) as e:
        log.error("!!! INTERRUPTED (%s). Recording state. Datasets marked [DONE] are complete and independent; "
                  "re-run with the same --run_name to resume.", e)
        board.mark_interrupted()
        log.info("\n%s", board.render())
        return 143
    log.info("ALL DONE in %s", fmt_secs(time.time() - t_run))
    log.info("\n%s", board.render())
    log.info("cross-dataset charts/tables: %s", run_dir / "aggregate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
