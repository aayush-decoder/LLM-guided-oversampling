# Version-agnostic experiment runner (the evaluation phase of Fig. 1 of the paper).

# For every seed:  split -> [fit generator] -> generate |D_fake| = gen_size*|D_real| rows
#                  -> save synthetic data -> TSTR + fidelity metrics -> metrics.json
# Finished seeds are skipped unless overwrite=True, so interrupted benchmarks resume where they stopped.

# Run directory layout (<version>/runs/<dataset>/<method>__<llm>/):
#     seed<k>/config.json  split.json  preprocessor.json
#     seed<k>/model/                  fine-tuned LLM (full weights or LoRA adapter) + llm_meta.json
#     seed<k>/train_log.jsonl         loss curve (LLM methods)
#     seed<k>/synthetic_encoded.csv   D_fake in the encoded (evaluation) space
#     seed<k>/synthetic_clean.csv     D_fake decoded to real categories / units
#     seed<k>/generation_stats.json   validity rate, retries, timings
#     seed<k>/metrics.json            TSTR (+ Original reference) + fidelity
#     seed<k>/plots/*.png             with --plots
#     summary.json                    mean/std over seeds
# 
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .baselines.sdv_models import SDV_METHODS, sdv_fit_sample
from .data.preprocess import TARGET, TabularSplit, make_split
from .eval.fidelity import fidelity_report
from .eval.tstr import tstr
from .paths import runs_root
from .utils import get_logger, load_json, save_json, set_seed

log = get_logger("tabgen.exp")
NON_LLM = ("original",) + SDV_METHODS

# make_generator(method, cfg, split, run_dir, seed) -> object with .fit() -> dict and .generate(n) -> (enc_df, clean_df|None, stats)
GeneratorFactory = Callable[[str, dict, TabularSplit, Path, int], object]


def exp_dir(version: str, dataset: str, method: str, llm_tag: str, cfg: dict) -> Path:
    """<version>/runs/[<runs_subdir>/]<dataset>[_bin]/<method>__<llm>"""
    root = runs_root(version) / cfg["runs_subdir"] if cfg.get("runs_subdir") else runs_root(version)
    ds = f"{dataset}_bin" if cfg["data"].get("binarize") else dataset
    return root / ds / f"{method}__{llm_tag if method not in NON_LLM else 'none'}"


def split_from_cfg(dataset: str, seed: int, cfg: dict) -> TabularSplit:
    d = cfg["data"]
    return make_split(dataset, seed=seed, test_size=d["test_size"], train_size=d["train_size"],
                      fit_scaler_on=d["fit_scaler_on"], binarize=d.get("binarize"),
                      max_rows=d.get("max_rows"), decimals=d["decimals"])


def evaluate(split: TabularSplit, enc: pd.DataFrame, cfg: dict, seed: int) -> dict:
    m = {"tstr": tstr(enc, split.test, split.features, split.task, split.n_classes, seed),
         "original": tstr(split.train, split.test, split.features, split.task, split.n_classes, seed),
         "n_synthetic": len(enc)}
    if cfg["eval"].get("fidelity", True) and len(enc) > 5:
        cols = split.features + [TARGET]
        discrete = set(split.cat_cols) | ({TARGET} if split.task == "classification" else set())
        # fidelity vs HELD-OUT test rows; privacy (DCR, copies) vs training rows (fidelity.py)
        m["fidelity"] = fidelity_report(split.test, enc, cols, discrete, seed,
                                        cfg["eval"]["max_fidelity_rows"], train=split.train)
    return m


def run_seed(version: str, dataset: str, method: str, seed: int, cfg: dict, make_generator: GeneratorFactory,
             llm_tag: str, overwrite: bool = False, plots: bool = False) -> dict:
    run_dir = exp_dir(version, dataset, method, llm_tag, cfg) / f"seed{seed}"
    if (run_dir / "metrics.json").exists() and not overwrite:
        log.info(f"[skip] {run_dir} already finished")
        return load_json(run_dir / "metrics.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    split = split_from_cfg(dataset, seed, cfg)
    save_json({"version": version, "dataset": dataset, "method": method, "seed": seed, "llm": llm_tag, **cfg},
              run_dir / "config.json")
    save_json(split.summary(), run_dir / "split.json")
    save_json(split.pre.to_dict(), run_dir / "preprocessor.json")
    n_gen = max(1, int(round(cfg["data"]["gen_size"] * len(split.train))))
    log.info(f"== {version} | {dataset} | {method} | seed {seed} | n_train {len(split.train)} -> n_gen {n_gen}")

    t0 = time.time()
    train_stats, gen_stats, clean = {}, {}, None
    if method == "original":
        enc = split.train.copy()
    elif method in SDV_METHODS:
        clean = sdv_fit_sample(method, split.raw_train, split.num_cols, split.cat_cols, split.target, split.task,
                               n_gen, epochs=cfg.get("baselines", {}).get("epochs", 300), seed=seed)
        enc = split.pre.transform(clean)
    else:
        gen = make_generator(method, cfg, split, run_dir, seed)
        train_stats = gen.fit()
        enc, clean, gen_stats = gen.generate(n_gen)
    elapsed = round(time.time() - t0, 1)

    if clean is None:
        clean = split.pre.inverse_transform(enc)
    enc.to_csv(run_dir / "synthetic_encoded.csv", index=False)
    clean.to_csv(run_dir / "synthetic_clean.csv", index=False)
    save_json(gen_stats, run_dir / "generation_stats.json")
    metrics = {"dataset": dataset, "method": method, "seed": seed, "llm": llm_tag, "task": split.task,
               **evaluate(split, enc, cfg, seed), "train": train_stats, "generation": gen_stats,
               "seconds_total": elapsed}
    save_json(metrics, run_dir / "metrics.json")
    if plots:
        from .viz.synth_plots import plot_run
        plot_run(run_dir, split, enc)
    log.info(f"   -> tstr {metrics['tstr']} | original {metrics['original']}")
    return metrics


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    key = "accuracy" if results[0]["task"] == "classification" else "mse"
    vals = np.array([r["tstr"].get(key, np.nan) for r in results], float)
    orig = np.array([r["original"].get(key, np.nan) for r in results], float)
    out = {"metric": key, "mean": round(float(np.nanmean(vals)), 4), "std": round(float(np.nanstd(vals)), 4),
           "original_mean": round(float(np.nanmean(orig)), 4), "per_seed": vals.tolist()}
    fids = [r["fidelity"] for r in results if "fidelity" in r]
    for k in ("discriminator", "discriminator_lr", "inverse_kl", "density", "coverage", "dcr_mean"):
        if fids:
            out[k] = round(float(np.mean([f[k] for f in fids])), 4)
    return out


def run_experiment(version: str, dataset: str, method: str, cfg: dict, make_generator: GeneratorFactory,
                   llm_tag: str, seeds: list[int], overwrite=False, plots=False) -> dict:
    results = [run_seed(version, dataset, method, s, cfg, make_generator, llm_tag, overwrite, plots)
               for s in seeds]
    summ = {"dataset": dataset, "method": method, "llm": llm_tag, "seeds": seeds, **summarize(results)}
    save_json(summ, exp_dir(version, dataset, method, llm_tag, cfg) / "summary.json")
    log.info(f"SUMMARY {dataset}/{method}: {summ.get('metric')} = {summ.get('mean')} +- {summ.get('std')}")
    return summ
