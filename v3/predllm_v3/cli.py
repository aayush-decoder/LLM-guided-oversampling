"""v3 CLI: python -m predllm_v3 <command> [flags]

  run        one dataset x preset(s) x seeds                      (--method / --methods)
  benchmark  datasets x presets. Defaults: --datasets paper --methods compare
  ablation   every v3 preset on --datasets
  train      fine-tune only
  generate   INFERENCE: sample synthetic rows from a finished run
  infer      INFERENCE: permutation-marginalised LLM posterior as a predictor on test set
  evaluate   recompute metrics.json
  smoke      end-to-end check on iris with tiny-gpt2
  methods    list presets and groups
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tabgen.cli_common import build_cfg, build_parser, seeds_of
from tabgen.data.preprocess import TARGET
from tabgen.data.registry import resolve_datasets
from tabgen.experiment import evaluate, exp_dir, run_experiment, split_from_cfg
from tabgen.llm.registry import resolve_llm
from tabgen.utils import load_json, save_json

from .generator import VERSION, load_generator, make_generator
from .methods import GROUPS, PRESETS, resolve_methods

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def cmd_run(a, datasets, methods):
    cfg = build_cfg(a, CONFIG)
    tag = resolve_llm(cfg["llm"], cfg.get("mode")).tag
    out = [
        run_experiment(VERSION, ds, m, cfg, make_generator, tag, seeds_of(a, cfg), a.overwrite, a.plots)
        for ds in datasets
        for m in methods
    ]
    print(pd.DataFrame(out)[["dataset", "method", "llm", "metric", "mean", "std", "original_mean"]].to_string())
    return 0


def cmd_train(a):
    cfg = build_cfg(a, CONFIG)
    method = a.method or "pred_llm_v3"
    tag = resolve_llm(cfg["llm"], cfg.get("mode")).tag
    for seed in seeds_of(a, cfg):
        rd = exp_dir(VERSION, a.dataset, method, tag, cfg) / f"seed{seed}"
        rd.mkdir(parents=True, exist_ok=True)
        save_json(
            {"version": VERSION, "dataset": a.dataset, "method": method, "seed": seed, "llm": tag, **cfg},
            rd / "config.json",
        )
        print(make_generator(method, cfg, split_from_cfg(a.dataset, seed, cfg), rd, seed).fit())
    return 0


def cmd_generate(a):
    g = load_generator(Path(a.run))
    n = a.n or len(g.split.train)
    enc, clean, stats = g.generate(n)
    out = Path(a.out) if a.out else Path(a.run) / f"generated_{n}.csv"
    clean.to_csv(out, index=False)
    enc.to_csv(out.with_name(out.stem + "_encoded.csv"), index=False)
    print(f"wrote {len(enc)} rows -> {out}\nstats: {stats}")
    return 0


def cmd_infer(a):
    from sklearn.metrics import accuracy_score, mean_squared_error
    from .codec import rows_as_text
    from .posterior import class_posterior, regression_values
    g = load_generator(Path(a.run))
    sp = g.split
    raw = g.cfg.get("serialization", "raw") == "raw"
    test_src = (sp.raw_test if raw else sp.test).head(a.n) if a.n else (sp.raw_test if raw else sp.test)
    rows = rows_as_text(test_src[sp.features], g.schema)
    dcfg = {**g.cfg.get("decoding", {}), **g.cfg.get("query", {})}
    y_true = sp.pre.transform(sp.raw_test.head(len(rows)))[TARGET].to_numpy()
    res = {"n": len(rows), "permutations": g.cfg["query"]["permutations"]}
    if sp.task == "classification":
        probs = class_posterior(g.model, g.codec, rows, g.cfg["query"]["permutations"], dcfg, g.rng)
        res["llm_accuracy"] = float(accuracy_score(y_true, probs.argmax(1)))
    else:
        emp = np.array([float(r[g.schema.target]) for r in g.train_text])
        y = regression_values(g.model, g.codec, rows, g.cfg["query"]["permutations"], dcfg, g.cons, g.gen, g.rng, emp)
        res["llm_mse"] = float(mean_squared_error(y_true, y))
    save_json(res, Path(a.run) / "infer_llm_as_predictor.json")
    print(res)
    return 0


def cmd_evaluate(a):
    rd = Path(a.run)
    cfg = load_json(rd / "config.json")
    split = split_from_cfg(cfg["dataset"], cfg["seed"], cfg)
    m = load_json(rd / "metrics.json") if (rd / "metrics.json").exists() else {}
    m.update(evaluate(split, pd.read_csv(rd / "synthetic_encoded.csv"), cfg, cfg["seed"]))
    save_json(m, rd / "metrics.json")
    print({k: m[k] for k in ("tstr", "original")}, m.get("fidelity"))
    return 0


def main(argv=None):
    a = build_parser("predllm_v3", __doc__).parse_args(argv)
    if a.command == "methods":
        for k, ov in PRESETS.items():
            print(f"{k:20s} overrides={ov}")
        print("groups:", GROUPS)
        return 0
    if a.command == "run":
        return cmd_run(a, resolve_datasets(a.dataset), resolve_methods(a.methods or a.method or "pred_llm_v3"))
    if a.command == "benchmark":
        return cmd_run(a, resolve_datasets(a.datasets or "paper"), resolve_methods(a.methods or "compare"))
    if a.command == "ablation":
        return cmd_run(a, resolve_datasets(a.datasets or "classification"), resolve_methods("ablation"))
    if a.command == "smoke":
        from .smoke import run_smoke
        return run_smoke(llm=a.llm or "tiny-gpt2", device=a.device or "auto", max_steps=a.max_steps or 30)
    return {"train": cmd_train, "generate": cmd_generate, "infer": cmd_infer, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
