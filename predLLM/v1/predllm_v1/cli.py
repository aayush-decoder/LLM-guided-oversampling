"""v1 CLI:  python -m predllm_v1 <command> [flags]

  run        one dataset x method(s) x seeds: train -> generate -> evaluate   (--method / --methods)
  benchmark  datasets x methods (Table II). Defaults: --datasets paper --methods paper
  ablation   Table III rows on --datasets (default: classification)
  train      fine-tune only (fills the model cache that later runs reuse)
  generate   INFERENCE: sample new synthetic rows from a finished run's fine-tuned LLM
  infer      INFERENCE: the fine-tuned LLM as a predictor p(y|x) on the real test set
  evaluate   recompute metrics.json from a run's synthetic_encoded.csv
  attention  Fig. 4: label-token attention, target last vs target first
  smoke      end-to-end check on iris that verifies every artifact
  methods    list presets and groups
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predLLM.common.tabgen.cli_common import build_cfg, build_parser, seeds_of
from predLLM.common.tabgen.data.preprocess import TARGET
from predLLM.common.tabgen.data.registry import resolve_datasets
from predLLM.common.tabgen.experiment import evaluate, exp_dir, run_experiment, split_from_cfg
from predLLM.common.tabgen.llm.registry import resolve_llm
from predLLM.common.tabgen.utils import get_logger, load_json, save_json

from .generator import VERSION, load_generator, make_generator
from .methods import GROUPS, PRESETS, resolve_methods

log = get_logger("predllm_v1")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def cmd_run(a, datasets, methods):
    cfg = build_cfg(a, CONFIG)
    tag = resolve_llm(cfg["llm"], cfg.get("mode")).tag
    out = [run_experiment(VERSION, ds, m, cfg, make_generator, tag, seeds_of(a, cfg), a.overwrite, a.plots)
           for ds in datasets for m in methods]
    print(pd.DataFrame(out)[["dataset", "method", "llm", "metric", "mean", "std", "original_mean"]].to_string())
    return 0


def cmd_train(a):
    cfg = build_cfg(a, CONFIG)
    method = a.method or "pred_llm"
    tag = resolve_llm(cfg["llm"], cfg.get("mode")).tag
    for seed in seeds_of(a, cfg):
        rd = exp_dir(VERSION, a.dataset, method, tag, cfg) / f"seed{seed}"
        rd.mkdir(parents=True, exist_ok=True)
        save_json({"version": VERSION, "dataset": a.dataset, "method": method, "seed": seed, "llm": tag, **cfg},
                  rd / "config.json")
        print(make_generator(method, cfg, split_from_cfg(a.dataset, seed, cfg), rd, seed).fit())
    return 0


def cmd_generate(a):
    g = load_generator(Path(a.run))
    n = a.n or len(g.split.train)
    enc, _, stats = g.generate(n)
    out = Path(a.out) if a.out else Path(a.run) / f"generated_{n}.csv"
    g.split.pre.inverse_transform(enc).to_csv(out, index=False)
    enc.to_csv(out.with_name(out.stem + "_encoded.csv"), index=False)
    print(f"wrote {len(enc)} rows -> {out}\nstats: {stats}")
    return 0


def cmd_infer(a):
    from sklearn.metrics import accuracy_score, mean_squared_error
    from .querier import query_llm
    g = load_generator(Path(a.run))
    test = g.split.test.head(a.n) if a.n else g.split.test
    y, st = query_llm(g.model, g.tok, test[g.split.features], g.split.features, g.cfg, fallback=np.zeros(len(test)))
    res = {"n": len(test), **st}
    if g.split.task == "classification":
        res["llm_accuracy"] = float(accuracy_score(test[TARGET], np.clip(np.round(y), 0, g.split.n_classes - 1)))
    else:
        res["llm_mse"] = float(mean_squared_error(test[TARGET], y))
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


def cmd_attention(a):
    from predLLM.common.tabgen.llm.loader import load_finetuned
    from predLLM.common.tabgen.viz.attention import field_attention, plot_attention_comparison
    from .serialize import encode
    g = load_generator(Path(a.run))
    tok, model, _ = load_finetuned(load_json(Path(a.run) / "model_ref.json")["model_dir"], g.device, attn="eager")
    feats = g.split.features
    layouts = {"target last (permute_x)": feats + [TARGET], "target first (permute_xy case)": [TARGET] + feats}
    res = {k: {f: 0.0 for f in feats} for k in layouts}
    rows = g.fields[:a.n or 20]
    for fields in rows:
        for label, order in layouts.items():
            for f, v in field_attention(model, tok, encode(fields, order), feats, TARGET).items():
                res[label][f] += v / len(rows)
    (Path(a.run) / "plots").mkdir(exist_ok=True)
    out = plot_attention_comparison(res, Path(a.run) / "plots" / "attention.png")
    save_json(res, Path(a.run) / "attention.json")
    print(f"wrote {out}")
    return 0


def main(argv=None):
    a = build_parser("predllm_v1", __doc__, extra_commands=["attention"]).parse_args(argv)
    if a.command == "methods":
        for k, m in PRESETS.items():
            print(f"{k:22s} perm={m.permutation:3s} sampling={m.sampling:8s} query={m.query:10s} "
                  f"aug={m.augment_original!s:5s} {m.paper_row}")
        print("groups:", GROUPS)
        return 0
    if a.command == "run":
        return cmd_run(a, resolve_datasets(a.dataset), resolve_methods(a.methods or a.method or "pred_llm"))
    if a.command == "benchmark":
        return cmd_run(a, resolve_datasets(a.datasets or "paper"), resolve_methods(a.methods or "paper"))
    if a.command == "ablation":
        return cmd_run(a, resolve_datasets(a.datasets or "classification"), resolve_methods("ablation"))
    if a.command == "smoke":
        from .smoke import run_smoke
        return run_smoke(llm=a.llm or "distilgpt2", device=a.device or "auto", max_steps=a.max_steps or 150)
    return {"train": cmd_train, "generate": cmd_generate, "infer": cmd_infer, "evaluate": cmd_evaluate,
            "attention": cmd_attention}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
