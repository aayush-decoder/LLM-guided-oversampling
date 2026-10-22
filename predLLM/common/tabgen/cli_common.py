"""Argument parsing + config resolution shared by the v1 and v2 CLIs (identical flags in both)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import apply_overrides, deep_update, load_yaml

COMMANDS = ["run", "benchmark", "ablation", "train", "generate", "infer", "evaluate", "smoke", "methods"]


def build_parser(prog: str, doc: str, extra_commands: list[str] = ()) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=COMMANDS + list(extra_commands))
    ap.add_argument("--dataset", default="iris", help="name | a,b | all | paper | user | classification | regression")
    ap.add_argument("--datasets", default=None, help="benchmark/ablation datasets (same syntax)")
    ap.add_argument("--method", default=None)
    ap.add_argument("--methods", default=None, help="presets or groups (see `methods`)")
    ap.add_argument("--llm", default=None, help="alias/key from configs/llms.yaml: tiny, distilgpt2, 1.5b, 3b, 8b ...")
    ap.add_argument("--mode", default=None, choices=[None, "auto", "full", "lora", "qlora"])
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", type=int, default=None, help="number of seeds (0..runs-1)")
    ap.add_argument("--seeds", default=None, help="explicit seeds, e.g. 0,1,2")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--binarize", default=None, choices=[None, "median"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=value")
    ap.add_argument("--run", default=None, help="finished seed dir for generate / infer / evaluate / attention")
    ap.add_argument("--n", type=int, default=None, help="rows to generate / test rows to infer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plots", action="store_true", help="write plots/ for every run")
    ap.add_argument("--overwrite", action="store_true", help="re-run seeds that already have metrics.json")
    return ap


def build_cfg(a, default_config: Path) -> dict:
    cfg = load_yaml(a.config or default_config)
    upd: dict = {}
    for key in ("llm", "mode", "device"):
        if getattr(a, key):
            upd[key] = getattr(a, key)
    if a.epochs is not None:
        upd.setdefault("finetune", {})["epochs"] = a.epochs
    if a.max_steps is not None:
        upd.setdefault("finetune", {})["max_steps"] = a.max_steps
    if a.binarize:
        upd.setdefault("data", {})["binarize"] = a.binarize
    if a.max_rows:
        upd.setdefault("data", {})["max_rows"] = a.max_rows
    return apply_overrides(deep_update(cfg, upd), a.set)


def seeds_of(a, cfg) -> list[int]:
    return [int(s) for s in a.seeds.split(",")] if a.seeds else list(range(a.runs or cfg["eval"]["runs"]))
