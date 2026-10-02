"""PredLLM v2 generator.

fit()      : schema (N1) -> weighted value-only fine-tuning (N3), with the same model cache as v1
generate() : pool = gamma*n rows -> field-wise constrained x (N2) -> exact posterior y (N4)
             -> map to the shared encoded space -> DCR gate + discriminator SIR down to n rows (N5)
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabgen.data.preprocess import TARGET, TabularSplit
from tabgen.llm.loader import load_finetuned, load_llm, save_finetuned
from tabgen.llm.registry import resolve_llm
from tabgen.llm.train_loop import train_lm
from tabgen.utils import apply_overrides, get_logger, load_json, pick_device, save_json

from .codec import FieldCodec, build_schema, rows_as_text
from .curation import curate
from .decoding import NumericConstraint
from .finetune import SchemaRowDataset
from .methods import PRESETS
from .posterior import class_posterior, decide_labels, regression_values
from .sampler import sample_features

log = get_logger("predllm_v2")
VERSION = "v2"


class PredLLMV2:
    def __init__(self, method: str, cfg: dict, split: TabularSplit, run_dir: Path, seed: int):
        self.method = method
        self.cfg = apply_overrides(cfg, PRESETS[method])
        self.split, self.run_dir, self.seed = split, Path(run_dir), seed
        self.spec = resolve_llm(self.cfg["llm"], self.cfg.get("mode"))
        self.device = pick_device(self.cfg.get("device", "auto"))
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.schema = build_schema(split, self.cfg["serialization"], self.cfg["data"]["max_value_decimals"])
        src = split.raw_train if self.cfg["serialization"] == "raw" else split.train
        self.train_text = rows_as_text(src, self.schema)
        self.tok = self.model = self.codec = self.cons = None

    # ------------------------------------------------------------------ fine-tuning (N1, N3)
    def _ft_cfg(self) -> dict:
        ft = dict(self.cfg["finetune"])
        if self.spec.mode in ("lora", "qlora"):
            ft["lr"] = ft.get("lr_lora", 2e-4)
            ft["micro_batch_size"] = ft.get("micro_batch_size") or (4 if self.spec.mode == "qlora" else 8)
        return ft

    def fingerprint(self) -> str:
        ft = {k: v for k, v in self._ft_cfg().items() if k not in ("log_every", "reuse")}
        key = {"dataset": self.split.name, "seed": self.seed, "data": self.cfg["data"], "llm": self.spec.hf_id,
               "mode": self.spec.mode, "ser": self.cfg["serialization"], "loss": self.cfg["loss"], "ft": ft}
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def model_dir(self) -> Path:
        return self.run_dir.parent.parent / "_models" / f"{self.spec.tag}_{self.fingerprint()}"

    def fit(self) -> dict:
        mdir = self.model_dir()
        if (mdir / "llm_meta.json").exists() and self.cfg["finetune"].get("reuse", True):
            log.info(f"reusing fine-tuned model {mdir}")
            self.tok, self.model, _ = load_finetuned(mdir, self.device)
            stats = {**load_json(mdir / "train_stats.json"), "reused": True}
            self._ready()
        else:
            self.tok, self.model = load_llm(self.spec, self.device)
            self._ready()
            L = self.cfg["loss"]
            ds = SchemaRowDataset(self.train_text, self.codec, L["target_weight"], L["scaffold_weight"],
                                  self.cfg["finetune"]["perm_copies"], self.cfg["finetune"]["orig_copies"], self.seed)
            log.info(f"fine-tuning {self.spec.key} ({self.spec.mode}) on {len(ds)} sequences; e.g. '{ds.text(0)}'")
            stats = train_lm(self.model, self.tok, ds, self._ft_cfg(), mdir, self.device)
            save_finetuned(self.model, self.tok, self.spec, mdir, {"method": self.method,
                                                                    "serialization": self.cfg["serialization"]})
            save_json(stats, mdir / "train_stats.json")
        save_json({"model_dir": str(mdir), "fingerprint": self.fingerprint()}, self.run_dir / "model_ref.json")
        save_json({"kinds": self.schema.kinds, "decimals": self.schema.decimals, "ranges": self.schema.ranges,
                   "n_categories": {c: len(v) for c, v in self.schema.cats.items()}}, self.run_dir / "schema.json")
        if (mdir / "train_log.jsonl").exists():
            shutil.copy(mdir / "train_log.jsonl", self.run_dir / "train_log.jsonl")
        return stats

    def _ready(self):
        self.model.eval()
        self.codec = FieldCodec(self.schema, self.tok)
        vocab = self.model.get_output_embeddings().weight.shape[0]
        self.cons = NumericConstraint(self.tok, vocab, self.model.device)

    # ------------------------------------------------------------------ generation (N2, N4, N5)
    def _to_frames(self, rows: list[dict], y) -> tuple[pd.DataFrame, pd.DataFrame]:
        sp, sch = self.split, self.schema
        df = pd.DataFrame(rows)[sch.features]
        for c in sch.features:
            if sch.kinds[c] == "num":
                df[c] = df[c].astype(float)
        if self.cfg["serialization"] == "raw":
            clean = df.assign(**{sp.target: y})
            enc = sp.pre.transform(clean)
        else:
            enc = df.astype(float).assign(**{TARGET: y})
            clean = sp.pre.inverse_transform(enc)
        return enc, clean

    def generate(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        cur = self.cfg["curation"]
        pool_n = math.ceil(cur["pool_factor"] * n) if cur["enabled"] else n
        dcfg = {**self.cfg["decoding"], **self.cfg["query"]}
        with torch.no_grad():
            rows, stats = sample_features(self.model, self.codec, self.train_text, pool_n, self.cfg, self.rng,
                                          self.gen, self.cons)
            if self.split.task == "classification":
                probs = class_posterior(self.model, self.codec, rows, self.cfg["query"]["permutations"], dcfg, self.rng)
                idx = decide_labels(probs, self.cfg["query"]["label_decode"], self.cfg["query"]["temperature_y"], self.rng)
                classes = self.schema.cats[self.schema.target]
                y = [classes[i] if self.cfg["serialization"] == "raw" else int(classes[i]) for i in idx]
                stats["posterior_entropy"] = round(float(-(probs * np.log(probs + 1e-12)).sum(1).mean()), 4)
            else:
                emp = np.array([float(r[self.schema.target]) for r in self.train_text])
                y = regression_values(self.model, self.codec, rows, self.cfg["query"]["permutations"], dcfg,
                                      self.cons, self.gen, self.rng, emp)
        enc, clean = self._to_frames(rows, y)
        assert len(enc) == len(clean), "encoded/clean misalignment"
        if cur["enabled"]:
            enc, idx, keep, cst = curate(self.split.train, enc, n, self.split.features, self.cfg, self.seed)
            clean = clean[keep].reset_index(drop=True).iloc[idx].reset_index(drop=True)
            stats.update(cst)
        stats["preset"] = self.method
        return enc.reset_index(drop=True), clean, stats


def make_generator(method: str, cfg: dict, split: TabularSplit, run_dir: Path, seed: int) -> PredLLMV2:
    return PredLLMV2(method, cfg, split, run_dir, seed)


def load_generator(run_dir: Path) -> PredLLMV2:
    from tabgen.experiment import split_from_cfg
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    split = split_from_cfg(cfg["dataset"], cfg["seed"], cfg)
    g = PredLLMV2(cfg["method"], cfg, split, run_dir, cfg["seed"])
    g.tok, g.model, _ = load_finetuned(load_json(run_dir / "model_ref.json")["model_dir"], g.device)
    g._ready()
    return g
