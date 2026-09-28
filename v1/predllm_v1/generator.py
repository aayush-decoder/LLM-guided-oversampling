"""PredLLM v1 generator = Algorithm 1 of the paper, parameterised by a MethodSpec.

fit()      : fine-tune the LLM on D'_real (or reuse an identical fine-tune from the model cache)
generate() : sample x (feature- or target-conditional) -> query y (llm / classifier / none) -> clip

Model cache: <version>/runs/<dataset>/_models/<fingerprint>/ . Methods whose fine-tuning is identical
(same data split, LLM, permutation, augmentation, hyper-parameters) share one model, e.g. pred_llm,
abl_x_target_llm and abl_x_target_none. Each run dir gets model_ref.json pointing at the model used.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tabgen.data.preprocess import TARGET, TabularSplit
from tabgen.llm.loader import load_finetuned, load_llm, save_finetuned
from tabgen.llm.registry import LLMSpec, resolve_llm
from tabgen.llm.train_loop import train_lm
from tabgen.utils import get_logger, load_json, pick_device, save_json

from .finetune import RowTextDataset
from .methods import PRESETS, MethodSpec
from .querier import query_classifier, query_llm
from .sampler import sample_rows
from .serialize import encode, row_fields

log = get_logger("predllm_v1")
VERSION = "v1"


def finetune_cfg(cfg: dict, spec: LLMSpec) -> dict:
    """Resolves mode-dependent defaults (LoRA needs a higher LR and smaller micro-batches)."""
    ft = dict(cfg["finetune"])
    if spec.mode in ("lora", "qlora"):
        ft["lr"] = ft.get("lr_lora", 2e-4)
        ft["micro_batch_size"] = ft.get("micro_batch_size") or (4 if spec.mode == "qlora" else 8)
    return ft


class PredLLMV1:
    def __init__(self, method: MethodSpec, cfg: dict, split: TabularSplit, run_dir: Path, seed: int):
        self.m, self.cfg, self.split, self.run_dir, self.seed = method, cfg, split, Path(run_dir), seed
        self.spec = resolve_llm(cfg["llm"], cfg.get("mode"))
        self.device = pick_device(cfg.get("device", "auto"))
        self.rng = np.random.default_rng(seed)
        self.model = self.tok = None
        d = cfg["data"]
        self.fields = [row_fields(r, split.features, split.task, d["decimals"], cfg["sampling"]["target_decimals"])
                       for r in split.train.to_dict("records")]

    # ------------------------------------------------------------------ fine-tuning
    def fingerprint(self) -> str:
        ft = {k: v for k, v in finetune_cfg(self.cfg, self.spec).items() if k not in ("log_every", "reuse")}
        key = {"dataset": self.split.name, "seed": self.seed, "data": self.cfg["data"], "llm": self.spec.hf_id,
               "mode": self.spec.mode, "perm": self.m.permutation, "aug": self.m.augment_original, "ft": ft}
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def model_dir(self) -> Path:
        return self.run_dir.parent.parent / "_models" / f"{self.spec.tag}_{self.fingerprint()}"

    def fit(self) -> dict:
        mdir = self.model_dir()
        ref = {"model_dir": str(mdir), "fingerprint": self.fingerprint()}
        if (mdir / "llm_meta.json").exists() and self.cfg["finetune"].get("reuse", True):
            log.info(f"reusing fine-tuned model {mdir}")
            self.tok, self.model, _ = load_finetuned(mdir, self.device)
            stats = {**load_json(mdir / "train_stats.json"), "reused": True}
        else:
            self.tok, self.model = load_llm(self.spec, self.device)
            ds = RowTextDataset(self.fields, self.split.features, self.tok, self.m.permutation,
                                self.m.augment_original, self.seed)
            log.info(f"fine-tuning {self.spec.key} ({self.spec.mode}) on {len(ds)} sentences; e.g. '{ds.text(0)}'")
            stats = train_lm(self.model, self.tok, ds, finetune_cfg(self.cfg, self.spec), mdir, self.device)
            save_finetuned(self.model, self.tok, self.spec, mdir, {"method": self.m.as_dict()})
            save_json(stats, mdir / "train_stats.json")
        save_json(ref, self.run_dir / "model_ref.json")
        if (mdir / "train_log.jsonl").exists():
            shutil.copy(mdir / "train_log.jsonl", self.run_dir / "train_log.jsonl")
        self._set_lengths()
        return stats

    def _set_lengths(self):
        sample = [encode(f, self.split.features + [TARGET]) for f in self.fields[:200]]
        L = max(len(self.tok(t)["input_ids"]) for t in sample)
        self.cfg["sampling"]["max_new_tokens"] = L + 8
        self.cfg["query"]["max_new_tokens"] = 16

    # ------------------------------------------------------------------ generation
    def generate(self, n: int) -> tuple[pd.DataFrame, None, dict]:
        sp, m, cfg = self.split, self.m, self.cfg
        rows, stats = sample_rows(self.model, self.tok, sp.train, sp.features, sp.task, n, m.sampling,
                                  require_target=(m.query == "none"), cfg=cfg, rng=self.rng)
        if len(rows) == 0:
            raise RuntimeError("LLM produced no valid rows - model is under-trained (increase epochs)")
        X = rows[sp.features].copy()
        gen_y = rows[TARGET].to_numpy(float) if TARGET in rows else np.full(len(rows), np.nan)
        empirical = self.rng.choice(sp.train[TARGET].to_numpy(float), size=len(rows))
        fallback = np.where(np.isnan(gen_y), empirical, gen_y)
        if m.query == "llm":
            y, qstats = query_llm(self.model, self.tok, X, sp.features, cfg, fallback)
            stats.update(qstats)
        elif m.query == "classifier":
            y = query_classifier(sp.train, X, sp.features, sp.task, self.seed)
        else:
            y = fallback
        clip = cfg["sampling"]["clip"]
        X = X.clip(-clip, clip).round(cfg["data"]["decimals"])
        if sp.task == "classification":
            y = np.clip(np.round(y), 0, sp.n_classes - 1).astype(int)
        enc = X.assign(**{TARGET: y}).reset_index(drop=True)
        stats["method"] = m.as_dict()
        return enc, None, stats


def make_generator(method: str, cfg: dict, split: TabularSplit, run_dir: Path, seed: int) -> PredLLMV1:
    return PredLLMV1(PRESETS[method], cfg, split, run_dir, seed)


def load_generator(run_dir: Path) -> PredLLMV1:
    """Rebuilds a fitted generator from a finished run dir (for `generate` / `infer` / `attention`)."""
    from tabgen.experiment import split_from_cfg
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    split = split_from_cfg(cfg["dataset"], cfg["seed"], cfg)
    g = PredLLMV1(PRESETS[cfg["method"]], cfg, split, run_dir, cfg["seed"])
    g.tok, g.model, _ = load_finetuned(load_json(run_dir / "model_ref.json")["model_dir"], g.device)
    g._set_lengths()
    return g
