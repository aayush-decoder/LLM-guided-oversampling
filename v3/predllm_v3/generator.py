"""PredLLM v3 generator: Negative-aware fine-tuning + Outlier support generation.

fit()      : builds schema + computes outlier boundaries + fits anomaly scorer
             -> trains LLM with joint positive NLL and majority unlikelihood penalty.
generate() : samples pool with outlier-bounded FSM and negative guidance
             -> permutation-marginalised exact posterior for y
             -> privacy gate + anomaly filter + discriminator SIR down to n rows.
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
from tabgen.utils import apply_overrides, get_logger, load_json, pick_device, save_json

from .codec import FieldCodec, build_schema, rows_as_text
from .curation import curate
from .decoding import NumericConstraint
from .finetune import V3RowDataset, train_lm_v3
from .methods import PRESETS
from .outlier import AnomalyScorer
from .posterior import class_posterior, decide_labels, regression_values
from .sampler import sample_features

log = get_logger("predllm_v3")
VERSION = "v3"


class PredLLMV3:
    def __init__(self, method: str, cfg: dict, split: TabularSplit, run_dir: Path, seed: int):
        self.method = method
        self.cfg = apply_overrides(cfg, PRESETS[method])
        self.split, self.run_dir, self.seed = split, Path(run_dir), seed
        self.spec = resolve_llm(self.cfg["llm"], self.cfg.get("mode"))
        self.device = pick_device(self.cfg.get("device", "auto"))
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        out_cfg = self.cfg.get("outlier", {})
        self.schema = build_schema(
            split,
            self.cfg.get("serialization", "raw"),
            self.cfg["data"].get("max_value_decimals", 4),
            outlier_mode=out_cfg.get("mode", "iqr"),
            outlier_factor=float(out_cfg.get("factor", 1.5)),
        )

        src = split.raw_train if self.cfg.get("serialization", "raw") == "raw" else split.train
        self.train_text = rows_as_text(src, self.schema)
        self.tok = self.model = self.codec = self.cons = None
        self.anomaly_scorer = AnomalyScorer(
            contamination=float(out_cfg.get("contamination", 0.1)),
            seed=seed,
        )

    def _ft_cfg(self) -> dict:
        ft = dict(self.cfg.get("finetune", {}))
        if self.spec.mode in ("lora", "qlora"):
            ft["lr"] = ft.get("lr_lora", 2e-4)
            ft["micro_batch_size"] = ft.get("micro_batch_size") or (4 if self.spec.mode == "qlora" else 8)
        return ft

    def fingerprint(self) -> str:
        ft = {k: v for k, v in self._ft_cfg().items() if k not in ("log_every", "reuse")}
        key = {
            "dataset": self.split.name,
            "seed": self.seed,
            "data": self.cfg["data"],
            "llm": self.spec.hf_id,
            "mode": self.spec.mode,
            "ser": self.cfg.get("serialization", "raw"),
            "loss": self.cfg.get("loss", {}),
            "outlier": self.cfg.get("outlier", {}),
            "ft": ft,
        }
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def model_dir(self) -> Path:
        return self.run_dir.parent.parent / "_models" / f"{self.spec.tag}_{self.fingerprint()}"

    def fit(self) -> dict:
        mdir = self.model_dir()
        feats = self.split.features

        # 1. Fit anomaly scorer on numeric features (or all encoded features if none)
        num_feats = [c for c in feats if self.schema.kinds[c] == "num"]
        target_feats = num_feats if num_feats else feats
        self.anomaly_scorer.fit(self.split.train[target_feats].to_numpy(float), feature_names=target_feats)

        # 2. Model fine-tuning (or cache reuse)
        if (mdir / "llm_meta.json").exists() and self.cfg["finetune"].get("reuse", True):
            log.info(f"reusing fine-tuned v3 model {mdir}")
            self.tok, self.model, _ = load_finetuned(mdir, self.device)
            stats = {**load_json(mdir / "train_stats.json"), "reused": True}
            self._ready()
        else:
            self.tok, self.model = load_llm(self.spec, self.device)
            self._ready()
            L = self.cfg.get("loss", {})
            ds = V3RowDataset(
                self.train_text,
                self.codec,
                lam=float(L.get("target_weight", 3.0)),
                w_scaffold=float(L.get("scaffold_weight", 0.0)),
                perm_copies=self.cfg["finetune"].get("perm_copies", 1),
                orig_copies=self.cfg["finetune"].get("orig_copies", 1),
                negative_weight=float(self.cfg["finetune"].get("negative_weight", 0.5)),
                seed=self.seed,
            )
            log.info(f"v3 fine-tuning {self.spec.key} ({self.spec.mode}) on {len(ds)} sequences; pos={ds.n_pos}, neg={ds.n_neg}")
            stats = train_lm_v3(self.model, self.tok, ds, self._ft_cfg(), mdir, self.device)
            save_finetuned(
                self.model,
                self.tok,
                self.spec,
                mdir,
                {"method": self.method, "version": VERSION, "serialization": self.cfg.get("serialization", "raw")},
            )
            save_json(stats, mdir / "train_stats.json")

        save_json({"model_dir": str(mdir), "fingerprint": self.fingerprint()}, self.run_dir / "model_ref.json")
        save_json(
            {
                "kinds": self.schema.kinds,
                "decimals": self.schema.decimals,
                "ranges": self.schema.ranges,
                "outlier_ranges": self.schema.outlier_ranges,
                "minority_classes": self.schema.minority_classes,
                "majority_classes": self.schema.majority_classes,
            },
            self.run_dir / "schema.json",
        )
        if (mdir / "train_log.jsonl").exists():
            shutil.copy(mdir / "train_log.jsonl", self.run_dir / "train_log.jsonl")
        return stats

    def _ready(self):
        self.model.eval()
        self.codec = FieldCodec(self.schema, self.tok)
        vocab = self.model.get_output_embeddings().weight.shape[0]
        self.cons = NumericConstraint(self.tok, vocab, self.model.device)

    def _to_frames(self, rows: list[dict], y) -> tuple[pd.DataFrame, pd.DataFrame]:
        sp, sch = self.split, self.schema
        df = pd.DataFrame(rows)[sch.features]
        for c in sch.features:
            if sch.kinds[c] == "num":
                df[c] = df[c].astype(float)
        if self.cfg.get("serialization", "raw") == "raw":
            clean = df.assign(**{sp.target: y})
            enc = sp.pre.transform(clean)
        else:
            enc = df.astype(float).assign(**{TARGET: y})
            clean = sp.pre.inverse_transform(enc)
        return enc, clean

    def generate(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        cur = self.cfg.get("curation", {})
        pool_n = math.ceil(cur.get("pool_factor", 1.5) * n) if cur.get("enabled", True) else n
        dcfg = {**self.cfg.get("decoding", {}), **self.cfg.get("query", {})}

        with torch.no_grad():
            rows, stats = sample_features(
                self.model,
                self.codec,
                self.train_text,
                pool_n,
                self.cfg,
                self.rng,
                self.gen,
                self.cons,
            )
            if self.split.task == "classification":
                probs = class_posterior(
                    self.model,
                    self.codec,
                    rows,
                    self.cfg["query"].get("permutations", 4),
                    dcfg,
                    self.rng,
                    minority_prior_boost=float(self.cfg["query"].get("minority_prior_boost", 0.0)),
                )
                idx = decide_labels(
                    probs,
                    self.cfg["query"].get("label_decode", "sample"),
                    self.cfg["query"].get("temperature_y", 0.7),
                    self.rng,
                )
                classes = self.schema.cats[self.schema.target]
                y = [classes[i] if self.cfg.get("serialization", "raw") == "raw" else int(classes[i]) for i in idx]
                stats["posterior_entropy"] = round(float(-(probs * np.log(probs + 1e-12)).sum(1).mean()), 4)
            else:
                emp = np.array([float(r[self.schema.target]) for r in self.train_text])
                y = regression_values(
                    self.model,
                    self.codec,
                    rows,
                    self.cfg["query"].get("permutations", 4),
                    dcfg,
                    self.cons,
                    self.gen,
                    self.rng,
                    emp,
                )

        enc, clean = self._to_frames(rows, y)
        assert len(enc) == len(clean), "encoded/clean misalignment"

        if cur.get("enabled", True):
            enc, idx, keep, cst = curate(
                self.split.train,
                enc,
                n,
                self.split.features,
                self.cfg,
                self.seed,
                scorer=self.anomaly_scorer,
            )
            clean = clean[keep].reset_index(drop=True).iloc[idx].reset_index(drop=True)
            stats.update(cst)

        stats["preset"] = self.method
        stats["version"] = VERSION
        return enc.reset_index(drop=True), clean, stats


def make_generator(method: str, cfg: dict, split: TabularSplit, run_dir: Path, seed: int) -> PredLLMV3:
    return PredLLMV3(method, cfg, split, run_dir, seed)


def load_generator(run_dir: Path) -> PredLLMV3:
    from tabgen.experiment import split_from_cfg
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    split = split_from_cfg(cfg["dataset"], cfg["seed"], cfg)
    g = PredLLMV3(cfg["method"], cfg, split, run_dir, cfg["seed"])
    g.tok, g.model, _ = load_finetuned(load_json(run_dir / "model_ref.json")["model_dir"], g.device)
    g._ready()
    return g
