"""The one generator skeleton every LLM-based method plugs into.

    fit(splits)                                          [LLM CALL 1: finetune]
        corpus  <- strategies.corpus(splits, rng)
        engine  <- model cache hit  OR  fine-tune(corpus) and cache it
    generate(splits, n)                                  [LLM CALL 2: complete]
        repeat up to max_rounds until n rows are accepted:
            prompts <- strategies.prompts(k, splits, rng)
            texts   <- engine.complete(prompts)
            rows    <- parse (optional category repair) -> optional clip
                       -> optional accept() filter (TapTap / Pred-LLM verifiers,
                          the latter is                  [LLM CALL 3: score])
        rows <- strategies.refine(rows)   (v2 SIR-Gibbs,  [LLM CALL 3: score])
        diagnostics <- strategies.measure(...)  (order sensitivity, entropy
                                                decomposition; [score]/[complete])

v1 and v2 differ ONLY in the Strategies object they hand to this class. When a
v2 flag is at its v1 value, v2 hands over v1's function itself, so
`--preset v1_exact` runs literally the same code (tests assert identical
corpora and prompts).

P8 knobs, available to every method: `init="random"` (same architecture, no
pre-trained prior) and `anonymize_columns=True` (column names -> f1..fM,
target -> y) -- the 2x2 that tests whether the LLM's semantic prior matters.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from ..core.data import kinds
from ..core.data.protocol import Schema, Splits
from ..core.llm import zoo
from ..core.llm import hardware
from ..core.runs import ModelCache
from ..core.serialize import (ParseStats, category_vocab, clip_continuous,
                              continuous_bounds, texts_to_frame)


@dataclass
class LLMSettings:
    """Backbone / schedule / sampling settings shared by every LLM method."""
    model: str = "distilgpt2"          # the paper's backbone
    hardware: str = "auto"             # auto | a configs/hardware profile name
    finetune_mode: str = "auto"        # auto | full | lora | qlora
    init: str = "pretrained"           # pretrained | random            (P8)
    anonymize_columns: bool = False    # named | anonymised             (P8)
    epochs: int = 50                   # paper
    lr: float = 5e-5
    effective_batch: int = 32          # paper; kept on every device via grad accumulation
    temperature: float = 0.7           # paper
    top_p: float = 1.0
    oversample_attempts: float = 1.5
    max_rounds: int = 3
    sig_digits: int = 5
    cache_models: bool = True
    log_every: int = 0


@dataclass
class Strategies:
    corpus: Callable                    # (splits, rng) -> (texts, diagnostics)
    prompts: Callable                   # (n, splits, rng) -> (prompts, conds)
    repair_categories: bool = False
    clip: bool = False
    snap_discrete: bool = False         # v2: integer columns onto their observed support
    accept: Callable | None = None      # (rows, splits, engine, rng) -> bool mask
    refine: Callable | None = None      # (rows, engine, splits, rng) -> (rows, diag)
    measure: Callable | None = None     # (gen, rows, conds, splits, rng) -> diag
    name: str = ""


# --------------------------------------------------------------------------- P8

def anonymize(splits: Splits) -> tuple[Splits, dict]:
    s = splits.schema
    mapping = {c: f"f{i + 1}" for i, c in enumerate(s.features)}
    mapping[s.target] = "y"
    sch = Schema(target="y", continuous=[mapping[c] for c in s.continuous],
                 categorical=[mapping[c] for c in s.categorical], classes=s.classes,
                 minority_label=s.minority_label, majority_label=s.majority_label,
                 task=s.task, source_task=s.source_task, threshold=s.threshold,
                 discrete=[mapping[c] for c in s.discrete],
                 ordinal={mapping[c]: lv for c, lv in s.ordinal.items()})
    ren = lambda df: df.rename(columns=mapping)  # noqa: E731
    return replace(splits, schema=sch, train_major=ren(splits.train_major),
                   train_minor=ren(splits.train_minor),
                   minor_heldout=ren(splits.minor_heldout), val=ren(splits.val),
                   test=ren(splits.test)), {v: k for k, v in mapping.items()}


# --------------------------------------------------------------------------- generator

class LLMOversampler:
    def __init__(self, settings: LLMSettings, strategies: Strategies, *,
                 cache: ModelCache | None = None):
        self.cfg = settings
        self.strat = strategies
        self.cache = cache or ModelCache(enabled=settings.cache_models)
        self.engine = None
        self.train_log: list[dict] = []
        self.parse_stats = ParseStats()
        self.diagnostics: dict = {}
        self.conds: list = []
        self.max_new_tokens = 256
        self._inverse: dict | None = None

    # ------------------------------------------------------------------ fit
    def _view(self, splits: Splits) -> Splits:
        if not self.cfg.anonymize_columns:
            return splits
        s2, inv = anonymize(splits)
        self._inverse = inv
        return s2

    def fit(self, splits: Splits, seed: int = 0) -> "LLMOversampler":
        from ..core.llm.engine import LLMEngine, TrainCfg
        cfg = self.cfg
        splits = self._view(splits)
        rng = np.random.default_rng(seed)
        corpus, diag = self.strat.corpus(splits, rng)
        self.diagnostics["corpus"] = diag
        self.diagnostics["corpus_sample"] = corpus[:3]
        self.diagnostics["n_finetune_sequences"] = len(corpus)

        spec = zoo.get(cfg.model)
        hw = hardware.get_hardware(cfg.hardware)
        est_len = int(max(map(len, corpus)) / 3) + 8 if corpus else 256
        pl = hardware.plan(spec, hw, mode=cfg.finetune_mode,
                           effective_batch=cfg.effective_batch, seq_len=min(est_len, 1024))
        self.diagnostics["plan"] = pl.as_dict()
        self.engine = LLMEngine(spec, pl, init=cfg.init, seed=seed)

        train = TrainCfg(epochs=cfg.epochs, lr=cfg.lr, seed=seed, log_every=cfg.log_every)
        fp = ModelCache.fingerprint(model=spec.key, init=cfg.init, mode=pl.mode, corpus=corpus,
                                    train={"epochs": cfg.epochs, "lr": cfg.lr,
                                           "effective_batch": cfg.effective_batch},
                                    seed=seed)
        self.diagnostics["model_fingerprint"] = fp
        if self.cache.has(fp):
            self.engine.load(self.cache.path(fp))
            import json
            p = self.cache.path(fp) / "train_log.json"
            self.train_log = json.loads(p.read_text()) if p.exists() else []
            self.diagnostics["model_cache"] = "hit"
        else:
            self.engine.load()
            lens = self.engine.token_lengths(corpus)
            train.max_len = max(lens)
            self.train_log = self.engine.finetune(corpus, train)
            if self.cache.enabled:
                self.engine.save(self.cache.path(fp))
                import json
                (self.cache.path(fp) / "train_log.json").write_text(json.dumps(self.train_log))
            self.diagnostics["model_cache"] = "miss (trained)"
        lens = self.engine.token_lengths(corpus)
        self.max_new_tokens = int(max(lens) * 1.25) + 8
        self.diagnostics["token_lengths"] = {"max": int(max(lens)),
                                             "p50": float(np.median(lens)),
                                             "max_new_tokens": self.max_new_tokens}
        return self

    def gen_batch(self) -> int:
        """Sampling batch: the profile's value for a ~1.5B model, scaled up for
        smaller backbones (KV-cache memory is ~linear in model size)."""
        base = hardware.get_hardware(self.cfg.hardware).gen_batch
        scale = min(16.0, max(1.0, 1.5 / max(self.engine.spec.params_b, 1e-3)))
        return int(min(512, base * scale))

    # ------------------------------------------------------------------ generate
    def generate(self, splits: Splits, n: int | None = None, seed: int = 0) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("fit() first")
        cfg = self.cfg
        splits = self._view(splits)
        s = splits.schema
        n = n or splits.n_target
        rng = np.random.default_rng(seed + 1)
        vocab = category_vocab(splits.imbalanced_train, s)
        bounds = continuous_bounds(splits.imbalanced_train, s)
        stats = ParseStats()
        frames, conds, have, est_yield = [], [], 0, 0.5
        gen_batch = self.gen_batch()

        for rnd in range(cfg.max_rounds):
            need = n - have
            if need <= 0:
                break
            # budget: never more than 4x the shortfall per round, whatever the yield
            ask = int(min(math.ceil(need * cfg.oversample_attempts / max(est_yield, 0.05)),
                          4 * need + 16))
            prompts, pconds = self.strat.prompts(ask, splits, rng)
            outs = self.engine.complete(prompts, temperature=cfg.temperature, top_p=cfg.top_p,
                                        max_new_tokens=self.max_new_tokens,
                                        batch_size=gen_batch, seed=seed * 1000 + rnd)
            stats.prompts_sent += len(prompts)
            stats.completions += len(outs)
            stats.finished_with_eos += sum(o[1] for o in outs)
            texts = [p + o[0] for p, o in zip(prompts, outs)]
            if rnd == 0:
                self.diagnostics["prompts"] = {
                    "n": len(prompts), "unique_frac": round(len(set(prompts)) / len(prompts), 4),
                    "example": prompts[0], "example_completion": texts[0][:300]}
            frame, keep = texts_to_frame(texts, s, vocab=vocab,
                                         repair=self.strat.repair_categories, stats=stats)
            kc = [pconds[i] for i in keep]
            if self.strat.clip and len(frame):
                frame, nclip = clip_continuous(frame, bounds)
                self.diagnostics["clipped_cells"] = self.diagnostics.get("clipped_cells", 0) + nclip
            if self.strat.snap_discrete and s.discrete and len(frame):
                frame, nsnap = kinds.snap_frame(frame, s.discrete, splits.imbalanced_train)
                self.diagnostics["snapped_cells"] = self.diagnostics.get("snapped_cells", 0) + nsnap
            if self.strat.accept is not None and len(frame):
                mask = np.asarray(self.strat.accept(frame, splits, self.engine, rng), bool)
                stats.rejected_by_verifier += int((~mask).sum())
                frame = frame[mask].reset_index(drop=True)
                kc = [c for c, m in zip(kc, mask) if m]
            frames.append(frame)
            conds.extend(kc)
            have += len(frame)
            est_yield = len(frame) / max(len(prompts), 1)
            if est_yield < 0.01:
                # a model that cannot write rows will not learn to in round 2;
                # stop and report the shortfall instead of burning 4x more prompts
                self.diagnostics["stopped_low_yield"] = {"round": rnd, "yield": est_yield}
                break

        self.parse_stats = stats
        out = pd.concat(frames, ignore_index=True).head(n) if frames else \
            pd.DataFrame(columns=s.features)
        self.conds = conds[:len(out)]
        if self.strat.refine is not None and len(out):
            out, rd = self.strat.refine(out, self.engine, splits, rng)
            self.diagnostics["refine"] = rd
        if self.strat.measure is not None and len(out):
            self.diagnostics.update(self.strat.measure(self, out, self.conds, splits, rng))
        self.diagnostics["n_generated"] = int(len(out))
        self.diagnostics["shortfall"] = int(n - len(out))
        self.diagnostics["engine"] = self.engine.describe()
        if self._inverse:
            out = out.rename(columns=self._inverse)
        return out.reset_index(drop=True)

    def release(self) -> None:
        if self.engine is not None:
            self.engine.release()
            self.engine = None


def settings_dict(cfg) -> dict:
    return asdict(cfg)


__all__ = ["LLMSettings", "Strategies", "LLMOversampler", "anonymize", "settings_dict", "field"]
