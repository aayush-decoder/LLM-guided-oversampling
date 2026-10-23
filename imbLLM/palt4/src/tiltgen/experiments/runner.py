"""The ONE runner used by every command (palt2 had a copy per version).

A WorkItem is (experiment, dataset, seed, method, kind, config). `run_item`
    prepares the splits (protocol + q), opens a RunContext (fresh directory,
    manifest status "running"), builds the method, generates |largest class|
    minority rows, and writes metrics / diagnostics / parse stats / synthetic.

kinds
    reference   original | imbalance | threshold_only | prior_shift | scale_pos_weight
    classical   ros | smote | adasyn | smote_nc                     (imblearn)
    deep        ctgan | tvae (SDV) | tabddpm | arf (synthcity)      (optional extras)
    v1          ImbLLM as published                 config: V1Config fields
    v2          ImbLLM corrected                    config: V2Config fields (+ preset)
    tilt        TILT (palt4's method)                       config: TiltConfig fields (+ preset)
    llm_base    great_style | taptap_style | predllm_style   config: V1Config fields
                (re-implementations on this repo's engine, not the official packages)

Any non-reference item may add `curate: cllm` to its config: the generated rows
are then filtered by core/baselines/curation.py before evaluation.

Every metrics.json also carries (docs/RESEARCH_REVIEW.md):
    <learner>_*   the same evaluation with LR / RF / MLP (/ LightGBM)       G1
    cost_*        wall-clock seconds, peak GPU memory, LLM tokens             G4
    cov_*         dataset covariates (imbalance, modes, overlap, ...)         G8
    order_*       order-sensitivity summary when measured (v2)                3.3
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field

from .. import config as C
from ..core.baselines import deep
from ..core.baselines import classical
from ..core.baselines import curation
from ..core.data.protocol import prepare
from ..core.evaluate import quality
from ..core.evaluate import covariates
from ..core.evaluate import diversity
from ..core.evaluate import downstream
from ..core.evaluate import privacy
from ..core.runs import ModelCache, RunContext

KINDS = ("reference", "classical", "deep", "v1", "v2", "llm_base", "tilt")


@dataclass
class WorkItem:
    experiment: str
    dataset: str
    seed: int
    method: str                       # display name / directory name
    kind: str
    config: dict = field(default_factory=dict)
    protocol: str = "clean"
    q: float = 0.2

    def run_config(self) -> dict:
        return {"kind": self.kind, "method": self.method, "protocol": self.protocol,
                "q": self.q, **self.config}


def _peak_vram_reset() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except ImportError:
        pass
    return 0.0


def _flatten_order(diag: dict) -> dict:
    o = diag.get("order_sensitivity") or {}
    if "synthetic" not in o:
        return {}
    return {"order_std_per_token_synthetic": o["synthetic"]["std_nats_per_token_mean"],
            "order_std_per_token_real": o["real"]["std_nats_per_token_mean"],
            "order_ratio_synthetic_to_real": o["ratio_synthetic_to_real"]}


def build_llm(item: WorkItem, cache: ModelCache | None = None):
    from ..methods import llm_baselines
    from ..methods.v1_paper.method import V1Config
    from ..methods.v1_paper.method import build as build_v1
    from ..methods.v2_corrected.method import PRESETS, V2Config
    from ..methods.tilt import method as tm
    from ..methods.v2_corrected.method import build as build_v2
    config = {k: v for k, v in item.config.items() if k != "curate"}
    if item.kind == "v1":
        return build_v1(C.resolve(V1Config, cli=config), cache=cache)
    if item.kind == "v2":
        cfg = dict(config)
        preset = cfg.pop("preset", None)
        return build_v2(C.resolve(V2Config, presets=PRESETS, preset=preset, cli=cfg),
                        cache=cache)
    if item.kind == "tilt":
        cfg = dict(config)
        preset = cfg.pop("preset", None) or "tilt_sir"
        return tm.build(C.resolve(tm.TiltConfig, presets=tm.PRESETS, preset=preset, cli=cfg),
                        cache=cache)
    if item.kind == "llm_base":
        cfg = dict(config)
        name = cfg.pop("baseline", item.method)
        return llm_baselines.build(name, C.resolve(V1Config, cli=cfg), cache=cache)
    raise KeyError(item.kind)


def run_item(item: WorkItem, *, resume: bool = False, verbose: bool = True) -> dict | None:
    ctx = RunContext(item.experiment, item.dataset, item.method, item.seed, item.run_config())
    if resume and ctx.finished():
        if verbose:
            print(f"  = {item.dataset:<11} {item.method:<28} seed{item.seed}  (done, skipped)")
        return None
    splits = prepare(item.dataset, seed=item.seed, q=item.q, protocol=item.protocol)
    with ctx:
        ctx.write("splits.json", {**splits.summary(), "log": splits.log})
        diag: dict = {}
        cov = covariates.compute(splits, seed=item.seed)
        _peak_vram_reset()
        llm_stats: dict = {}
        if item.kind == "reference":
            metrics = downstream.evaluate_reference(item.config.get("reference", item.method),
                                                    splits, seed=item.seed)
            synth = None
        else:
            if item.kind == "classical":
                synth = classical.run(item.config.get("baseline", item.method), splits,
                                      seed=item.seed, scale=item.config.get("scale", True))
            elif item.kind == "deep":
                synth = deep.REGISTRY[item.config.get("baseline", item.method)](
                    splits, seed=item.seed, epochs=item.config.get("epochs"))
            else:
                gen = build_llm(item)
                try:
                    gen.fit(splits, seed=item.seed)
                    synth = gen.generate(splits, seed=item.seed)
                finally:
                    diag = gen.diagnostics
                    ctx.write("train_log.json", gen.train_log)
                    ctx.write("parse_stats.json", gen.parse_stats.as_dict())
                    ctx.write("diagnostics.json", diag)
                    if gen.engine is not None:
                        ctx.note(llm=gen.engine.describe())
                        llm_stats = dict(gen.engine.stats)
                    gen.release()
            if item.config.get("curate"):
                synth, cur = curation.CURATORS[item.config["curate"]](synth, splits,
                                                                      seed=item.seed)
                ctx.write("curation.json", cur)
            ctx.write_csv("synthetic.csv", synth)
            if len(synth) == 0:
                raise RuntimeError("generator produced no usable rows (see parse_stats.json"
                                   " / curation.json)")
            metrics = {**downstream.evaluate_synthetic(synth, splits, seed=item.seed),
                       **quality.quality_report(synth, splits),
                       **privacy.mia_report(synth, splits, seed=item.seed)}
            t = diag.get("tilt") or {}
            for k in ("lambda", "kl", "ess", "valid_share", "z_selected_mean", "z_pool_mean"):
                if isinstance(t.get(k), (int, float)):
                    metrics[f"tilt_{k}"] = float(t[k])
            if len(splits.minor_heldout):
                fp = diversity.frontier_point(synth, splits.minor_heldout, splits.schema,
                                              splits.train_minor_full)
                metrics.update({f"frontier_{k}": v for k, v in fp.items()})
            metrics["shortfall"] = int(splits.n_target - len(synth))
            metrics.update(_flatten_order(diag))
        metrics.update(cov)
        metrics.update({"cost_elapsed_sec": ctx.elapsed(), "cost_peak_vram_gb": _peak_vram_gb(),
                        "cost_train_tokens": int(llm_stats.get("train_tokens", 0)),
                        "cost_gen_tokens": int(llm_stats.get("gen_tokens", 0)),
                        "cost_score_tokens": int(llm_stats.get("score_tokens", 0))})
        ctx.write("metrics.json", metrics)
    if verbose:
        f1 = metrics.get("f1", float("nan"))
        extra = f" real_f1={metrics['real_f1']:.4f}" if "real_f1" in metrics else ""
        print(f"  + {item.dataset:<11} {item.method:<28} seed{item.seed}  f1={f1:.4f}"
              f" auc={metrics.get('auc', float('nan')):.4f}{extra}  ({ctx.elapsed()}s)")
    return metrics


def run_all(items: list[WorkItem], *, resume: bool = False, strict: bool = False) -> dict:
    ok, failed, skipped = 0, [], 0
    for it in items:
        try:
            r = run_item(it, resume=resume)
            if r is None:
                skipped += 1
            else:
                ok += 1
        except Exception as exc:
            failed.append(f"{it.dataset}/{it.method}/seed{it.seed}: {exc!r}")
            print(f"  ! {it.dataset:<11} {it.method:<28} seed{it.seed}  FAILED: {exc}",
                  file=sys.stderr)
            if strict:
                traceback.print_exc()
                raise
    print(f"\n{ok} done, {skipped} skipped (resume), {len(failed)} failed")
    for f in failed:
        print(f"  failed: {f}")
    return {"ok": ok, "skipped": skipped, "failed": failed}
