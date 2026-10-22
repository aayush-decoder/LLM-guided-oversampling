"""ImbLLM v2: every audit correction behind a NAMED flag.

RULE: with every flag at its v1 value (`--preset v1_exact`) v2 hands the
pipeline v1's own functions, so it is v1 -- tests assert identical corpora and
prompts for the same seed. Any v2-vs-v1 difference is attributable to named
flags, and the `v2_flags` experiment toggles them one at a time.

FLAG                    v1 value   recommended   finding (docs/what_corrected.md)
interpolation_mode      random     knn           B1  2/3 covariance contraction + off-manifold
categorical_from_anchor False      True          B3  94% of the corpus lacks categoricals
tau                     1.0        0.0           B2  conditioning on the pooled marginal
repair_categories       False      True          B6  one OOV category kills a row
clip_continuous         False      True          B6  out-of-range values reach XGBoost
gibbs_sweeps            0          0 (full: 1)   B4  order-inconsistent dependency network
snap_discrete           False      True          H1  integer columns snapped to their observed
                                                     support (interpolants, parsed rows, Gibbs)
ordinal_interpolation   False      True          H2  ordered categoricals interpolated on rank
measure_*               --         on            B4/B5 diagnostics only; never change rows

Not a flag because it costs nothing: every oversampler is evaluated BOTH with
and without the real minority rows in the classifier's training set (D4); see
core/evaluate/downstream.evaluate_synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from ...core.data import kinds
from ...core.data.protocol import Splits
from ...core.evaluate import diversity as dv
from ..pipeline import LLMOversampler, Strategies
from ..v1_paper import condition as v1c
from ..v1_paper import method as v1m
from ..v1_paper.method import V1Config, serialise_frames
from . import condition as c2
from . import gibbs as gb
from . import interpolate as i2


@dataclass
class V2Config(V1Config):
    interpolation_mode: str = "knn"
    knn_k: int = 5
    rose_bandwidth: float | None = None
    categorical_from_anchor: bool = True
    tau: float = 0.0
    n_clauses: int = 1
    repair_categories: bool = True
    clip_continuous: bool = True
    snap_discrete: bool = True
    ordinal_interpolation: bool = True
    gibbs_sweeps: int = 0
    gibbs_candidates: int = 16
    gibbs_max_rows: int | None = 256
    measure_order_sensitivity: bool = True
    order_rows: int = 128
    order_orders: int = 12
    measure_decomposition: bool = True
    decomposition_samples: int = 2000
    preset: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.interpolation_mode not in i2.MODES:
            raise ValueError(f"interpolation_mode must be one of {i2.MODES}")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")


# the corrected value of every method flag, and its v1 value
V1_VALUES = {"interpolation_mode": "random", "categorical_from_anchor": False, "tau": 1.0,
             "repair_categories": False, "clip_continuous": False, "snap_discrete": False,
             "ordinal_interpolation": False, "gibbs_sweeps": 0}
V2_VALUES = {"interpolation_mode": "knn", "categorical_from_anchor": True, "tau": 0.0,
             "repair_categories": True, "clip_continuous": True, "snap_discrete": True,
             "ordinal_interpolation": True, "gibbs_sweeps": 1}

PRESETS = {
    "v1_exact": {**V1_VALUES, "measure_order_sensitivity": False,
                 "measure_decomposition": False},
    "recommended": {**V2_VALUES, "gibbs_sweeps": 0},
    "geometry_only": {**V1_VALUES, "interpolation_mode": "knn"},
    "full": dict(V2_VALUES),
}


# --------------------------------------------------------------------------- corpus

def build_corpus(splits: Splits, rng: np.random.Generator, *, cfg: V2Config):
    """Delegates to v1 when every corpus flag is at its v1 value."""
    if cfg.interpolation_mode == "random" and not cfg.categorical_from_anchor \
            and not cfg.snap_discrete and not cfg.ordinal_interpolation:
        return v1m.build_corpus(splits, rng, cfg=cfg)
    s = splits.schema
    frames, n_inter, diag = [], 0, {}
    if cfg.finetune_on == "major+minor":
        frames.append(splits.imbalanced_train)
    else:
        frames.append(splits.train_minor)
        if cfg.finetune_on == "minor+interpolate":
            n_inter = int(round(cfg.interpolation_ratio * splits.n_target))
            ordinal = s.ordinal if cfg.ordinal_interpolation else None
            inter, anchors, _ = i2.interpolate(splits.train_minor, s, n_inter, rng,
                                               mode=cfg.interpolation_mode, k=cfg.knn_k,
                                               rose_bandwidth=cfg.rose_bandwidth,
                                               ordinal=ordinal)
            if len(inter):
                cov = v1m.interpolate.covariance_ratio(splits.train_minor, inter,
                                                       s.continuous) \
                    if s.continuous else float("nan")
                offm = i2.off_manifold_fraction(splits.train_minor, inter, s.continuous)
                nsnap = 0
                if cfg.snap_discrete and s.discrete:
                    inter, nsnap = kinds.snap_frame(inter, s.discrete, splits.imbalanced_train)
                if cfg.categorical_from_anchor and s.categorical:
                    inter = i2.with_anchor_categoricals(inter, splits.train_minor, anchors, s)
                inter = inter.copy()
                inter[s.target] = s.minority_label
                frames.append(inter)
                diag["interpolation"] = {"mode": cfg.interpolation_mode,
                                         "covariance_ratio": cov,
                                         "off_manifold_fraction": offm,
                                         "discrete_cells_snapped": nsnap,
                                         "ordinal_interpolated": list(ordinal or {}),
                                         "categoricals": "anchor" if cfg.categorical_from_anchor
                                         else "dropped"}
    n_min = len(splits.train_minor)
    diag["composition"] = {
        "n_rows": sum(len(f) for f in frames), "n_real_minority": n_min,
        "n_interpolated": n_inter,
        "interpolated_share": round(n_inter / max(n_min + n_inter, 1), 4),
        "share_with_categoricals": 1.0 if (cfg.categorical_from_anchor or not s.categorical
                                           or cfg.finetune_on == "major+minor")
        else round(n_min / max(n_min + n_inter, 1), 4)}
    diag["label_free_feature_fraction"] = v1m.permute.label_free_fraction(
        len(s.features), cfg.permutation)
    return serialise_frames(frames, splits, cfg.permutation, rng, cfg.sig_digits), diag


# --------------------------------------------------------------------------- prompts

def make_prompts(cfg: V2Config):
    """Delegates to v1's builder when tau == 1 with one clause (or condition_y)."""
    if cfg.condition == "condition_y" or (cfg.tau == 1.0 and cfg.n_clauses == 1):
        return lambda n, splits, rng: v1c.build_prompts(
            n, splits, rng, strategy=cfg.condition, value_source="pooled", sig=cfg.sig_digits)
    cache: dict = {}

    def prompts(n, splits, rng):
        key = id(splits.train_minor)
        if key not in cache:
            cache[key] = c2.TauBridge(splits.schema, splits.train_minor,
                                      splits.imbalanced_train, tau=cfg.tau)
        return c2.build_prompts(n, splits, rng, bridge=cache[key], n_clauses=cfg.n_clauses,
                                sig=cfg.sig_digits)
    return prompts


# --------------------------------------------------------------------------- diagnostics

def make_measure(cfg: V2Config):
    def measure(gen: LLMOversampler, rows, conds, splits: Splits, rng) -> dict:
        out: dict = {}
        s = splits.schema
        ref = splits.train_minor_full
        if len(splits.minor_heldout):
            out["frontier"] = dv.frontier_point(rows, splits.minor_heldout, s, ref)
        if cfg.condition == "condition_yx":
            b = c2.TauBridge(s, splits.train_minor, splits.imbalanced_train, tau=cfg.tau)
            out["tau_bridge"] = b.measured_kl()
        if cfg.measure_order_sensitivity:
            out["order_sensitivity"] = gb.order_sensitivity(
                rows, gen.engine, splits, rng, n_rows=cfg.order_rows,
                n_orders=cfg.order_orders, sig=cfg.sig_digits)
        if cfg.measure_decomposition and cfg.condition == "condition_yx" and len(conds):
            # condition_y twin from the SAME fine-tuned model (sampling only)
            n_y = min(cfg.decomposition_samples, len(rows))
            p_y = [v1c.build_prompts(1, splits, rng, strategy="condition_y")[0][0]] * n_y
            outs = gen.engine.complete(p_y, temperature=cfg.temperature, top_p=cfg.top_p,
                                       max_new_tokens=gen.max_new_tokens, seed=12345,
                                       batch_size=gen.gen_batch())
            from ...core.serialize import category_vocab, texts_to_frame
            rows_y, _ = texts_to_frame([p + o[0] for p, o in zip(p_y, outs)], s,
                                       vocab=category_vocab(splits.imbalanced_train, s),
                                       repair=cfg.repair_categories)
            if len(rows_y) >= 20:
                groups = dv.condition_groups([c[0] if c else None for c in conds],
                                             [c[1] if c else None for c in conds], s, ref)
                out["entropy_decomposition"] = dv.decompose(rows, groups, rows_y, s, ref)
        return out
    return measure


def strategies(cfg: V2Config) -> Strategies:
    refine = None
    if cfg.gibbs_sweeps > 0:
        refine = lambda rows, engine, splits, rng: gb.sir_gibbs(  # noqa: E731
            rows, engine, splits, rng, sweeps=cfg.gibbs_sweeps,
            n_candidates=cfg.gibbs_candidates, max_rows=cfg.gibbs_max_rows,
            sig=cfg.sig_digits, snap_discrete=cfg.snap_discrete)
    return Strategies(corpus=partial(build_corpus, cfg=cfg), prompts=make_prompts(cfg),
                      repair_categories=cfg.repair_categories, clip=cfg.clip_continuous,
                      snap_discrete=cfg.snap_discrete,
                      refine=refine, measure=make_measure(cfg), name="v2_corrected")


def build(cfg: V2Config, **kw) -> LLMOversampler:
    return LLMOversampler(cfg, strategies(cfg), **kw)
