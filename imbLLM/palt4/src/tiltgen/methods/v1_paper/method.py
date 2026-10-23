"""ImbLLM as published (arXiv 2510.09783v2), assembled from C1/C2/C3.

    finetune_on  major+minor        GReaT-style corpus: all of D_train
                 minor              D_minor only                     (Table IV "-inter")
                 minor+interpolate  D_minor U D_inter                (ImbLLM, Eq. 5)
    permutation  permute_xy | fix_y | fix_y_end | identity           (C2)
    condition    condition_y | condition_yx  (+ value_source)        (C1)

Setting (condition_y, permute_xy, major+minor) is the GReaT row of Table III
(0.6833); (condition_yx, fix_y, minor+interpolate) is full ImbLLM (0.8002).

Paper-faithful on purpose (each is corrected, behind a flag, in v2):
    random-partner interpolation, categoricals dropped from D_inter, pooled
    conditioning values, no category repair, no clipping.
Engineering bugs from palt2 are NOT reproduced (they were never the paper's
method): precision, padding, EOS, prompt tokenisation, parse accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from . import condition, interpolate
from ...core.data.protocol import Splits
from ...core.serialize import row_to_text
from ..pipeline import LLMOversampler, LLMSettings, Strategies
from . import permute

FINETUNE_SOURCES = ("major+minor", "minor", "minor+interpolate")


@dataclass
class V1Config(LLMSettings):
    condition: str = "condition_yx"            # C1
    permutation: str = "fix_y"                 # C2
    finetune_on: str = "minor+interpolate"     # C3
    interpolation_ratio: float = 1.0           # r:  |D_inter| = r * |D_major|  (Fig. 5)
    value_source: str = "pooled"               # C1 ambiguity; see condition.py

    def __post_init__(self):
        if self.condition not in condition.STRATEGIES:
            raise ValueError(f"condition must be one of {condition.STRATEGIES}")
        if self.permutation not in permute.STRATEGIES:
            raise ValueError(f"permutation must be one of {permute.STRATEGIES}")
        if self.finetune_on not in FINETUNE_SOURCES:
            raise ValueError(f"finetune_on must be one of {FINETUNE_SOURCES}")
        if not 0.0 <= self.interpolation_ratio <= 10.0:
            raise ValueError("interpolation_ratio must be in [0, 10]")


def serialise_frames(frames, splits: Splits, permutation: str, rng, sig: int) -> list[str]:
    s = splits.schema
    texts = []
    for frame in frames:
        present = [c for c in s.features if c in frame.columns]
        for row in frame.to_dict("records"):
            order = permute.order_columns(present, s.target, permutation, rng)
            texts.append(row_to_text(row, order, s, sig))
    rng.shuffle(texts)
    return texts


def build_corpus(splits: Splits, rng: np.random.Generator, *, cfg: V1Config):
    s = splits.schema
    diag: dict = {}
    if cfg.finetune_on == "major+minor":
        frames = [splits.imbalanced_train]
        n_inter = 0
    else:
        frames = [splits.train_minor]
        n_inter = 0
        if cfg.finetune_on == "minor+interpolate":
            n_inter = int(round(cfg.interpolation_ratio * splits.n_target))
            inter, _, _ = interpolate.interpolate(splits.train_minor, s, n_inter, rng)
            if len(inter):
                inter = inter.copy()
                inter[s.target] = s.minority_label
                frames.append(inter)
                diag["interpolation"] = {
                    "mode": "random (Eq. 5)",
                    "covariance_ratio": interpolate.covariance_ratio(
                        splits.train_minor, inter, s.continuous),
                    "covariance_ratio_theory": interpolate.theoretical_covariance_ratio(
                        len(splits.train_minor))}
    n_min = len(splits.train_minor)
    diag["composition"] = {
        "n_rows": sum(len(f) for f in frames), "n_real_minority": n_min,
        "n_interpolated": n_inter,
        "interpolated_share": round(n_inter / max(n_min + n_inter, 1), 4),
        "share_with_categoricals": 1.0 if not s.categorical or cfg.finetune_on == "major+minor"
        else round(n_min / max(n_min + n_inter, 1), 4)}
    diag["label_free_feature_fraction"] = permute.label_free_fraction(
        len(s.features), cfg.permutation)
    return serialise_frames(frames, splits, cfg.permutation, rng, cfg.sig_digits), diag


def strategies(cfg: V1Config) -> Strategies:
    return Strategies(
        corpus=partial(build_corpus, cfg=cfg),
        prompts=lambda n, splits, rng: condition.build_prompts(
            n, splits, rng, strategy=cfg.condition, value_source=cfg.value_source,
            sig=cfg.sig_digits),
        repair_categories=False, clip=False, name="v1_paper")


def build(cfg: V1Config, **kw) -> LLMOversampler:
    return LLMOversampler(cfg, strategies(cfg), **kw)


# Table III (8 rows) + Table IV "-inter" arm.  tag -> (condition, permutation, finetune_on)
ABLATION = {
    "great_baseline": ("condition_y", "permute_xy", "major+minor"),
    "c1_only": ("condition_yx", "permute_xy", "major+minor"),
    "c2_only": ("condition_y", "fix_y", "major+minor"),
    "c3_only": ("condition_y", "permute_xy", "minor+interpolate"),
    "c1_c2": ("condition_yx", "fix_y", "major+minor"),
    "c1_c3": ("condition_yx", "permute_xy", "minor+interpolate"),
    "c2_c3": ("condition_y", "fix_y", "minor+interpolate"),
    "full_imbllm": ("condition_yx", "fix_y", "minor+interpolate"),
    "imbllm_minus_inter": ("condition_yx", "fix_y", "minor"),
}
