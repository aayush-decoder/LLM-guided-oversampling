"""C1 -- the sampling condition.   Paper Sec. III-A1, Eq. (2).

    condition_y   (GReaT baseline)   every prompt is  "Y is y_min,"
    condition_yx  (ImbLLM)           "Y is y_min, X_i is v_i,"
                                     i ~ Uniform{1..M},  v_i ~ p(X_i)

AMBIGUITY: the paper writes v ~ p(X_i) without saying which p.
    value_source="pooled"   (default, the literal reading) v from all of D_train
    value_source="minority" v from D_minor only
Under heavy imbalance the pooled marginal ~= the majority's marginal, so the
pooled reading conditions the minority generator on majority-typical values:
the induced conditioning law is the minority marginal re-weighted by
w(v) = p(v) / p(v | y_min), the Bayes likelihood ratio -- an accidental,
un-tunable boundary tilt. v2 replaces the binary choice with the tau-bridge.

Returns (prompts, conds): conds[i] = (feature, value) the prompt fixed, or None.
"""

from __future__ import annotations

import numpy as np

from ...core.data.protocol import Splits
from ...core.serialize import clause, label_head, prompt_from_clauses

STRATEGIES = ("condition_y", "condition_yx")
VALUE_SOURCES = ("pooled", "minority")


def build_prompts(n: int, splits: Splits, rng: np.random.Generator, *,
                  strategy: str = "condition_yx", value_source: str = "pooled",
                  sig: int = 5) -> tuple[list[str], list]:
    s = splits.schema
    head = label_head(s)
    if strategy == "condition_y":
        return [prompt_from_clauses([head])] * n, [None] * n
    if strategy != "condition_yx":
        raise ValueError(f"unknown condition {strategy!r}; have {STRATEGIES}")
    if value_source not in VALUE_SOURCES:
        raise ValueError(f"unknown value_source {value_source!r}")
    src = splits.imbalanced_train if value_source == "pooled" else splits.train_minor
    feats = s.features
    picks = rng.integers(0, len(feats), size=n)
    rows = rng.integers(0, len(src), size=n)
    prompts, conds = [], []
    for f_i, r_i in zip(picks, rows):
        col = feats[int(f_i)]
        val = src.iloc[int(r_i)][col]
        prompts.append(prompt_from_clauses([head, clause(col, val, s, sig)]))
        conds.append((col, val))
    return prompts, conds
