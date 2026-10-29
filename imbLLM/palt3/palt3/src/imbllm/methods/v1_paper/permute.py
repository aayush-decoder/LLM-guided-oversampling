"""C2 -- column order of each fine-tuning row.   Paper Sec. III-A2, Fig. 2.

    permute_xy  (GReaT/TapTap)  shuffle all M+1 clauses; Y lands anywhere
    fix_y       (ImbLLM)        shuffle the M features; Y pinned first
    fix_y_end   (Pred-LLM-like) shuffle the M features; Y pinned last
    identity                    Y first, features in schema order

WHY POSITION MATTERS -- exactly, not via "attention scores"
A decoder-only LM factorises causally, p(a_1..a_{M+1}) = prod_t p(a_t | a_<t).
Under permute_xy with Y at slot k (uniform on 0..M), the k features before it
are trained from contexts that do NOT CONTAIN Y. E[k] = M/2, so the expected
fraction of FEATURE clauses trained label-free is exactly 1/2 for every M
(label_free_fraction). [palt2's docstring said M/(2(M+1)); that is the fraction
of ALL M+1 clauses, not of features.]
"""

from __future__ import annotations

import numpy as np

STRATEGIES = ("permute_xy", "fix_y", "fix_y_end", "identity")


def order_columns(features: list[str], target: str, strategy: str,
                  rng: np.random.Generator) -> list[str]:
    if strategy == "permute_xy":
        cols = [*features, target]
        rng.shuffle(cols)
        return cols
    if strategy == "fix_y":
        f = list(features)
        rng.shuffle(f)
        return [target, *f]
    if strategy == "fix_y_end":
        f = list(features)
        rng.shuffle(f)
        return [*f, target]
    if strategy == "identity":
        return [target, *features]
    raise ValueError(f"unknown permutation {strategy!r}; have {STRATEGIES}")


def label_free_fraction(m: int, strategy: str) -> float:
    """Expected fraction of FEATURE clauses trained without Y in their context."""
    if strategy in ("fix_y", "identity"):
        return 0.0
    if strategy == "fix_y_end":
        return 1.0
    if strategy == "permute_xy":
        return float(np.mean([k / m for k in range(m + 1)])) if m else 0.0
    raise ValueError(strategy)
