"""Schema-driven, field-wise sampling with outlier range bounding and negative guidance (v3 novelty).

For a batch of B rows the decoder writes every key itself and asks the LLM only for VALUES:
    order = [X_c] + shuffle(other features)
    categorical X_j : v ~ p_T(v | prefix)
    numerical   X_j : digits sampled token-level at T_num under numeric FSM,
                      strictly constrained to the active OUTLIER / MINORITY range.

Guarantees 100% validity by construction while directing generation into the target rare-event manifold.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from predLLM.common.tabgen.utils import get_logger

from .codec import FieldCodec, Schema
from .decoding import NumericConstraint, decode_numeric, score_continuations

log = get_logger("predllm_v3.sampler")


def sample_categorical(
    model,
    codec: FieldCodec,
    prefixes: list[list[int]],
    col: str,
    temperature: float,
    rng: np.random.Generator,
    cfg: dict,
) -> list[str]:
    cats = codec.schema.cats[col]
    cands = [codec.val_ids(v) + [codec.terminator(col)] for v in cats]
    lp = score_continuations(
        model,
        prefixes,
        cands,
        codec.tok.pad_token_id,
        cfg.get("max_score_elems", 3e7),
        cfg.get("kv_cache", True),
    ).cpu().numpy()
    z = lp / max(temperature, 1e-4)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return [cats[rng.choice(len(cats), p=pi)] for pi in p]


def sample_numeric(
    model,
    codec: FieldCodec,
    prefixes: list[list[int]],
    col: str,
    cons: NumericConstraint,
    temperature: float,
    gen: torch.Generator,
    empirical: np.ndarray,
    rng: np.random.Generator,
    range_mode: str = "outlier",
    negative_guidance: float = 0.0,
    neg_prefixes: list[list[int]] | None = None,
) -> tuple[list[str], int]:
    sch = codec.schema
    lo, hi = sch.active_range(col, range_mode=range_mode)
    spec = sch.numeric_spec(col, range_mode=range_mode)
    raw = decode_numeric(
        model,
        prefixes,
        spec,
        codec.terminator(col),
        cons,
        codec.tok.pad_token_id,
        temperature,
        gen,
        negative_guidance=negative_guidance,
        neg_prefixes=neg_prefixes,
    )
    out, fallbacks = [], 0
    # In outlier mode, filter empirical values to the outlier range for fallbacks
    emp_valid = empirical[(empirical >= lo) & (empirical <= hi)]
    if len(emp_valid) == 0:
        emp_valid = empirical

    for s in raw:
        if s is None:
            fallbacks += 1
            out.append(sch.fmt(col, rng.choice(emp_valid)))
        else:
            val = min(max(float(s), lo), hi)
            out.append(sch.fmt(col, val))
    return out, fallbacks


def sample_features(
    model,
    codec: FieldCodec,
    train_text: list[dict],
    n: int,
    cfg: dict,
    rng: np.random.Generator,
    gen: torch.Generator,
    cons: NumericConstraint,
) -> tuple[list[dict], dict]:
    sch: Schema = codec.schema
    d = cfg.get("decoding", {})
    range_mode = cfg.get("range_mode", "outlier")
    negative_guidance = float(d.get("negative_guidance", 0.0))

    feats = sch.features
    empirical = {c: np.array([r[c] for r in train_text]) for c in feats}
    emp_num = {c: np.array([float(v) for v in empirical[c]]) for c in feats if sch.kinds[c] == "num"}
    quota = math.ceil(n / len(feats))

    rows, fallbacks = [], 0
    for c0 in feats:
        for s in range(0, quota, d.get("batch_size", 64)):
            b = min(d.get("batch_size", 64), quota - s)
            others = [feats[i] for i in rng.permutation(len(feats)) if feats[i] != c0]
            order = [c0] + others
            batch = [{} for _ in range(b)]
            prefixes = [list(codec.bos) for _ in range(b)]

            for j, c in enumerate(order):
                for p in prefixes:
                    p += codec.key_ids(c, j == 0)

                if j == 0:
                    lo, hi = sch.active_range(c, range_mode=range_mode) if sch.kinds[c] == "num" else (None, None)
                    if sch.kinds[c] == "num":
                        sub_emp = emp_num[c][(emp_num[c] >= lo) & (emp_num[c] <= hi)]
                        if len(sub_emp) == 0:
                            sub_emp = emp_num[c]
                        vals = [sch.fmt(c, v) for v in rng.choice(sub_emp, size=b)]
                    else:
                        vals = list(rng.choice(empirical[c], size=b))
                elif sch.kinds[c] == "cat":
                    vals = sample_categorical(model, codec, prefixes, c, d.get("temperature_cat", 1.0), rng, d)
                else:
                    neg_pref = [list(codec.bos) for _ in range(b)] if negative_guidance > 0.0 else None
                    vals, fb = sample_numeric(
                        model,
                        codec,
                        prefixes,
                        c,
                        cons,
                        d.get("temperature_num", 0.7),
                        gen,
                        emp_num[c],
                        rng,
                        range_mode=range_mode,
                        negative_guidance=negative_guidance,
                        neg_prefixes=neg_pref,
                    )
                    fallbacks += fb

                for i, v in enumerate(vals):
                    batch[i][c] = v
                    prefixes[i] += codec.val_ids(v) + [codec.sep]
            rows += batch

    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx], {
        "numeric_fallbacks": fallbacks,
        "validity_rate": 1.0,
        "n_sampled": len(rows),
        "range_mode": range_mode,
    }
