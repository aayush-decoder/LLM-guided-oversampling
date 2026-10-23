"""Schema-driven, field-wise sampling of x (v2 novelty N2) - LLM USED HERE.

For a batch of B rows the decoder writes every key itself and asks the LLM only for VALUES:

    order = [X_c] + shuffle(other features)      X_c cycled uniformly = feature-conditional (paper)
    X_c   ~ empirical p(X_c)                      (paper's condition)
    categorical X_j : v ~ p_T(v | prefix) = p(v, SEP | prefix)^(1/T_cat) / Z   over the column's categories
    numerical   X_j : digits sampled token-level at T_num under the numeric FSM, clipped to the train range

The target is NOT generated here; it is queried afterwards from p(y | x) (posterior.py).
Every returned row is valid by construction (validity_rate = 1).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from predLLM.common.tabgen.utils import get_logger

from .codec import FieldCodec, Schema
from .decoding import NumericConstraint, decode_numeric, score_continuations

log = get_logger("predllm_v2.sampler")


def sample_categorical(model, codec: FieldCodec, prefixes, col: str, temperature: float, rng, cfg) -> list[str]:
    cats = codec.schema.cats[col]
    cands = [codec.val_ids(v) + [codec.terminator(col)] for v in cats]
    lp = score_continuations(model, prefixes, cands, codec.tok.pad_token_id, cfg["max_score_elems"],
                             cfg["kv_cache"]).cpu().numpy()                                      # (B, K)
    z = lp / max(temperature, 1e-4)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return [cats[rng.choice(len(cats), p=pi)] for pi in p]


def sample_numeric(model, codec: FieldCodec, prefixes, col: str, cons: NumericConstraint, temperature: float,
                   gen: torch.Generator, empirical: np.ndarray, rng) -> tuple[list[str], int]:
    sch = codec.schema
    lo, hi = sch.ranges[col]
    raw = decode_numeric(model, prefixes, sch.numeric_spec(col), codec.terminator(col), cons,
                         codec.tok.pad_token_id, temperature, gen)
    out, fallbacks = [], 0
    for s in raw:
        if s is None:
            fallbacks += 1
            out.append(sch.fmt(col, rng.choice(empirical)))
        else:
            out.append(sch.fmt(col, min(max(float(s), lo), hi)))
    return out, fallbacks


def sample_features(model, codec: FieldCodec, train_text: list[dict], n: int, cfg: dict, rng: np.random.Generator,
                    gen: torch.Generator, cons: NumericConstraint) -> tuple[list[dict], dict]:
    sch: Schema = codec.schema
    d = cfg["decoding"]
    feats = sch.features
    empirical = {c: np.array([r[c] for r in train_text]) for c in feats}
    emp_num = {c: np.array([float(v) for v in empirical[c]]) for c in feats if sch.kinds[c] == "num"}
    quota = math.ceil(n / len(feats))                               # n / M rows per conditioning feature
    rows, fallbacks = [], 0
    for c0 in feats:
        for s in range(0, quota, d["batch_size"]):
            b = min(d["batch_size"], quota - s)
            others = [feats[i] for i in rng.permutation(len(feats)) if feats[i] != c0]
            order = [c0] + others
            batch = [{} for _ in range(b)]
            prefixes = [list(codec.bos) for _ in range(b)]
            for j, c in enumerate(order):
                for p in prefixes:
                    p += codec.key_ids(c, j == 0)
                if j == 0:
                    vals = list(rng.choice(empirical[c], size=b))
                elif sch.kinds[c] == "cat":
                    vals = sample_categorical(model, codec, prefixes, c, d["temperature_cat"], rng, d)
                else:
                    vals, fb = sample_numeric(model, codec, prefixes, c, cons, d["temperature_num"], gen, emp_num[c], rng)
                    fallbacks += fb
                for i, v in enumerate(vals):
                    batch[i][c] = v
                    prefixes[i] += codec.val_ids(v) + [codec.sep]
            rows += batch
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx], {"numeric_fallbacks": fallbacks, "validity_rate": 1.0,
                                    "n_sampled": len(rows)}
