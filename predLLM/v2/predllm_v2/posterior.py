"""Label querying via a permutation-marginalised exact posterior (v2 novelty N4) - LLM USED HERE.

Classification: for R feature orders pi_1..pi_R (pi_1 = original order, the rest random) score every
class exactly:
        p_r(y=c | x) = p(" c" EOS | pi_r(x), "target is") / sum_c' p(" c'" EOS | ...)
        p_hat(y | x) = 1/R * sum_r p_r(y | x)            (mixture over orders)
then  y ~ p_hat^(1/T_y)  (label_decode=sample)  or  y = argmax p_hat  (label_decode=argmax).

Regression: y_r ~ FSM-constrained decode after pi_r(x); y = mean_r y_r (reg_aggregate=mean) or y_1.
"""
from __future__ import annotations

import numpy as np
import torch

from .codec import FieldCodec
from .decoding import NumericConstraint, decode_numeric, score_continuations


def _orders(features, R, rng):
    return [list(features)] + [[features[i] for i in rng.permutation(len(features))] for _ in range(R - 1)]


def class_posterior(model, codec: FieldCodec, rows: list[dict], R: int, cfg: dict, rng) -> np.ndarray:
    sch = codec.schema
    classes = sch.cats[sch.target]
    cands = [codec.val_ids(c) + [codec.eos] for c in classes]
    probs = np.zeros((len(rows), len(classes)))
    bs = cfg["batch_size"]
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        for order in _orders(sch.features, R, rng):
            prefixes = [codec.prefix(r, order, then_key=sch.target) for r in chunk]
            lp = score_continuations(model, prefixes, cands, codec.tok.pad_token_id, cfg["max_score_elems"],
                                     cfg["kv_cache"]).cpu().numpy()
            p = np.exp(lp - lp.max(1, keepdims=True))
            probs[s:s + len(chunk)] += p / p.sum(1, keepdims=True) / R
    return probs


def decide_labels(probs: np.ndarray, how: str, temperature: float, rng) -> np.ndarray:
    if how == "argmax":
        return probs.argmax(1)
    z = np.log(probs + 1e-12) / max(temperature, 1e-4)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.array([rng.choice(p.shape[1], p=pi) for pi in p])


def regression_values(model, codec: FieldCodec, rows: list[dict], R: int, cfg: dict, cons: NumericConstraint,
                      gen: torch.Generator, rng, empirical: np.ndarray) -> np.ndarray:
    sch = codec.schema
    spec = sch.numeric_spec(sch.target)
    lo, hi = sch.ranges[sch.target]
    draws = np.full((len(rows), R), np.nan)
    bs = cfg["batch_size"]
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        for r, order in enumerate(_orders(sch.features, R, rng)):
            prefixes = [codec.prefix(row, order, then_key=sch.target) for row in chunk]
            vals = decode_numeric(model, prefixes, spec, codec.eos, cons, codec.tok.pad_token_id,
                                  cfg["temperature_num"], gen)
            draws[s:s + len(chunk), r] = [np.clip(float(v), lo, hi) if v is not None else np.nan for v in vals]
    y = np.nanmean(draws, 1) if cfg["reg_aggregate"] == "mean" else draws[:, 0]
    miss = np.isnan(y)
    y[miss] = rng.choice(empirical, size=int(miss.sum()))
    return y
