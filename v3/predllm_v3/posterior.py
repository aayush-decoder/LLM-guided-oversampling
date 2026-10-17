"""Label querying via a permutation-marginalised exact posterior (v3 novelty).

Classification: for R feature orders pi_1..pi_R score every class exactly:
        p_r(y=c | x) = p(" c" EOS | pi_r(x), "target is") / sum_c' p(" c'" EOS | ...)
        p_hat(y | x) = 1/R * sum_r p_r(y | x)
then apply optional minority prior tilting and sample/argmax.

Regression: y_r ~ FSM-constrained decode after pi_r(x); y = mean_r y_r.
"""
from __future__ import annotations

import numpy as np
import torch

from .codec import FieldCodec
from .decoding import NumericConstraint, decode_numeric, score_continuations


def _orders(features, R, rng):
    return [list(features)] + [[features[i] for i in rng.permutation(len(features))] for _ in range(R - 1)]


def class_posterior(
    model,
    codec: FieldCodec,
    rows: list[dict],
    R: int,
    cfg: dict,
    rng,
    minority_prior_boost: float = 0.0,
) -> np.ndarray:
    sch = codec.schema
    classes = sch.cats[sch.target]
    cands = [codec.val_ids(c) + [codec.eos] for c in classes]
    probs = np.zeros((len(rows), len(classes)))
    bs = cfg["batch_size"]
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        for order in _orders(sch.features, R, rng):
            prefixes = [codec.prefix(r, order, then_key=sch.target) for r in chunk]
            lp = score_continuations(
                model,
                prefixes,
                cands,
                codec.tok.pad_token_id,
                cfg.get("max_score_elems", 3e7),
                cfg.get("kv_cache", True),
            ).cpu().numpy()
            p = np.exp(lp - lp.max(1, keepdims=True))
            probs[s:s + len(chunk)] += p / p.sum(1, keepdims=True) / R

    # Optional prior tilt towards minority classes if configured
    if minority_prior_boost > 0.0 and sch.minority_classes:
        min_indices = [i for i, c in enumerate(classes) if str(c) in sch.minority_classes]
        if min_indices:
            for mi in min_indices:
                probs[:, mi] *= (1.0 + minority_prior_boost)
            probs /= probs.sum(1, keepdims=True)

    return probs


def decide_labels(probs: np.ndarray, how: str, temperature: float, rng) -> np.ndarray:
    K = probs.shape[1]
    if how == "argmax":
        labels = probs.argmax(1)
    else:
        p_smooth = probs + 1e-3
        p_smooth /= p_smooth.sum(1, keepdims=True)
        z = np.log(p_smooth + 1e-12) / max(temperature, 1e-4)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        labels = np.array([rng.choice(p.shape[1], p=pi) for pi in p])

    if len(labels) >= K:
        missing = set(range(K)) - set(np.unique(labels))
        for m in missing:
            cand_idx = int(probs[:, m].argmax())
            labels[cand_idx] = m
    return labels


def regression_values(
    model,
    codec: FieldCodec,
    rows: list[dict],
    R: int,
    cfg: dict,
    cons: NumericConstraint,
    gen: torch.Generator,
    rng,
    empirical: np.ndarray,
) -> np.ndarray:
    sch = codec.schema
    spec = sch.numeric_spec(sch.target)
    lo, hi = sch.ranges[sch.target]
    draws = np.full((len(rows), R), np.nan)
    bs = cfg["batch_size"]
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        for r, order in enumerate(_orders(sch.features, R, rng)):
            prefixes = [codec.prefix(row, order, then_key=sch.target) for row in chunk]
            vals = decode_numeric(
                model,
                prefixes,
                spec,
                codec.eos,
                cons,
                codec.tok.pad_token_id,
                cfg["temperature_num"],
                gen,
            )
            draws[s:s + len(chunk), r] = [np.clip(float(v), lo, hi) if v is not None else np.nan for v in vals]
    y = np.nanmean(draws, 1) if cfg["reg_aggregate"] == "mean" else draws[:, 0]
    miss = np.isnan(y)
    y[miss] = rng.choice(empirical, size=int(miss.sum()))
    return y
