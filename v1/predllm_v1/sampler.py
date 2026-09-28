"""Sampling phase (Alg. 1 lines 13-22) - LLM USED HERE (free-running text generation).

Next token ~ softmax(z / T) (Eq. 2), T = 0.7, HF defaults top_k=50 (as in GReaT/Pred-LLM code).
  feature-conditional : for each feature X_i: n/M prompts "X_i is v," with v ~ empirical p(X_i)
  target-conditional  : n prompts "target is y," with y ~ empirical p(Y)
Generated text is parsed back to rows; rows missing a required column or with a non-numeric value
are rejected and re-generated (bounded by max_rounds).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from tabgen.data.preprocess import TARGET
from tabgen.utils import format_number, get_logger

from .serialize import parse

log = get_logger("predllm_v1.sampler")


@torch.no_grad()
def generate_text(model, tok, prompts: list[str], max_new_tokens: int, temperature: float, top_k: int,
                  batch_size: int) -> list[str]:
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_k=top_k, top_p=1.0,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    tok.padding_side = "right"
    return out


def _fmt_start(v, col: str, task: str, cfg: dict) -> str:
    """Must match serialize.row_fields exactly so the start token is in-distribution."""
    if col == TARGET:
        return str(int(v)) if task == "classification" else format_number(v, cfg["sampling"]["target_decimals"])
    return format_number(v, cfg["data"]["decimals"])


def sample_rows(model, tok, train: pd.DataFrame, features: list[str], task: str, n: int, strategy: str,
                require_target: bool, cfg: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Returns (rows, stats). Rows have all features (+ target if generated)."""
    s = cfg["sampling"]
    required = features + ([TARGET] if require_target else [])
    columns = features + [TARGET]
    if strategy == "feature":
        plan = [(f, math.ceil(n / len(features))) for f in features]     # n = N / M per feature
    elif strategy == "target":
        plan = [(TARGET, n)]
    else:
        raise ValueError(strategy)
    rows, n_generated = [], 0
    for col, quota in plan:
        got, rounds = [], 0
        pool = train[col].to_numpy()
        while len(got) < quota and rounds < s["max_rounds"]:
            need = quota - len(got)
            k = max(need, min(s["batch_size"], 2 * need))            # over-generate a little
            starts = rng.choice(pool, size=k, replace=True)
            prompts = [f"{col} is {_fmt_start(v, col, task, cfg)}," for v in starts]
            texts = generate_text(model, tok, prompts, s["max_new_tokens"], s["temperature"], s["top_k"],
                                  s["batch_size"])
            n_generated += len(texts)
            for t in texts:
                r = parse(t, columns)
                if all(c in r for c in required):
                    got.append(r)
            rounds += 1
        if len(got) < quota:
            log.warning(f"start column {col}: only {len(got)}/{quota} valid rows after {rounds} rounds")
        rows += got[:quota]
    df = pd.DataFrame(rows)
    rng_idx = rng.permutation(len(df))[:n]
    df = df.iloc[rng_idx].reset_index(drop=True)
    stats = {"n_generated_texts": n_generated, "n_valid": len(rows),
             "validity_rate": round(len(rows) / max(n_generated, 1), 4)}
    return df, stats
