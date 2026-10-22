"""Querying phase (Alg. 1 lines 23-27) - LLM USED HERE (label prompting).

  llm        : prompt "X_1 is v_1, ..., X_M is v_M, target is" in the ORIGINAL column order and let the
               fine-tuned LLM complete y ~ p(y | x) (Eq. 2). Invalid answers are re-queried up to
               max_retries times, then fall back to the y generated during sampling (authors' code).
  classifier : TapTap-style - XGBoost trained on D_real predicts y.
  none       : GReaT-style - keep the y generated jointly with x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predLLM.common.tabgen.data.preprocess import TARGET
from predLLM.common.tabgen.eval.tstr import xgb_model
from predLLM.common.tabgen.utils import format_number

from .sampler import generate_text
from .serialize import parse


def query_llm(model, tok, X: pd.DataFrame, features: list[str], cfg: dict, fallback: np.ndarray) -> tuple[np.ndarray, dict]:
    q, s = cfg["query"], cfg["sampling"]
    suffix = f", {TARGET} is" if q["force_target_key"] else ""
    prompts = [", ".join(f"{f} is {format_number(r[f], cfg['data']['decimals'])}" for f in features) + suffix
               for r in X.to_dict("records")]
    y = np.full(len(prompts), np.nan)
    todo = np.arange(len(prompts))
    for _ in range(1 + q["max_retries"]):
        if len(todo) == 0:
            break
        texts = generate_text(model, tok, [prompts[i] for i in todo], q["max_new_tokens"], s["temperature"],
                              s["top_k"], s["batch_size"])
        for i, t in zip(todo, texts):
            v = parse(t, [TARGET]).get(TARGET)
            if v is not None:
                y[i] = v
        todo = np.where(np.isnan(y))[0]
    n_fallback = len(todo)
    y[todo] = fallback[todo]
    return y, {"query_fallbacks": int(n_fallback)}


def query_classifier(train: pd.DataFrame, X: pd.DataFrame, features: list[str], task: str, seed: int) -> np.ndarray:
    m = xgb_model(task, seed).fit(train[features].to_numpy(float), train[TARGET].to_numpy())
    return m.predict(X[features].to_numpy(float))
