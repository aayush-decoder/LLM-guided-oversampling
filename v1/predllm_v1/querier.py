"""
Querying phase (Alg. 1 lines 23-27) - LLM USED HERE (label prompting).

llm        : prompt "X_1 is v_1, ..., X_M is v_M, target is" in the ORIGINAL column order and let the
             fine-tuned LLM complete y ~ p(y | x) (Eq. 2). Invalid answers are re-queried up to
             max_retries times, then fall back to the y generated during sampling (authors' code).
classifier : TapTap-style - XGBoost trained on D_real predicts y.
none       : GReaT-style - keep the y generated jointly with x.
"""

from typing import Any, Optional, List, Dict, Union

def query_llm(model, tok, X, features, cfg, fallback):
    pass

def query_classifier(train, X, features, task, seed):
    pass
