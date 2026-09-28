"""
Sampling phase (Alg. 1 lines 13-22) - LLM USED HERE (free-running text generation).

Next token ~ softmax(z / T) (Eq. 2), T = 0.7, HF defaults top_k=50 (as in GReaT/Pred-LLM code).
  feature-conditional : for each feature X_i: n/M prompts "X_i is v," with v ~ empirical p(X_i)
  target-conditional  : n prompts "target is y," with y ~ empirical p(Y)
Generated text is parsed back to rows; rows missing a required column or with a non-numeric value
are rejected and re-generated (bounded by max_rounds).
"""

from typing import Any, Optional, List, Dict, Union

def generate_text(model, tok, prompts, max_new_tokens, temperature, top_k, batch_size):
    pass

def _fmt_start(v, col, task, cfg):
    """Must match serialize.row_fields exactly so the start token is in-distribution."""
    pass

def sample_rows(model, tok, train, features, task, n, strategy, require_target, cfg, rng):
    """Returns (rows, stats). Rows have all features (+ target if generated)."""
    pass
