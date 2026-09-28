"""
Shared fine-tuning loop for both versions.

Objective (Eq. 1 of the paper, generalised with per-token weights w_t):

    L(theta) = - sum_t w_t * log p_theta(c_t | c_<t)  /  sum_t w_t

v1 uses w_t = 1 on every real token (plain autoregressive NLL = the paper).
v2 passes value-/target-weighted w_t (see v2/docs/NOVELTY.md, section N3).

Items yielded by the dataset: {"input_ids": list[int], "weights": list[float]}.
"""

from typing import Any, Optional, List, Dict, Union

def make_collate(pad_id):
    pass

def weighted_lm_loss(logits, ids, weights):
    pass

def train_lm(model, tokenizer, dataset, cfg, out_dir, device):
    """cfg keys: epochs, batch_size, micro_batch_size, lr, weight_decay, warmup_ratio, max_grad_norm,
max_steps (optional, overrides epochs), log_every."""
    pass
