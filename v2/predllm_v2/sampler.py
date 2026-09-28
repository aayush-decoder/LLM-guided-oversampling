"""
Schema-driven, field-wise sampling of x (v2 novelty N2) - LLM USED HERE.

For a batch of B rows the decoder writes every key itself and asks the LLM only for VALUES:

    order = [X_c] + shuffle(other features)      X_c cycled uniformly = feature-conditional (paper)
    X_c   ~ empirical p(X_c)                      (paper's condition)
    categorical X_j : v ~ p_T(v | prefix) = p(v, SEP | prefix)^(1/T_cat) / Z   over the column's categories
    numerical   X_j : digits sampled token-level at T_num under the numeric FSM, clipped to the train range

The target is NOT generated here; it is queried afterwards from p(y | x) (posterior.py).
Every returned row is valid by construction (validity_rate = 1).
"""

from typing import Any, Optional, List, Dict, Union

def sample_categorical(model, codec, prefixes, col, temperature, rng, cfg):
    pass

def sample_numeric(model, codec, prefixes, col, cons, temperature, gen, empirical, rng):
    pass

def sample_features(model, codec, train_text, n, cfg, rng, gen, cons):
    pass
