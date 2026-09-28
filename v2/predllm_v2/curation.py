"""
Post-hoc curation of an over-generated pool (v2 novelty N5). No LLM involved.

1. Privacy gate (DCR): tau = alpha-quantile of DCR(real holdout -> real reference). Pool rows closer
   than tau to a real training row are treated as near-copies and dropped (at most max_drop of the pool).
2. Discriminator-guided sampling-importance-resampling (SIR): a cross-fitted classifier D separates
   real (1) from synthetic (0). By the density-ratio trick
        p_real(x) / p_syn(x)  ~=  (n_syn / n_real) * D(x) / (1 - D(x))
   Weights w = clip(ratio, 1/c, c)^beta; n rows are drawn without replacement with prob ~ w, which
   moves the synthetic distribution towards the real one (beta=0 disables, beta=1 is exact SIR).
"""

from typing import Any, Optional, List, Dict, Union

def privacy_gate(real, pool, alpha, max_drop, seed):
    pass

def sir_weights(real, pool, beta, clip, folds, seed):
    pass

def curate(real_enc, pool_enc, n, features, cfg, seed):
    """Returns (selected rows, stats). Distances use features only; the discriminator sees (x, y)."""
    pass
