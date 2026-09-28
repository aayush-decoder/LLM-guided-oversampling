"""
Non-LLM baselines from the paper via SDV: CTGAN, TVAE, CopulaGAN.

They are trained in the *clean* space (real categories, real units), which is their native input and
the stronger setting, and their output is mapped to the encoded space by the shared preprocessor.
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
SDV_METHODS: Any = None

def _metadata(df, num_cols, cat_cols, target, task):
    pass

def sdv_fit_sample(method, train_clean, num_cols, cat_cols, target, task, n, epochs, seed, use_gpu):
    pass
