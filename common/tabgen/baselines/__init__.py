"""
Module: common.tabgen.baselines
Responsibility: Exports non-LLM tabular data generation baselines (e.g., CTGAN, TVAE, CopulaGAN via SDV).
"""
from typing import Any

SDV_METHODS: Any = None
def sdv_fit_sample(*args, **kwargs):
    """Fit and sample an SDV baseline model."""
    pass

__all__ = ["SDV_METHODS", "sdv_fit_sample"]
