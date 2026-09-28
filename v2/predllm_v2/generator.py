"""
PredLLM v2 generator.

fit()      : schema (N1) -> weighted value-only fine-tuning (N3), with the same model cache as v1
generate() : pool = gamma*n rows -> field-wise constrained x (N2) -> exact posterior y (N4)
             -> map to the shared encoded space -> DCR gate + discriminator SIR down to n rows (N5)
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
VERSION: Any = None

class PredLLMV2:
    def __init__(self, method, cfg, split, run_dir, seed):
        pass

    def _ft_cfg(self):
        pass

    def fingerprint(self):
        pass

    def model_dir(self):
        pass

    def fit(self):
        pass

    def _ready(self):
        pass

    def _to_frames(self, rows, y):
        pass

    def generate(self, n):
        pass

def make_generator(method, cfg, split, run_dir, seed):
    pass

def load_generator(run_dir):
    pass
