"""
PredLLM v1 generator = Algorithm 1 of the paper, parameterised by a MethodSpec.

fit()      : fine-tune the LLM on D'_real (or reuse an identical fine-tune from the model cache)
generate() : sample x (feature- or target-conditional) -> query y (llm / classifier / none) -> clip

Model cache: <version>/runs/<dataset>/_models/<fingerprint>/ . Methods whose fine-tuning is identical
(same data split, LLM, permutation, augmentation, hyper-parameters) share one model, e.g. pred_llm,
abl_x_target_llm and abl_x_target_none. Each run dir gets model_ref.json pointing at the model used.
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
VERSION: Any = None

def finetune_cfg(cfg, spec):
    """Resolves mode-dependent defaults (LoRA needs a higher LR and smaller micro-batches)."""
    pass

class PredLLMV1:
    def __init__(self, method, cfg, split, run_dir, seed):
        pass

    def fingerprint(self):
        pass

    def model_dir(self):
        pass

    def fit(self):
        pass

    def _set_lengths(self):
        pass

    def generate(self, n):
        pass

def make_generator(method, cfg, split, run_dir, seed):
    pass

def load_generator(run_dir):
    """Rebuilds a fitted generator from a finished run dir (for `generate` / `infer` / `attention`)."""
    pass
