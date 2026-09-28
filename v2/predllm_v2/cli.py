"""
v2 CLI:  python -m predllm_v2 <command> [flags]   (same flags as v1)

run        one dataset x preset(s) x seeds                      (--method / --methods)
benchmark  datasets x presets. Defaults: --datasets paper --methods compare
ablation   every v2 preset (N1..N5 switched off one at a time) on --datasets
train      fine-tune only
generate   INFERENCE: sample synthetic rows from a finished run
infer      INFERENCE: permutation-marginalised LLM posterior as a predictor on the real test set
evaluate   recompute metrics.json
smoke      end-to-end check on iris (tiny-gpt2 works: the decoder cannot emit invalid rows)
methods    list presets and groups
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
CONFIG: Any = None

def cmd_run(a, datasets, methods):
    pass

def cmd_train(a):
    pass

def cmd_generate(a):
    pass

def cmd_infer(a):
    pass

def cmd_evaluate(a):
    pass

def main(argv):
    pass
