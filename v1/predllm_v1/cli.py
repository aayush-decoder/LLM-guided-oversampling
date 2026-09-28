"""
v1 CLI:  python -m predllm_v1 <command> [flags]

run        one dataset x method(s) x seeds: train -> generate -> evaluate   (--method / --methods)
benchmark  datasets x methods (Table II). Defaults: --datasets paper --methods paper
ablation   Table III rows on --datasets (default: classification)
train      fine-tune only (fills the model cache that later runs reuse)
generate   INFERENCE: sample new synthetic rows from a finished run's fine-tuned LLM
infer      INFERENCE: the fine-tuned LLM as a predictor p(y|x) on the real test set
evaluate   recompute metrics.json from a run's synthetic_encoded.csv
attention  Fig. 4: label-token attention, target last vs target first
smoke      end-to-end check on iris that verifies every artifact
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

def cmd_attention(a):
    pass

def main(argv):
    pass
