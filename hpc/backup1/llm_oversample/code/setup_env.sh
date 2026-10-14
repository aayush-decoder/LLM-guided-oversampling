#!/bin/bash
# One-time environment setup. Run on the LOGIN node (needs internet), from the project root:
#     bash setup_env.sh
# It creates a conda env "imbllm", installs packages, and pre-downloads DistilGPT2 into ./.hf_cache
set -euo pipefail

ENV_NAME=imbllm

# If your cluster uses modules, uncomment/adjust (check with: module avail):
# module load anaconda3
# module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
# The default PyTorch wheel bundles its own CUDA runtime; it only needs a new-enough NVIDIA driver.
python -m pip install torch
python -m pip install -r requirements.txt

export HF_HOME="$PWD/.hf_cache"
python - <<'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("distilgpt2"); AutoModelForCausalLM.from_pretrained("distilgpt2")
print("distilgpt2 cached")
import torch, xgboost, sklearn, imblearn
print("torch", torch.__version__, "| xgboost", xgboost.__version__, "| sklearn", sklearn.__version__, "| imblearn", imblearn.__version__)
PYEOF
echo "Env '$ENV_NAME' ready."
