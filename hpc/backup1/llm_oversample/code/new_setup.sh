
#!/usr/bin/env bash
# One-shot, resumable setup for ImbLLM on SVNIT HPC.
# Run from ~/llm_oversample/code with internet access.

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-imbllm}"
PROJECT_ROOT="$(pwd)"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export PIP_CONFIG_FILE=/dev/null
export PIP_DEFAULT_TIMEOUT=60
export PIP_RETRIES=5

trap 'echo "[ERROR] Setup failed at line $LINENO. See the message above." >&2' ERR

if [[ ! -f requirements.txt ]]; then
    echo "ERROR: Run this script from the directory containing requirements.txt."
    exit 1
fi

echo "=== 1. Detecting HPC environment ==="
if command -v module >/dev/null 2>&1; then
    module list 2>&1 || true
fi

if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
fi

# Create the environment only if it does not already exist.
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

echo "Python: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

echo "=== 2. Select PyTorch CUDA wheel ==="

# Set TORCH_CUDA=cu121, cu124, cu126, cu128, or cpu
# explicitly when necessary. Do not confuse toolkit version with driver support.
TORCH_CUDA="${TORCH_CUDA:-}"

if [[ -z "$TORCH_CUDA" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER="$(nvidia-smi --query-gpu=driver_version \
            --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"

        if [[ -z "$DRIVER" ]]; then
            echo "ERROR: Could not detect an NVIDIA driver."
            echo "Set TORCH_CUDA explicitly after checking HPC GPU-node details."
            exit 1
        fi

        echo "NVIDIA driver: $DRIVER"

        # Conservative minimum Linux driver versions for these CUDA runtimes.
        if [[ "$(printf '%s\n' 570.26 "$DRIVER" | sort -V | head -n1)" == "570.26" ]]; then
            TORCH_CUDA=cu128
        elif [[ "$(printf '%s\n' 560.28 "$DRIVER" | sort -V | head -n1)" == "560.28" ]]; then
            TORCH_CUDA=cu126
        elif [[ "$(printf '%s\n' 550.54 "$DRIVER" | sort -V | head -n1)" == "550.54" ]]; then
            TORCH_CUDA=cu124
        elif [[ "$(printf '%s\n' 525.60 "$DRIVER" | sort -V | head -n1)" == "525.60" ]]; then
            TORCH_CUDA=cu121
        else
            echo "ERROR: Driver $DRIVER is too old for the automatic CUDA choices."
            echo "Check with your HPC administrator before choosing a wheel."
            exit 1
        fi
    else
        echo "No visible NVIDIA driver on this node."
        echo "Check the GPU-node driver and rerun with TORCH_CUDA=cuXXX."
        echo "Example: TORCH_CUDA=cu124 bash setup_env.sh"
        exit 1
    fi
fi

case "$TORCH_CUDA" in
    cu121|cu124|cu126|cu128)
        TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_CUDA"
        ;;
    cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        ;;
    *)
        echo "ERROR: Unsupported TORCH_CUDA=$TORCH_CUDA"
        exit 1
        ;;
esac

echo "Selected PyTorch index: $TORCH_INDEX"

echo "=== 3. Install PyTorch ==="
# Official PyTorch wheel index; ignores broken global pip indexes.
python -m pip install --index-url "$TORCH_INDEX" torch

echo "=== 4. Install project dependencies ==="
# Install the remaining requirements without letting pip replace the
# selected PyTorch build with an unrelated default wheel.
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([<=>!~[]|[[:space:]]|$)' \
    requirements.txt > "$TMP_REQ" || true

python -m pip install \
    --index-url https://pypi.org/simple \
    -r "$TMP_REQ"

echo "=== 5. Verify PyTorch and CUDA build ==="
python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available on this node:", torch.cuda.is_available())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GiB")
PY

echo "=== 6. Download DistilGPT2 ==="
mkdir -p "$HF_HOME"

python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "distilgpt2"
AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name)
print("DistilGPT2 downloaded and cached.")
PY

echo "=== 7. Final dependency check ==="
python -m pip check

echo
echo "Setup complete."
echo "Environment: $ENV_NAME"
echo "Model cache: $HF_HOME"
echo "PyTorch CUDA build: $TORCH_CUDA"
echo "Remember: CUDA availability must be checked inside a GPU allocation."

