#!/bin/bash
#SBATCH --job-name=imbllm
#SBATCH --partition=gpu          # find with: sinfo -s     (pick a GPU partition)
#SBATCH --account=svnitcluster            # find with: sacctmgr show assoc user=$USER format=account%30   (delete line if not needed)
#SBATCH --gres=shard:20                # you said you can get ~15-25 GPU "shards". If sbatch rejects this, use: --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00                # max allowed: sinfo -o "%P %l". Re-submitting resumes, so a shorter limit is fine.
#SBATCH --signal=B:USR1@180            # warn the script 3 min before the time limit
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -uo pipefail
mkdir -p logs

# --- Environment ---------------------------------------------------------
# module load anaconda3
# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate imbllm

export HF_HOME="$PWD/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

RUN_NAME="${RUN_NAME:-exp1}"           # same name => resume. Override: sbatch --export=ALL,RUN_NAME=exp2 run_imbllm.slurm
DATA_DIR="${DATA_DIR:-data/raw}"

# --- Sanity checks --------------------------------------------------------
python - << 'PYEOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, "| %.1f GB visible" % (p.total_memory / 1e9))
PYEOF
if [ ! -f main.py ] || [ ! -d "$DATA_DIR" ]; then
    echo "main.py or $DATA_DIR not found -- submit from the project root." >&2
    exit 1
fi

# --- Run (background + trap so Slurm's time-limit warning stops the script cleanly) ----
python -u main.py --data_dir "$DATA_DIR" --run_name "$RUN_NAME" &
PY_PID=$!
trap 'echo "[slurm] time-limit/cancel signal -> asking main.py to stop"; kill -TERM "$PY_PID" 2>/dev/null' USR1 TERM
wait "$PY_PID"
wait "$PY_PID" 2>/dev/null          # 2nd wait collects the real exit code after a trapped signal
RC=$?

echo "main.py exit code: $RC"
echo "Status board:  cat runs/$RUN_NAME/status.txt"
echo "Not finished? Just resubmit this same script: finished datasets are skipped."
exit $RC
