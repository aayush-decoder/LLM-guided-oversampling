#!/usr/bin/env bash
# ==============================================================================
# ML RESEARCH PROJECT - AUTOMATED RUNNER (v1 + v2)
# Optimized for: NVIDIA RTX 5050 GPU | 16 GB RAM | Windows (Git Bash)
# LLMs Included: distilgpt2 (paper baseline), Qwen2.5-0.5B, Qwen2.5-1.5B (LoRA), tiny-gpt2
# ==============================================================================

# Enable UTF-8 encoding
export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1

# ANSI Color codes for kid-friendly and readable terminal output
CLR_RESET="\033[0m"
CLR_BOLD="\033[1m"
CLR_GREEN="\033[1;32m"
CLR_CYAN="\033[1;36m"
CLR_YELLOW="\033[1;33m"
CLR_RED="\033[1;31m"
CLR_MAGENTA="\033[1;35m"

# Print banner function
print_banner() {
    echo -e "${CLR_CYAN}"
    echo "======================================================================"
    echo "    🚀  ML RESEARCH PROJECT: AUTOMATIC RUNNER (v1 & v2)  🚀          "
    echo "======================================================================"
    echo -e "${CLR_RESET}"
    echo -e "${CLR_BOLD}System Target:${CLR_RESET} NVIDIA RTX 5050 GPU  |  16 GB RAM  |  Windows"
    echo -e "${CLR_BOLD}LLMs Executed:${CLR_RESET} distilgpt2 (82M), Qwen2.5-0.5B (LoRA), Qwen2.5-1.5B (LoRA), tiny-gpt2"
    echo -e "${CLR_BOLD}What this does:${CLR_RESET} Auto-installs 'uv', configures CUDA 12.8 for RTX 5050,"
    echo "                runs smoke tests, trains v1 & v2 (distilgpt2 + Qwen),"
    echo "                creates plots & tables, and pushes everything to Git!"
    echo "======================================================================"
    echo ""
}

# User-friendly error handler
report_failure() {
    local step_name="$1"
    local error_hint="$2"
    local exit_code="$3"
    
    echo ""
    echo -e "${CLR_RED}======================================================================${CLR_RESET}"
    echo -e "${CLR_RED} ❌  OOPS! Something went wrong at:${CLR_RESET} ${CLR_BOLD}${step_name}${CLR_RESET}"
    echo -e "${CLR_RED}======================================================================${CLR_RESET}"
    echo -e "${CLR_YELLOW}💡 Friendly Hint:${CLR_RESET} ${error_hint}"
    echo -e "${CLR_YELLOW}Exit code:${CLR_RESET} ${exit_code}"
    echo ""
    echo -e "${CLR_BOLD}👉 What you should do right now:${CLR_RESET}"
    echo "   1. Don't worry! Your computer is totally safe."
    echo "   2. Scroll up a little bit to see the red/error message."
    echo "   3. Take a screenshot or copy the text on this screen."
    echo "   4. Send it to Aayush so he can help you!"
    echo "======================================================================"
    echo ""
    read -p "Press [Enter] key to exit..."
    exit 1
}

TOTAL_STEPS=16
CURRENT_STEP=0

run_step() {
    local step_title="$1"
    local error_hint="$2"
    shift 2
    local cmd=("$@")

    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
    echo -e "${CLR_BOLD}[Step ${CURRENT_STEP}/${TOTAL_STEPS}] ⏳ ${step_title}...${CLR_RESET}"
    echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
    
    "${cmd[@]}"
    local status=$?
    
    if [ $status -ne 0 ]; then
        report_failure "${step_title}" "${error_hint}" "${status}"
    fi

    echo -e "${CLR_GREEN}✅ [Step ${CURRENT_STEP}/${TOTAL_STEPS}] Done: ${step_title}!${CLR_RESET}"
}

print_banner

# Move to the script's directory (repo root)
cd "$(dirname "$0")" || exit 1
REPO_DIR="$(pwd)"

# ==============================================================================
# STEP 1: Check Python and Windows Environment
# ==============================================================================
CURRENT_STEP=0
run_step "Checking Python Installation" \
    "Python is not found or not in PATH. Please install Python 3.12 from python.org and ensure 'Add python.exe to PATH' is checked!" \
    python --version

# ==============================================================================
# STEP 2: Check NVIDIA GPU (RTX 5050) & Drivers
# ==============================================================================
run_step "Checking NVIDIA RTX 5050 GPU & Drivers" \
    "nvidia-smi could not detect your NVIDIA GPU! Please make sure your NVIDIA graphics drivers are installed and up to date." \
    nvidia-smi

echo -e "${CLR_GREEN}✨ Awesome! Your NVIDIA GPU was detected successfully!${CLR_RESET}"

# Configure PyTorch CUDA memory management to prevent out-of-memory spikes on 16GB RAM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/common:${REPO_DIR}/v1:${REPO_DIR}/v2:${PYTHONPATH}"

# ==============================================================================
# STEP 3: Auto-Install uv (Ultra-Fast Package & Environment Manager)
# ==============================================================================
echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}⏳ Checking and Auto-Installing 'uv'...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

if ! command -v uv &> /dev/null; then
    echo "Installing 'uv' package manager via pip..."
    python -m pip install --upgrade pip --quiet
    python -m pip install uv --quiet || echo "uv pip install warning, will fallback to python pip if needed."
fi

if command -v uv &> /dev/null; then
    echo -e "${CLR_GREEN}⚡ 'uv' is ready: $(uv --version)${CLR_RESET}"
    USE_UV=1
else
    echo -e "${CLR_YELLOW}ℹ️ 'uv' not on PATH; using standard python pip.${CLR_RESET}"
    USE_UV=0
fi

# ==============================================================================
# STEP 4: Setup and Activate Virtual Environment (.venv)
# ==============================================================================
echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}⏳ Setting up Virtual Environment (.venv)...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment (.venv)..."
    if [ $USE_UV -eq 1 ]; then
        uv venv .venv || python -m venv .venv || report_failure "Creating Virtual Environment" "Failed to create .venv." "$?"
    else
        python -m venv .venv || report_failure "Creating Virtual Environment" "Failed to create .venv." "$?"
    fi
fi

# Activate virtual environment
if [ -f ".venv/Scripts/activate" ]; then
    source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    report_failure "Activating Virtual Environment" "Could not find .venv activate script!" 1
fi
echo -e "${CLR_GREEN}✅ Virtual environment (.venv) activated!${CLR_RESET}"

# ==============================================================================
# STEP 5: Install PyTorch with CUDA 12.8 (Optimized for RTX 5050 / Blackwell)
# ==============================================================================
echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}⏳ Checking PyTorch with CUDA 12.8 support for RTX 5050...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

HAS_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$HAS_CUDA" != "True" ]; then
    echo "Installing PyTorch with CUDA 12.8 wheel (specifically optimized for RTX 5050)..."
    if [ $USE_UV -eq 1 ]; then
        uv pip install torch --index-url https://download.pytorch.org/whl/cu128 || \
        pip install torch --index-url https://download.pytorch.org/whl/cu128 || \
        report_failure "Installing PyTorch CUDA 12.8" "PyTorch CUDA download failed. Check your internet connection." "$?"
    else
        pip install torch --index-url https://download.pytorch.org/whl/cu128 || \
        report_failure "Installing PyTorch CUDA 12.8" "PyTorch CUDA download failed. Check your internet connection." "$?"
    fi
fi

# Verify CUDA is active
run_step "Verifying PyTorch CUDA Acceleration on RTX 5050" \
    "PyTorch is installed, but cannot see your GPU (CUDA is False). Make sure CUDA 12.8 drivers are installed." \
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('PyTorch Version:', torch.__version__, '| GPU Detected:', torch.cuda.get_device_name(0))"

# ==============================================================================
# STEP 6: Install All Project Dependencies & Modules
# ==============================================================================
echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}⏳ Installing all required project modules...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

if [ $USE_UV -eq 1 ]; then
    uv pip install -r requirements.txt || pip install -r requirements.txt
    uv pip install -e . || pip install -e .
else
    pip install -r requirements.txt
    pip install -e .
fi

echo -e "${CLR_GREEN}✅ All project packages and modules installed successfully!${CLR_RESET}"

# ==============================================================================
# STEP 7: Run Environment Audit & LLM Sanity Checks (distilgpt2 + Qwen)
# ==============================================================================
run_step "Auditing System Environment & Packages" \
    "tabgen audit env reported missing packages or environment errors." \
    python -m tabgen audit env

run_step "Checking LLM Compatibility Table" \
    "Could not query LLM registry." \
    python -m tabgen llm list

run_step "Running Fast Offline Unit Tests" \
    "Pytest unit tests failed." \
    python -m pytest -m "not smoke"

run_step "Sanity Check with Tiny LLM (1 train step + 1 generation)" \
    "LLM check failed on tiny-gpt2." \
    python -m tabgen llm check --llm tiny

run_step "Sanity Check with distilgpt2 (Paper LLM on GPU)" \
    "LLM check failed on distilgpt2. Check your GPU memory." \
    python -m tabgen llm check --llm distilgpt2

run_step "Sanity Check with Qwen2.5-0.5B (LoRA Mode on GPU)" \
    "LLM check failed on Qwen2.5-0.5B. Check internet connection for download." \
    python -m tabgen llm check --llm 0.5b

run_step "Sanity Check with Qwen2.5-1.5B (LoRA Mode on GPU)" \
    "LLM check failed on Qwen2.5-1.5B. Check internet connection for download." \
    python -m tabgen llm check --llm 1.5b

# ==============================================================================
# STEP 8: Download Paper Datasets & Pre-training Visualizations
# ==============================================================================
run_step "Downloading Paper Datasets" \
    "Failed to download paper datasets. Please check internet connection." \
    python -m tabgen data download --dataset paper

run_step "Listing Available Datasets" \
    "Could not list datasets." \
    python -m tabgen data list

run_step "Visualizing Iris Dataset (Raw, Clean, Encoded)" \
    "Failed to generate Iris visualizations." \
    python -m tabgen viz data --dataset iris

run_step "Visualizing Adult Dataset (Raw, Clean, Encoded)" \
    "Failed to generate Adult visualizations." \
    python -m tabgen viz data --dataset adult

# ==============================================================================
# STEP 9: Run Smoke Tests (Verification for both v1 and v2)
# ==============================================================================
echo ""
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"
echo -e "${CLR_BOLD}🧪 RUNNING SMOKE TESTS (End-to-End Verification of v1 & v2)...${CLR_RESET}"
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"

run_step "Smoke Test v1 (Pred-LLM v1 pipeline check on Iris)" \
    "v1 smoke test failed. If it says 'LLM produced no valid rows', the model needed more steps." \
    python -m predllm_v1 smoke

run_step "Smoke Test v2 (Pred-LLM v2 schema-constrained check on Iris)" \
    "v2 smoke test failed." \
    python -m predllm_v2 smoke

# ==============================================================================
# STEP 10: Full Training & Generation for v1 (distilgpt2 + Qwen2.5-0.5B)
# ==============================================================================
echo ""
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"
echo -e "${CLR_BOLD}🧠 TRAINING & GENERATING v1 (distilgpt2 + Qwen2.5-0.5B)...${CLR_RESET}"
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"

run_step "Training v1 on Iris with distilgpt2 (Pred-LLM, GReaT, TapTap)" \
    "v1 distilgpt2 training run failed on Iris." \
    python -m predllm_v1 run --dataset iris --methods pred_llm,great,taptap --plots

run_step "Training v1 on Iris with Qwen2.5-0.5B LoRA" \
    "v1 Qwen2.5-0.5B training run failed on Iris." \
    python -m predllm_v1 run --dataset iris --method pred_llm --llm 0.5b --plots

# ==============================================================================
# STEP 11: Full Training & Generation for v2 (distilgpt2 + Qwen2.5-0.5B + Qwen2.5-1.5B)
# ==============================================================================
echo ""
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"
echo -e "${CLR_BOLD}🧠 TRAINING & GENERATING v2 (distilgpt2 + Qwen2.5-0.5B + Qwen2.5-1.5B)...${CLR_RESET}"
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"

run_step "Training v2 on Iris with distilgpt2 (Pred-LLM v2 + Ablations)" \
    "v2 distilgpt2 training run failed on Iris." \
    python -m predllm_v2 run --dataset iris --methods pred_llm_v2,v2_encoded,v2_plain_loss --plots

run_step "Training v2 on Iris with Qwen2.5-0.5B LoRA" \
    "v2 Qwen2.5-0.5B training run failed on Iris." \
    python -m predllm_v2 run --dataset iris --method pred_llm_v2 --llm 0.5b --plots

run_step "Training v2 on Iris with Qwen2.5-1.5B LoRA" \
    "v2 Qwen2.5-1.5B training run failed on Iris." \
    python -m predllm_v2 run --dataset iris --method pred_llm_v2 --llm 1.5b --plots

# ==============================================================================
# STEP 12: Inference, Generation & Attention Testing
# ==============================================================================
echo ""
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"
echo -e "${CLR_BOLD}📊 RUNNING INFERENCE, GENERATION & ATTENTION VISUALIZATION...${CLR_RESET}"
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"

run_step "v1 Inference (distilgpt2 as classifier on test set)" \
    "v1 inference test failed." \
    python -m predllm_v1 infer --run v1/runs/iris/pred_llm__distilgpt2/seed0

run_step "v1 Generation (Sample 150 synthetic rows with distilgpt2)" \
    "v1 generation failed." \
    python -m predllm_v1 generate --run v1/runs/iris/pred_llm__distilgpt2/seed0 --n 150

run_step "v2 Inference (distilgpt2 permutation-marginalized posterior)" \
    "v2 inference test failed." \
    python -m predllm_v2 infer --run v2/runs/iris/pred_llm_v2__distilgpt2/seed0

run_step "v2 Generation (Sample 150 synthetic rows with distilgpt2)" \
    "v2 generation failed." \
    python -m predllm_v2 generate --run v2/runs/iris/pred_llm_v2__distilgpt2/seed0 --n 150

run_step "v2 Generation (Sample 150 synthetic rows with Qwen2.5-0.5B)" \
    "v2 Qwen generation failed." \
    python -m predllm_v2 generate --run v2/runs/iris/pred_llm_v2__qwen2.5-0.5b/seed0 --n 150

run_step "Attention Heatmap Visualization (Figure 4)" \
    "Attention visualization failed." \
    python -m predllm_v1 attention --run v1/runs/iris/pred_llm__distilgpt2/seed0

# ==============================================================================
# STEP 13: Visualizing Results, Running Audits & Building Tables
# ==============================================================================
run_step "Visualizing Synthetic vs Real Distributions (v1)" \
    "Failed to plot v1 run results." \
    python -m tabgen viz run --run v1/runs

run_step "Visualizing Synthetic vs Real Distributions (v2)" \
    "Failed to plot v2 run results." \
    python -m tabgen viz run --run v2/runs

run_step "Auditing v1 Run Artifacts & Metrics" \
    "v1 audit run encountered errors." \
    python -m tabgen audit run --run v1/runs

run_step "Auditing v2 Run Artifacts & Metrics" \
    "v2 audit run encountered errors." \
    python -m tabgen audit run --run v2/runs

run_step "Generating Final Comparison Tables (Table II / III)" \
    "Failed to build summary tables." \
    python -m tabgen report --versions v1,v2

# ==============================================================================
# STEP 14: Commit Results to Git & Push to Remote
# ==============================================================================
echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}💾 Staging and Committing Runs to Git...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

# Ensure user.name and user.email are set so git commit doesn't fail
if [ -z "$(git config user.name)" ]; then
    git config user.name "Research Runner (Friend PC)"
fi
if [ -z "$(git config user.email)" ]; then
    git config user.email "runner@research.local"
fi

# Add run results, reports, plots, and updated .gitignore
# Model binaries (*.safetensors, *_models/) are automatically excluded by .gitignore to respect GitHub 100MB limit
git add .gitignore
if [ -d "v1/runs" ]; then git add v1/runs; fi
if [ -d "v2/runs" ]; then git add v2/runs; fi
if [ -d "reports" ]; then git add reports; fi

# Check if there is anything to commit
if git diff --staged --quiet; then
    echo "No new changes to commit in runs/ or reports/."
else
    git commit -m "Auto-run: experiment runs, metrics, plots, and tables for v1 and v2 (distilgpt2 + Qwen)"
    echo -e "${CLR_GREEN}✅ Successfully committed run results to Git!${CLR_RESET}"
fi

echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}🚀 Pushing Committed Results to GitHub Remote...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo "Pushing branch: ${CURRENT_BRANCH} to origin..."

if git push origin "${CURRENT_BRANCH}"; then
    echo -e "${CLR_GREEN}✅ Successfully pushed all results to GitHub!${CLR_RESET}"
else
    echo ""
    echo -e "${CLR_YELLOW}⚠️  Git push needed authentication or failed to reach GitHub directly.${CLR_RESET}"
    echo -e "${CLR_BOLD}Don't worry! All your experiment results, plots, and tables are safely saved and committed locally.${CLR_RESET}"
    echo "Aayush can push them anytime by running:"
    echo "   git push origin ${CURRENT_BRANCH}"
fi

# ==============================================================================
# STEP 15: Celebration & Summary
# ==============================================================================
echo ""
echo -e "${CLR_GREEN}======================================================================${CLR_RESET}"
echo -e "${CLR_GREEN}  🎉🎉🎉  CONGRATULATIONS! ALL TASKS COMPLETED SUCCESSFULLY!  🎉🎉🎉  ${CLR_RESET}"
echo -e "${CLR_GREEN}======================================================================${CLR_RESET}"
echo ""
echo -e "${CLR_BOLD}Here is what was completed on your NVIDIA RTX 5050 GPU:${CLR_RESET}"
echo "  1. ⚡ Auto-installed 'uv' & configured Python virtual environment"
echo "  2. 🎮 PyTorch with CUDA 12.8 installed and verified on RTX 5050"
echo "  3. 📦 All project modules (transformers, peft, xgboost, etc.) installed"
echo "  4. 🧪 Smoke tests passed for v1 and v2"
echo "  5. 🧠 v1 Models trained (distilgpt2 + Qwen2.5-0.5B)"
echo "  6. 🧠 v2 Models trained (distilgpt2 + Qwen2.5-0.5B + Qwen2.5-1.5B)"
echo "  7. 🔍 Inference testing & generation executed across models"
echo "  8. 📊 Visualizations created (Marginals, t-SNE, DCR, Attention heatmap)"
echo "  9. 📋 Accuracy & MSE comparison tables compiled in: reports/tables/"
echo " 10. 💾 All run results safely committed to Git!"
echo ""
echo -e "${CLR_CYAN}Your generated files are stored in:${CLR_RESET}"
echo "  - v1/runs/         (v1 synthetic datasets, metrics, and plots)"
echo "  - v2/runs/         (v2 synthetic datasets, metrics, and plots)"
echo "  - reports/data_viz (raw vs clean vs encoded dataset distributions)"
echo "  - reports/tables   (final comparison tables in Markdown)"
echo ""
echo -e "${CLR_BOLD}You did an amazing job! Thank you for letting us use your GPU! 🚀${CLR_RESET}"
echo "======================================================================"
echo ""
read -p "Press [Enter] key to finish and close this window..."
exit 0
