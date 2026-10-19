#!/usr/bin/env bash
# ==============================================================================
# ML RESEARCH PROJECT - AUTOMATED UNIFIED RUNNER (v1 + v2 + v3)
# Optimized for: NVIDIA RTX 5050 GPU | 16 GB RAM | Windows (Git Bash)
# Benchmarks: Pred-LLM v1, v2 (schema-constrained), and v3 (negative unlikelihood + outlier bounding)
# ==============================================================================

# Enable UTF-8 encoding
export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1

# ANSI Color codes for clean and readable terminal output
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
    echo "  🚀  ML RESEARCH PROJECT: UNIFIED RUNNER (v1 + v2 + v3)  🚀         "
    echo "======================================================================"
    echo -e "${CLR_RESET}"
    echo -e "${CLR_BOLD}System Target:${CLR_RESET} NVIDIA RTX 5050 GPU  |  16 GB RAM  |  Windows (Git Bash)"
    echo -e "${CLR_BOLD}Versions Run :${CLR_RESET} v1 (Paper baseline), v2 (Constrained FSM), v3 (Negative loss + Outlier bounds)"
    echo -e "${CLR_BOLD}Datasets     :${CLR_RESET} ${BENCH_DATASETS} (Max rows per dataset: ${BENCH_MAX_ROWS})"
    echo -e "${CLR_BOLD}What this does:${CLR_RESET} Prepares environment, verifies CUDA 12.8,"
    echo "                runs smoke tests across v1, v2 & v3, executes all benchmarks,"
    echo "                builds comparison tables, plots graphs, and saves to Git!"
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
    echo "   1. Don't worry! Your computer and progress are safe."
    echo "   2. Scroll up a little bit to review the error message above."
    echo "   3. Take a screenshot or copy the text on this screen."
    echo "======================================================================"
    echo ""
    read -p "Press [Enter] key to exit..."
    exit 1
}

TOTAL_STEPS=12
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

# Move to the script's directory (repo root)
cd "$(dirname "$0")" || exit 1
REPO_DIR="$(pwd)"

# Default configurations (can be overridden by command-line arguments)
BENCH_DATASETS="${BENCH_DATASETS:-paper}"
BENCH_MAX_ROWS="${BENCH_MAX_ROWS:-1000}"
BENCH_EPOCHS="${BENCH_EPOCHS:-35}"
BENCH_LLM="${BENCH_LLM:-distilgpt2}"
BENCH_VERSIONS="${BENCH_VERSIONS:-v1,v2,v3}"

print_banner

# ==============================================================================
# STEP 1: Check Python and Virtual Environment
# ==============================================================================
run_step "Checking Python Installation" \
    "Python is not found or not in PATH. Please install Python 3.12 and ensure it is in PATH." \
    python --version

if [ -f ".venv/Scripts/activate" ]; then
    source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo -e "${CLR_YELLOW}⚠️ Virtual environment (.venv) not found. Using system python.${CLR_RESET}"
fi

# Set PYTHONPATH to discover all version packages and common modules
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/common:${REPO_DIR}/v1:${REPO_DIR}/v2:${REPO_DIR}/v3:${PYTHONPATH}"

# ==============================================================================
# STEP 2: Verify NVIDIA GPU & CUDA Acceleration
# ==============================================================================
run_step "Checking NVIDIA GPU & PyTorch CUDA Acceleration" \
    "PyTorch cannot detect CUDA. Check NVIDIA drivers or CUDA 12.8 wheel installation." \
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"

# ==============================================================================
# STEP 3: Download & Verify Datasets
# ==============================================================================
run_step "Ensuring Paper Benchmark Datasets are Downloaded" \
    "Failed to fetch paper datasets. Please verify your internet connection." \
    python -m tabgen data download --dataset paper

# ==============================================================================
# STEP 4: Smoke Test Pred-LLM v1 Pipeline
# ==============================================================================
run_step "Smoke Test v1 Pipeline (Paper baseline verification on Iris)" \
    "v1 smoke test failed." \
    python -m predllm_v1 smoke

# ==============================================================================
# STEP 5: Smoke Test Pred-LLM v2 Pipeline
# ==============================================================================
run_step "Smoke Test v2 Pipeline (Schema-constrained FSM verification on Iris)" \
    "v2 smoke test failed." \
    python -m predllm_v2 smoke

# ==============================================================================
# STEP 6: Smoke Test Pred-LLM v3 Pipeline
# ==============================================================================
run_step "Smoke Test v3 Pipeline (Negative loss + Outlier envelope verification on Iris)" \
    "v3 smoke test failed." \
    python -m predllm_v3 smoke

# ==============================================================================
# STEP 7: Run Cross-Version Benchmark (v1, v2, v3) Across All Datasets
# ==============================================================================
echo ""
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"
echo -e "${CLR_BOLD}🧠 EXECUTING ALL BENCHMARKS: v1, v2, v3 (Datasets: ${BENCH_DATASETS}, Capped: ${BENCH_MAX_ROWS} rows)...${CLR_RESET}"
echo -e "${CLR_MAGENTA}======================================================================${CLR_RESET}"

run_step "Running Unified Cross-Version Benchmark Suite" \
    "Benchmark execution failed. Check GPU memory or dataset configurations." \
    python run_all_versions.py \
        --datasets "${BENCH_DATASETS}" \
        --max-rows "${BENCH_MAX_ROWS}" \
        --versions "${BENCH_VERSIONS}" \
        --llm "${BENCH_LLM}" \
        --epochs "${BENCH_EPOCHS}" \
        "$@"

# ==============================================================================
# STEP 8: Generate Diagnostic Visualizations for Runs
# ==============================================================================
run_step "Visualizing Distributions for v1 Runs" \
    "Failed to generate v1 distribution visualizations." \
    python -m tabgen viz run --run v1/runs

run_step "Visualizing Distributions for v2 Runs" \
    "Failed to generate v2 distribution visualizations." \
    python -m tabgen viz run --run v2/runs

run_step "Visualizing Distributions for v3 Runs" \
    "Failed to generate v3 distribution visualizations." \
    python -m tabgen viz run --run v3/runs

# ==============================================================================
# STEP 9: Compile Cross-Version Comparison Tables
# ==============================================================================
run_step "Compiling Summary Comparison Tables (Table II / III for v1, v2, v3)" \
    "Failed to generate summary tables." \
    python -m tabgen report --versions v1,v2,v3

# ==============================================================================
# STEP 10: Generate Cross-Version Comparative Graphs & Loss Dynamics
# ==============================================================================
run_step "Rendering Comparative Visualization Charts & v3 Loss Curves" \
    "Failed to render comparative graphs." \
    python -m reports.make_comparison_graphs

# ==============================================================================
# STEP 11: Audit Artifacts & Verify Run Integrity
# ==============================================================================
run_step "Auditing All Run Artifacts" \
    "Run audit encountered missing files or corrupted metrics." \
    python -m tabgen audit all-runs

# ==============================================================================
# STEP 12: Stage & Commit Results to Git
# ==============================================================================
echo ""
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"
echo -e "${CLR_BOLD}💾 Staging and Committing Results to Git...${CLR_RESET}"
echo -e "${CLR_CYAN}----------------------------------------------------------------------${CLR_RESET}"

if [ -z "$(git config user.name)" ]; then
    git config user.name "Research Runner"
fi
if [ -z "$(git config user.email)" ]; then
    git config user.email "runner@research.local"
fi

git add .gitignore
if [ -d "v1/runs" ]; then git add v1/runs; fi
if [ -d "v2/runs" ]; then git add v2/runs; fi
if [ -d "v3/runs" ]; then git add v3/runs; fi
if [ -d "reports" ]; then git add reports; fi

if git diff --staged --quiet; then
    echo "No new changes to commit in runs/ or reports/."
else
    git commit -m "Auto-run: unified benchmark across v1, v2, and v3 on ${BENCH_DATASETS}"
    echo -e "${CLR_GREEN}✅ Successfully committed run results to Git!${CLR_RESET}"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo "Pushing branch: ${CURRENT_BRANCH} to origin..."

if git push origin "${CURRENT_BRANCH}" 2>/dev/null; then
    echo -e "${CLR_GREEN}✅ Successfully pushed all results to GitHub!${CLR_RESET}"
else
    echo -e "${CLR_YELLOW}ℹ️ Git push skipped or requires authentication. Local commits are safely stored.${CLR_RESET}"
fi

# ==============================================================================
# SUMMARY & COMPLETION
# ==============================================================================
echo ""
echo -e "${CLR_GREEN}======================================================================${CLR_RESET}"
echo -e "${CLR_GREEN}  🎉🎉🎉  ALL BENCHMARKS ACROSS v1, v2, & v3 COMPLETED!  🎉🎉🎉  ${CLR_RESET}"
echo -e "${CLR_GREEN}======================================================================${CLR_RESET}"
echo ""
echo -e "${CLR_BOLD}Results Summary:${CLR_RESET}"
echo "  📋 Comparison Tables       : reports/tables/ (table_accuracy.md, table_f1_macro.md, fidelity.md)"
echo "  📊 Comparative Charts      : reports/data_viz/comparison/ (TSTR bars, fidelity, privacy, loss curves)"
echo "  💾 Version 1 Runs & Data   : v1/runs/"
echo "  💾 Version 2 Runs & Data   : v2/runs/"
echo "  💾 Version 3 Runs & Data   : v3/runs/"
echo ""
echo -e "${CLR_CYAN}Done! You can review your reports anytime in the 'reports/' directory.${CLR_RESET}"
echo "======================================================================"
echo ""
read -p "Press [Enter] key to exit..."
exit 0
