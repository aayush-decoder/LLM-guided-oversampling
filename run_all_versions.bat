@echo off
setlocal
chcp 65001 >nul 2>nul

title Pred-LLM Unified Runner (v1, v2, v3)
color 0a

REM Configure Python environment
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Get repository directory
set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"

REM Set PYTHONPATH
set "PYTHONPATH=%REPO_DIR%;%REPO_DIR%common;%REPO_DIR%v1;%REPO_DIR%v2;%REPO_DIR%v3;%PYTHONPATH%"

REM Detect Python executable
if exist "%REPO_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%REPO_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

REM Default parameters
set "DEFAULT_DATASETS=paper"
set "DEFAULT_MAX_ROWS=1000"
set "DEFAULT_EPOCHS=35"
set "DEFAULT_LLM=distilgpt2"
set "DEFAULT_VERSIONS=v1,v2,v3"

echo ======================================================================
echo    ML RESEARCH PROJECT: UNIFIED RUNNER (v1 + v2 + v3)
echo ======================================================================
echo System Target : NVIDIA RTX 5050 GPU / 16 GB RAM / Windows
echo Python Path   : %PYTHON_EXE%
echo Versions Run  : %DEFAULT_VERSIONS% (v1, v2, v3)
echo Datasets      : %DEFAULT_DATASETS% (Max rows per dataset: %DEFAULT_MAX_ROWS%)
echo Epochs        : %DEFAULT_EPOCHS%
echo ======================================================================
echo.

REM ----------------------------------------------------------------------
REM STEP 1: Verify Python Environment
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 1/12] Checking Python Environment...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" --version
if errorlevel 1 goto :FAILED_PYTHON
echo [Step 1/12] Done: Python verified!
echo.

REM ----------------------------------------------------------------------
REM STEP 2: Verify NVIDIA GPU and PyTorch CUDA Acceleration
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 2/12] Checking NVIDIA GPU and PyTorch CUDA Acceleration...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available!'; print('PyTorch Version:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :FAILED_CUDA
echo [Step 2/12] Done: CUDA is active!
echo.

REM ----------------------------------------------------------------------
REM STEP 3: Download Paper Datasets
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 3/12] Ensuring Paper Benchmark Datasets are Downloaded...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m tabgen data download --dataset paper
if errorlevel 1 goto :FAILED_DATA
echo [Step 3/12] Done: Datasets verified!
echo.

REM ----------------------------------------------------------------------
REM STEP 4: Smoke Test Pred-LLM v1
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 4/12] Smoke Testing Pred-LLM v1 Pipeline...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m predllm_v1 smoke
if errorlevel 1 goto :FAILED_SMOKE_V1
echo [Step 4/12] Done: v1 smoke test passed!
echo.

REM ----------------------------------------------------------------------
REM STEP 5: Smoke Test Pred-LLM v2
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 5/12] Smoke Testing Pred-LLM v2 Pipeline (Schema FSM)...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m predllm_v2 smoke
if errorlevel 1 goto :FAILED_SMOKE_V2
echo [Step 5/12] Done: v2 smoke test passed!
echo.

REM ----------------------------------------------------------------------
REM STEP 6: Smoke Test Pred-LLM v3
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 6/12] Smoke Testing Pred-LLM v3 Pipeline (Negative Loss + Outlier bounds)...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m predllm_v3 smoke
if errorlevel 1 goto :FAILED_SMOKE_V3
echo [Step 6/12] Done: v3 smoke test passed!
echo.

REM ----------------------------------------------------------------------
REM STEP 7: Run Unified Cross-Version Benchmark Suite
REM ----------------------------------------------------------------------
echo ======================================================================
echo [Step 7/12] EXECUTING ALL BENCHMARKS: v1, v2, v3
echo Datasets: %DEFAULT_DATASETS% ^| Max Rows: %DEFAULT_MAX_ROWS% ^| Epochs: %DEFAULT_EPOCHS%
echo ======================================================================
"%PYTHON_EXE%" run_all_versions.py --datasets %DEFAULT_DATASETS% --max-rows %DEFAULT_MAX_ROWS% --versions %DEFAULT_VERSIONS% --llm %DEFAULT_LLM% --epochs %DEFAULT_EPOCHS% %*
if errorlevel 1 goto :FAILED_BENCH
echo [Step 7/12] Done: All benchmarks completed!
echo.

REM ----------------------------------------------------------------------
REM STEP 8: Generate Distribution Visualizations
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 8/12] Generating Diagnostic Plots for Runs...
echo ----------------------------------------------------------------------
if exist "v1\runs" "%PYTHON_EXE%" -m tabgen viz run --run v1/runs
if exist "v2\runs" "%PYTHON_EXE%" -m tabgen viz run --run v2/runs
if exist "v3\runs" "%PYTHON_EXE%" -m tabgen viz run --run v3/runs
echo [Step 8/12] Done: Diagnostic plots generated!
echo.

REM ----------------------------------------------------------------------
REM STEP 9: Compile Cross-Version Comparison Tables
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 9/12] Compiling Summary Comparison Tables (Table II / III)...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m tabgen report --versions v1,v2,v3
if errorlevel 1 goto :FAILED_REPORT
echo [Step 9/12] Done: Summary tables compiled in reports\tables!
echo.

REM ----------------------------------------------------------------------
REM STEP 10: Render Comparison Charts and Loss Curves
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 10/12] Rendering Comparison Charts and v3 Loss Curves...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m reports.make_comparison_graphs
if errorlevel 1 goto :FAILED_PLOTS
echo [Step 10/12] Done: Charts rendered in reports\data_viz\comparison!
echo.

REM ----------------------------------------------------------------------
REM STEP 11: Audit Artifacts
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 11/12] Auditing Run Artifacts...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" -m tabgen audit all-runs
echo [Step 11/12] Done: Audit complete!
echo.

REM ----------------------------------------------------------------------
REM STEP 12: Stage and Commit Results to Git
REM ----------------------------------------------------------------------
echo ----------------------------------------------------------------------
echo [Step 12/12] Staging and Committing Results to Git...
echo ----------------------------------------------------------------------
where git >nul 2>nul
if %ERRORLEVEL% equ 0 (
    git add .gitignore
    if exist "v1\runs" git add v1/runs
    if exist "v2\runs" git add v2/runs
    if exist "v3\runs" git add v3/runs
    if exist "reports" git add reports

    git diff --staged --quiet
    if errorlevel 1 (
        git commit -m "Auto-run: unified benchmark across v1, v2, and v3 on %DEFAULT_DATASETS%"
        echo [Git] Committed new run results locally!
        git push origin HEAD 2>nul
    ) else (
        echo [Git] No new files to commit.
    )
) else (
    echo [Git] Git CLI not found in PATH; skipping git commit.
)
echo [Step 12/12] Done: Results saved!
echo.

REM ----------------------------------------------------------------------
REM SUCCESS BANNER
REM ----------------------------------------------------------------------
echo ======================================================================
echo    ALL BENCHMARKS ACROSS v1, v2, and v3 COMPLETED!
echo ======================================================================
echo.
echo Results Summary:
echo   - Summary Tables  : reports\tables\ (table_accuracy.md, table_f1_macro.md, fidelity.md)
echo   - Visual Charts   : reports\data_viz\comparison\ (TSTR bars, fidelity, privacy, loss curves)
echo   - v1 Runs         : v1\runs\
echo   - v2 Runs         : v2\runs\
echo   - v3 Runs         : v3\runs\
echo.
echo All generated reports and charts are saved in the 'reports\' folder!
echo ======================================================================
echo.
pause
exit /b 0

:FAILED_PYTHON
echo.
echo [X] ERROR: Python executable was not found.
echo Hint: Make sure .venv exists or Python 3.12 is in PATH.
pause
exit /b 1

:FAILED_CUDA
echo.
echo [X] ERROR: PyTorch CUDA GPU acceleration failed.
echo Hint: Make sure NVIDIA drivers are installed and CUDA 12.8 wheel is configured in .venv.
pause
exit /b 1

:FAILED_DATA
echo.
echo [X] ERROR: Failed to download paper benchmark datasets.
pause
exit /b 1

:FAILED_SMOKE_V1
echo.
echo [X] ERROR: Pred-LLM v1 smoke test failed.
pause
exit /b 1

:FAILED_SMOKE_V2
echo.
echo [X] ERROR: Pred-LLM v2 smoke test failed.
pause
exit /b 1

:FAILED_SMOKE_V3
echo.
echo [X] ERROR: Pred-LLM v3 smoke test failed.
pause
exit /b 1

:FAILED_BENCH
echo.
echo [X] ERROR: Benchmark suite encountered an error during execution.
pause
exit /b 1

:FAILED_REPORT
echo.
echo [X] ERROR: Failed compiling cross-version summary tables.
pause
exit /b 1

:FAILED_PLOTS
echo.
echo [X] ERROR: Failed rendering comparative charts.
pause
exit /b 1
