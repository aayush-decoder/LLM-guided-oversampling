@echo off
title Pred-LLM Unified Runner (v1, v2, v3)
color 0a

echo ======================================================================
echo    Pred-LLM Multi-Version Runner (v1, v2, v3)
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "run_all_versions.py" %*
) else (
    python "run_all_versions.py" %*
)

echo.
pause
