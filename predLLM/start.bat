@echo off
title ML Research Project - One-Click Runner
color 0b
echo ======================================================================
echo    Starting ML Research Project Automated Runner on your GPU...
echo ======================================================================
echo.

set "BASH_PATH="
if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH_PATH=%ProgramFiles%\Git\bin\bash.exe"
if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "BASH_PATH=%ProgramFiles(x86)%\Git\bin\bash.exe"
if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" set "BASH_PATH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"

if defined BASH_PATH (
    echo Launching with Git Bash: %BASH_PATH%
    echo.
    "%BASH_PATH%" --login -i "run_all.sh"
) else (
    where bash >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        bash "run_all.sh"
    ) else (
        color 0c
        echo ======================================================================
        echo  [!] Git Bash was not found on your computer!
        echo ======================================================================
        echo  Please download and install Git for Windows from:
        echo  https://git-scm.com/download/win
        echo.
        echo  Once installed, double-click this file again!
        echo ======================================================================
        pause
    )
)
