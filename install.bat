@echo off
setlocal EnableExtensions
title comfy-kitchen one-click install (ROCm/HIP)

rem ================================================================
rem  comfy-kitchen one-click build & install for Windows + ROCm/HIP.
rem
rem  Usage: double-click this file, or run  install.bat  in a terminal.
rem
rem  Optional environment variables:
rem    PYTHON              python executable (default: python)
rem    COMFY_HIP_ARCHS     target GPU arch, e.g. gfx1103
rem                        (default: auto-detect the current GPU)
rem
rem  IMPORTANT: do NOT drop "--no-build-isolation" below. pip's default
rem  isolated build environment cannot see the pip-installed rocm-sdk
rem  (_rocm_sdk_devel) or torch, so setup.py silently skips the HIP
rem  backend and you get a pure-Python package whose int8 GEMM falls
rem  back to triton/eager and runs much slower.
rem ================================================================

set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"
if not exist "%REPO_DIR%setup.py" (
    echo [ERROR] setup.py not found. Put this script in the comfy-kitchen repo root.
    exit /b 1
)
if "%PYTHON%"=="" set "PYTHON=python"

echo ================================================================
echo  [1/6] Checking Python
echo ================================================================
"%PYTHON%" --version
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.12+ and add it to PATH.
    exit /b 1
)

echo.
echo ================================================================
echo  [2/6] Checking PyTorch (ROCm/HIP build required)
echo ================================================================
"%PYTHON%" -c "import torch; assert torch.version.hip, 'no-hip'; print('PyTorch', torch.__version__, '| HIP', torch.version.hip)"
if errorlevel 1 (
    echo [ERROR] No ROCm build of PyTorch found (torch.version.hip is empty^).
    echo         Install it first, e.g.:
    echo           pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm10.0
    exit /b 1
)

echo.
echo ================================================================
echo  [3/6] Checking build tools cmake / ninja
echo ================================================================
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake not found. Install it and retry:  winget install cmake
    exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ninja not found. Install it and retry:  winget install ninja-build.ninja
    exit /b 1
)
cmake --version | findstr /b "cmake version"
ninja --version

echo.
echo ================================================================
echo  [4/6] Installing build dependency nanobind
echo ================================================================
"%PYTHON%" -m pip install nanobind
if errorlevel 1 (
    echo [ERROR] Failed to install nanobind.
    exit /b 1
)

echo.
echo ================================================================
echo  [5/6] Building and installing comfy-kitchen
echo ================================================================
if "%COMFY_HIP_ARCHS%"=="" (
    echo Target GPU arch: auto-detect current device
) else (
    echo Target GPU arch: %COMFY_HIP_ARCHS%
)
"%PYTHON%" -m pip install . --no-build-isolation --no-deps --force-reinstall
if errorlevel 1 (
    echo.
    echo [ERROR] Build/install failed.
    echo         Check that:
    echo           1. The ROCm SDK is visible:  python -m rocm_sdk path --root
    echo           2. cmake and ninja are on PATH
    echo           3. The log contains "Building HIP extension with CMake"
    echo              and "HIP architectures: gfx...."
    exit /b 1
)

echo.
echo ================================================================
echo  [6/6] Verifying install (HIP backend + int8 GEMM smoke test)
echo ================================================================
rem The check must run OUTSIDE the repo: the source tree has no compiled
rem artifacts, so importing comfy_kitchen from the repo would pick the
rem source and falsely report a broken install.
pushd "%TEMP%"
"%PYTHON%" "%REPO_DIR%install_verify.py"
set "VERIFY_RC=%ERRORLEVEL%"
popd
if not "%VERIFY_RC%"=="0" (
    echo [ERROR] Verification failed. See the output above.
    exit /b 1
)

echo.
echo ================================================================
echo  Done. comfy-kitchen was compiled for the current GPU and installed
echo  into the current Python environment.
echo.
echo  Run the GEMM benchmark OUTSIDE the repo directory, e.g.:
echo      cd /d C:\Build\benchmark_attn
echo      python benchmark_gemm.py
echo ================================================================
endlocal
exit /b 0