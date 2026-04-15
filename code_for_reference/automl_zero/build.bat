@echo off
REM Build AutoML-Zero using VS 2022 MSVC toolchain with Bazel 6.5.0
REM This script initializes the MSVC environment first, then runs Bazel

REM Set required Bazel environment variables
set BAZEL_SH=C:\Program Files\Git\usr\bin\bash.exe
set BAZEL_VC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC
set BAZEL_VC_FULL_VERSION=14.42.34433

REM Verify cl.exe is accessible
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo Initializing MSVC environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

echo.
echo === Starting Bazel Build ===
echo.

REM Build with Bazel 6.5.0
bazel-6.5.0.exe build -c opt ^
    --copt=-DMAX_SCALAR_ADDRESSES=4 ^
    --copt=-DMAX_VECTOR_ADDRESSES=3 ^
    --copt=-DMAX_MATRIX_ADDRESSES=1 ^
    --verbose_failures ^
    :run_search_experiment

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo BUILD SUCCEEDED
echo Binary: bazel-bin\run_search_experiment.exe
