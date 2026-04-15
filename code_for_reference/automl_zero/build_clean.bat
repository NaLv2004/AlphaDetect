@echo off
REM Build AutoML-Zero with a clean environment (no OSS CAD Suite interference)
setlocal

REM Kill any existing Bazel server
D:\ChannelCoding\RCOM\AlphaDetect\code_for_reference\automl_zero\bazel-5.4.1.exe shutdown 2>nul

REM Set required Bazel environment variables
set BAZEL_SH=C:\Program Files\Git\usr\bin\bash.exe
set BAZEL_VC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC
set BAZEL_VC_FULL_VERSION=14.42.34433

REM Remove old output base
rmdir /s /q "C:\Users\31363\_bazel_31363" 2>nul

REM Change to source directory
cd /d D:\ChannelCoding\RCOM\AlphaDetect\code_for_reference\automl_zero

echo === Environment ===
echo BAZEL_VC=%BAZEL_VC%
echo BAZEL_SH=%BAZEL_SH%
echo BAZEL_VC_FULL_VERSION=%BAZEL_VC_FULL_VERSION%

echo.
echo === Testing MSVC ===
dir "%BAZEL_VC%\Tools\MSVC\%BAZEL_VC_FULL_VERSION%\bin\Hostx64\x64\cl.exe" 2>&1
dir "%BAZEL_VC%\Auxiliary\Build\vcvarsall.bat" 2>&1

echo.
echo === Starting Bazel Build ===
bazel-5.4.1.exe build -c opt ^
    --copt=-DMAX_SCALAR_ADDRESSES=4 ^
    --copt=-DMAX_VECTOR_ADDRESSES=3 ^
    --copt=-DMAX_MATRIX_ADDRESSES=1 ^
    --verbose_failures ^
    :run_search_experiment

if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo.
echo BUILD SUCCEEDED
