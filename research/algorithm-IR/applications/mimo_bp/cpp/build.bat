@echo off
REM Build the BP-IR evaluator DLL using MSVC
REM Must be run from a Visual Studio Developer Command Prompt
REM or with vcvarsall.bat already called.

setlocal
set "SRC=%~dp0bp_ir_decoder.cpp"
set "OUT=%~dp0bp_ir_eval.dll"

echo Building bp_ir_eval.dll...
cl.exe /EHsc /O2 /openmp /std:c++17 /DBUILD_DLL "%SRC%" /LD /Fe:"%OUT%" /link /out:"%OUT%"

if %ERRORLEVEL% EQU 0 (
    echo Build succeeded: %OUT%
) else (
    echo Build FAILED with error %ERRORLEVEL%
)
endlocal
