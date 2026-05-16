@echo off
REM ============================================================
REM  launch_ui.bat - start the PushGP evolution Web UI
REM
REM  Opens http://127.0.0.1:8765 in your default browser.
REM  Press Ctrl+C in this window to stop the UI; the Job Object
REM  guarantees all evolution children die with it.
REM ============================================================
setlocal
set "PY=C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe"
set "HERE=%~dp0"
pushd "%HERE%\.."
"%PY%" -B ui\launch_ui.py %*
popd
endlocal
