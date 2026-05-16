@echo off
REM ===================================================================
REM  evo_run.bat - One-shot large-scale Push-GP evolution + math parser
REM
REM  Usage:
REM      evo_run.bat
REM      evo_run.bat my_run_name
REM      evo_run.bat my_run 300 30
REM ===================================================================

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

set "REPO=D:\ChannelCoding\RCOM\AlphaDetect"
set "BPS=%REPO%\research\bp-deocder-synthesis"
set "PY=C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe"
set "RESULTS=%BPS%\results\logged_evolution"

if "%~1"=="" (
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
    set "RUN=dce_e2e_pop200_gens20_!TS!"
) else (
    set "RUN=%~1"
)
if "%~2"=="" ( set "POP=200" ) else ( set "POP=%~2" )
if "%~3"=="" ( set "GENS=20" ) else ( set "GENS=%~3" )

set "EVO_LOG=%RESULTS%\%RUN%.console.log"
set "EVO_ERR=%RESULTS%\%RUN%.console.err.log"
set "WATCH_LOG=%RESULTS%\%RUN%.watcher.console.log"
set "WATCH_ERR=%RESULTS%\%RUN%.watcher.console.err.log"
set "EVO_INNER=%RESULTS%\%RUN%.__evo_inner.bat"
set "WATCH_INNER=%RESULTS%\%RUN%.__watch_inner.bat"

if not exist "%RESULTS%" mkdir "%RESULTS%"

REM ---- Write inner evolution launcher (single command, redirections inside) ----
> "%EVO_INNER%" echo @echo off
>> "%EVO_INNER%" echo "%PY%" -B -u "%BPS%\experiments\run_logged_evolution.py" --run-name %RUN% --pop-size %POP% --gens %GENS% --seed 2025 --snr-list=-3,-2,-1 --n-frames 6 --max-iter 8 --rand-min-size 4 --rand-max-size 30 --workers 16 --cpp-seeder --dce-bp --dce-bp-max-iter 8 --dce-bp-decimals 6 --dce-bp-max-passes 800 --dce-bp-max-decode-evals -1 --dce-bp-threads 0 --dce-bp-snr-db=-2 --dce-bp-n-frames 1 1^> "%EVO_LOG%" 2^> "%EVO_ERR%"

REM ---- Write inner watcher launcher ----
> "%WATCH_INNER%" echo @echo off
>> "%WATCH_INNER%" echo "%PY%" -B -u "%BPS%\experiments\parse_individuals_math.py" %RUN% --watch --poll-interval 3.0 --max-chars 500 1^> "%WATCH_LOG%" 2^> "%WATCH_ERR%"

echo ===================================================================
echo  EVO RUN LAUNCH
echo ===================================================================
echo  Python      : %PY%
echo  Run name    : %RUN%
echo  Pop size    : %POP%
echo  Generations : %GENS%
echo.
echo  Logs:
echo    Evolution stdout : %EVO_LOG%
echo    Evolution stderr : %EVO_ERR%
echo    Watcher   stdout : %WATCH_LOG%
echo    Watcher   stderr : %WATCH_ERR%
echo.
echo  Run-dir artifacts (will appear shortly):
echo    %RESULTS%\%RUN%\meta.json
echo    %RESULTS%\%RUN%\baseline.json
echo    %RESULTS%\%RUN%\individuals.jsonl
echo    %RESULTS%\%RUN%\gen_summary.jsonl
echo    %RESULTS%\%RUN%\champion.json          (end of run)
echo    %RESULTS%\%RUN%\final_summary.json     (end of run)
echo    %RESULTS%\%RUN%\parsed_math\parsed_math_live.log    (live)
echo    %RESULTS%\%RUN%\parsed_math\parsed_math_live.jsonl  (live)
echo ===================================================================
echo.

REM ---- 1) Launch evolution via inner bat (detached, no window) ----
start "evo" /B "%EVO_INNER%"
echo [evo_run] evolution launched -- sleeping 8s for run-dir init...
powershell -NoProfile -Command "Start-Sleep -Seconds 8"

REM ---- 2) Launch watcher via inner bat ----
start "watch" /B "%WATCH_INNER%"
echo [evo_run] watcher launched.
echo.
echo [evo_run] Both processes detached. Verify with:
echo     PowerShell:  Get-Process python ^| Format-Table Id, StartTime
echo     PowerShell:  Get-Content -Wait "%EVO_LOG%"
echo.

endlocal
