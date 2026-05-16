@echo off
REM ===========================================================================
REM  evo_run.bat  --  One-click large-scale logged evolution (pop=200) + live
REM                   symbolic-expression watcher.
REM
REM  Usage (double-click or from a cmd prompt in this folder):
REM      evo_run.bat                          (defaults: pop=200, gens=20)
REM      evo_run.bat 300 30                   (custom pop, gens)
REM      evo_run.bat 200 20 my_run_name       (custom pop, gens, run name)
REM
REM  Launches two background processes:
REM    1. python ... experiments\run_logged_evolution.py    (the evolution)
REM    2. python ... experiments\parse_individuals_math.py --watch
REM       (tails individuals.jsonl, parses live math expressions on the fly)
REM
REM  Both processes write to dedicated .log files printed below; press any
REM  key after the launch to view them.  The script DOES NOT block on the
REM  runs; close this window and they keep running.
REM ===========================================================================

setlocal EnableDelayedExpansion

REM ---- Pin paths so the script works from any cwd ----------------------------
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"

set "PY=C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe"
if not exist "%PY%" (
    echo [ERROR] python not found: %PY%
    echo Edit evo_run.bat and fix the PY= line.
    pause
    exit /b 1
)

REM ---- Args with sensible defaults --------------------------------------------
set "POP=%~1"
if "%POP%"=="" set "POP=200"
set "GENS=%~2"
if "%GENS%"=="" set "GENS=20"
set "RUN_NAME=%~3"
if "%RUN_NAME%"=="" (
    for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
    set "RUN_NAME=dce_e2e_pop!POP!_gens!GENS!_!TS!"
)

REM ---- Fixed experiment knobs (edit here if needed) --------------------------
set "SEED=2025"
set "SNR_LIST=-3,-2,-1"
set "N_FRAMES=6"
set "MAX_ITER=8"
set "RAND_MIN=4"
set "RAND_MAX=30"
set "WORKERS=16"
set "DCE_MAX_ITER=8"
set "DCE_DECIMALS=6"
set "DCE_MAX_PASSES=800"
set "DCE_SNR=-2"
set "DCE_N_FRAMES=1"

REM ---- Output layout ---------------------------------------------------------
set "OUT_ROOT=%HERE%\results\logged_evolution"
set "RUN_DIR=%OUT_ROOT%\%RUN_NAME%"
set "CONSOLE_LOG=%OUT_ROOT%\%RUN_NAME%.console.log"
set "CONSOLE_ERR=%OUT_ROOT%\%RUN_NAME%.console.err.log"
set "PARSE_DIR=%RUN_DIR%\parsed_math"
set "WATCHER_LOG=%PARSE_DIR%\watcher.console.log"
set "WATCHER_ERR=%PARSE_DIR%\watcher.console.err.log"

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%RUN_DIR%"  mkdir "%RUN_DIR%"
if not exist "%PARSE_DIR%" mkdir "%PARSE_DIR%"

echo.
echo ===========================================================================
echo  RUN_NAME    : %RUN_NAME%
echo  POP / GENS  : %POP% / %GENS%
echo  SEED        : %SEED%
echo  SNR_LIST    : %SNR_LIST%
echo  DCE         : ON  (snr=%DCE_SNR% dB, max_iter=%DCE_MAX_ITER%, decimals=%DCE_DECIMALS%)
echo  WORKERS     : %WORKERS%
echo.
echo  Run dir     : %RUN_DIR%
echo  Console log : %CONSOLE_LOG%
echo  Console err : %CONSOLE_ERR%
echo  Watcher log : %WATCHER_LOG%
echo  Watcher err : %WATCHER_ERR%
echo.
echo  Live products in %RUN_DIR%\:
echo     meta.json                  (full cfg, written at start)
echo     baseline.json              (OMS reference)
echo     individuals.jsonl          (1 line per individual per gen)
echo     gen_summary.jsonl          (1 line per gen)
echo     champion.json              (written at end)
echo     final_summary.json         (written at end)
echo     parsed_math\parsed_math_live.jsonl   (live, by watcher)
echo     parsed_math\parsed_math_live.log     (live, human readable)
echo ===========================================================================
echo.

REM ---- Write inner evolution launcher (single line avoids nested-quote issues) ----
set "EVO_INNER=%OUT_ROOT%\%RUN_NAME%.__evo_inner.bat"
set "WATCH_INNER=%OUT_ROOT%\%RUN_NAME%.__watch_inner.bat"

> "%EVO_INNER%" echo @echo off
>> "%EVO_INNER%" echo "%PY%" -B -u "%HERE%\experiments\run_logged_evolution.py" --run-name %RUN_NAME% --pop-size %POP% --gens %GENS% --seed %SEED% --snr-list=%SNR_LIST% --n-frames %N_FRAMES% --max-iter %MAX_ITER% --rand-min-size %RAND_MIN% --rand-max-size %RAND_MAX% --workers %WORKERS% --cpp-seeder --dce-bp --dce-bp-max-iter %DCE_MAX_ITER% --dce-bp-decimals %DCE_DECIMALS% --dce-bp-max-passes %DCE_MAX_PASSES% --dce-bp-max-decode-evals -1 --dce-bp-threads 0 --dce-bp-snr-db=%DCE_SNR% --dce-bp-n-frames %DCE_N_FRAMES% 1^> "%CONSOLE_LOG%" 2^> "%CONSOLE_ERR%"

> "%WATCH_INNER%" echo @echo off
>> "%WATCH_INNER%" echo "%PY%" -B -u "%HERE%\experiments\parse_individuals_math.py" %RUN_NAME% --watch --poll-interval 3.0 --max-chars 500 1^> "%WATCHER_LOG%" 2^> "%WATCHER_ERR%"

echo [launch] starting evolution ...
start "evo_%RUN_NAME%" /B "%EVO_INNER%"

REM ---- Sleep 8s for run-dir init, then start watcher (watch_loop handles
REM      missing individuals.jsonl by polling, so no need to wait here).
echo [launch] sleeping 8s for run-dir init ...
powershell -NoProfile -Command "Start-Sleep -Seconds 8" >nul

echo [launch] starting math watcher ...
start "watch_%RUN_NAME%" /B "%WATCH_INNER%"

echo.
echo [done] both processes launched in background.
echo.
echo  To tail the evolution console:
echo     powershell -NoProfile -Command "Get-Content -Wait '%CONSOLE_LOG%'"
echo  To tail the live math watcher:
echo     powershell -NoProfile -Command "Get-Content -Wait '%PARSE_DIR%\parsed_math_live.log'"
echo  To stop everything:
echo     taskkill /F /FI "WINDOWTITLE eq evo_%RUN_NAME%*"
echo     taskkill /F /FI "WINDOWTITLE eq watch_%RUN_NAME%*"
echo.

endlocal
