@echo off
setlocal
set RUN=%1
set RUNDIR=D:\ChannelCoding\RCOM\AlphaDetect\research\bp-deocder-synthesis\results\logged_evolution\%RUN%
set PY=C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe
cd /d D:\ChannelCoding\RCOM\AlphaDetect\research\bp-deocder-synthesis
if not exist "%RUNDIR%\parsed_math" mkdir "%RUNDIR%\parsed_math"
"%PY%" -B experiments\parse_individuals_math.py %RUN% --watch --poll-interval 2.0 1>"%RUNDIR%\parsed_math\watcher.console.log" 2>"%RUNDIR%\parsed_math\watcher.console.err.log"
endlocal
