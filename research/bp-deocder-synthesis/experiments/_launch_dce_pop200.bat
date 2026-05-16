@echo off
setlocal
set RUN=%1
set RUNDIR=D:\ChannelCoding\RCOM\AlphaDetect\research\bp-deocder-synthesis\results\logged_evolution\%RUN%
set PY=C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe
cd /d D:\ChannelCoding\RCOM\AlphaDetect\research\bp-deocder-synthesis
"%PY%" -B experiments\run_logged_evolution.py --run-name %RUN% --pop-size 200 --gens 20 --snr-list=-3,-2,-1 --n-frames 6 --cpp-seeder --dce-bp --dce-bp-threads 0 1>"%RUNDIR%.console.log" 2>"%RUNDIR%.console.err.log"
echo EXIT_CODE=%ERRORLEVEL% >>"%RUNDIR%.console.log"
endlocal
