@echo off
set "PYTHONPATH=%~dp0"
set "PYTHONUNBUFFERED=1"
"C:\ProgramData\anaconda3\envs\AutoGenOld\python.exe" "%~dp0train_gnn.py" --gens 4 --proposals 1000 --snr-start 16 --snr-target 16 --warmstart-gens 2 --warmstart-trials 1 --warmstart-timeout 0.5 --warmstart-eval-workers 8 --proposal-batch --proposal-batch-size 128 --progress --viz-grafts-per-gen 40 --n-trials 5 --timeout 1.5 --seed 17 --micro-pop-size 16 --micro-generations 2
