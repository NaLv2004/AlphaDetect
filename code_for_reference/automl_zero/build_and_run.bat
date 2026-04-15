@echo off
REM ============================================================
REM AutoML-Zero Build and Run Script for Windows
REM ============================================================
REM Prerequisites:
REM   1. bazel-6.5.0.exe in the automl_zero directory
REM   2. Visual Studio 2022 Community with C++ build tools
REM   3. Git for Windows (provides bash.exe for Bazel)
REM   4. Internet connection (for first-time dependency download)
REM
REM Usage:
REM   build_and_run.bat [demo|full]
REM   Default: demo
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Set Bazel environment variables for VS 2022 + Git bash
set "BAZEL_SH=C:\Program Files\Git\usr\bin\bash.exe"
set "BAZEL_VC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC"
set "BAZEL_VC_FULL_VERSION=14.42.34433"

REM Check for bazel
if not exist bazel-6.5.0.exe (
    echo ERROR: bazel-6.5.0.exe not found in %~dp0
    echo Download it from: https://github.com/bazelbuild/bazel/releases/tag/6.5.0
    exit /b 1
)

set MODE=%1
if "%MODE%"=="" set MODE=demo

echo ============================================================
echo AutoML-Zero - Mode: %MODE%
echo ============================================================

if "%MODE%"=="demo" goto :run_demo
if "%MODE%"=="full" goto :run_full
echo Unknown mode: %MODE%. Use 'demo' or 'full'.
exit /b 1

:run_demo
echo.
echo [1/2] Building run_search_experiment (optimized)...
echo.
bazel-6.5.0.exe build -c opt ^
  --copt=-DMAX_SCALAR_ADDRESSES=4 ^
  --copt=-DMAX_VECTOR_ADDRESSES=3 ^
  --copt=-DMAX_MATRIX_ADDRESSES=1 ^
  :run_search_experiment
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo.
echo [2/2] Running demo: discovering linear regression algorithm...
echo       (Expected runtime: seconds to minutes)
echo.

REM Create log directory
if not exist logs mkdir logs

REM Generate log filename
set LOGFILE=logs\demo_run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

echo Log file: %LOGFILE%
echo.

bazel-out\x64_windows-opt\bin\run_search_experiment.exe ^
  --search_experiment_spec="search_tasks {tasks {scalar_linear_regression_task {} features_size: 4 num_train_examples: 100 num_valid_examples: 100 num_tasks: 1 eval_type: RMS_ERROR}} setup_size_init: 10 predict_size_init: 6 learn_size_init: 8 setup_ops: SCALAR_SUM_OP setup_ops: SCALAR_DIFF_OP setup_ops: SCALAR_PRODUCT_OP setup_ops: SCALAR_DIVISION_OP setup_ops: SCALAR_ABS_OP setup_ops: SCALAR_RECIPROCAL_OP setup_ops: SCALAR_EXP_OP setup_ops: SCALAR_LOG_OP setup_ops: SCALAR_CONST_SET_OP setup_ops: SCALAR_UNIFORM_SET_OP setup_ops: SCALAR_GAUSSIAN_SET_OP setup_ops: VECTOR_SUM_OP setup_ops: VECTOR_DIFF_OP setup_ops: VECTOR_PRODUCT_OP setup_ops: VECTOR_INNER_PRODUCT_OP setup_ops: VECTOR_OUTER_PRODUCT_OP setup_ops: MATRIX_VECTOR_PRODUCT_OP setup_ops: MATRIX_SUM_OP setup_ops: MATRIX_DIFF_OP setup_ops: MATRIX_PRODUCT_OP setup_ops: NO_OP predict_ops: SCALAR_SUM_OP predict_ops: SCALAR_DIFF_OP predict_ops: SCALAR_PRODUCT_OP predict_ops: SCALAR_DIVISION_OP predict_ops: SCALAR_ABS_OP predict_ops: SCALAR_RECIPROCAL_OP predict_ops: SCALAR_EXP_OP predict_ops: SCALAR_LOG_OP predict_ops: SCALAR_CONST_SET_OP predict_ops: SCALAR_UNIFORM_SET_OP predict_ops: VECTOR_SUM_OP predict_ops: VECTOR_DIFF_OP predict_ops: VECTOR_PRODUCT_OP predict_ops: VECTOR_INNER_PRODUCT_OP predict_ops: MATRIX_VECTOR_PRODUCT_OP predict_ops: NO_OP learn_ops: SCALAR_SUM_OP learn_ops: SCALAR_DIFF_OP learn_ops: SCALAR_PRODUCT_OP learn_ops: SCALAR_DIVISION_OP learn_ops: SCALAR_ABS_OP learn_ops: SCALAR_RECIPROCAL_OP learn_ops: SCALAR_EXP_OP learn_ops: SCALAR_LOG_OP learn_ops: SCALAR_CONST_SET_OP learn_ops: SCALAR_UNIFORM_SET_OP learn_ops: VECTOR_SUM_OP learn_ops: VECTOR_DIFF_OP learn_ops: VECTOR_PRODUCT_OP learn_ops: VECTOR_INNER_PRODUCT_OP learn_ops: VECTOR_OUTER_PRODUCT_OP learn_ops: MATRIX_VECTOR_PRODUCT_OP learn_ops: MATRIX_SUM_OP learn_ops: MATRIX_DIFF_OP learn_ops: MATRIX_PRODUCT_OP learn_ops: NO_OP fec {num_train_examples: 10 num_valid_examples: 10 cache_size: 100000} fitness_combination_mode: MEAN_FITNESS_COMBINATION population_size: 1000 tournament_size: 10 initial_population: RANDOM_ALGORITHM progress_every: 10000 max_train_steps: 100000 allowed_mutation_types {mutation_types: ALTER_PARAM_MUTATION_TYPE mutation_types: RANDOMIZE_INSTRUCTION_MUTATION_TYPE mutation_types: INSERT_INSTRUCTION_MUTATION_TYPE mutation_types: REMOVE_INSTRUCTION_MUTATION_TYPE mutation_types: TRADE_INSTRUCTION_MUTATION_TYPE mutation_types: RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE}" ^
  --select_tasks="tasks {scalar_linear_regression_task {} features_size: 4 num_train_examples: 100 num_valid_examples: 100 num_tasks: 1 eval_type: RMS_ERROR}" ^
  --final_tasks="tasks {scalar_linear_regression_task {} features_size: 4 num_train_examples: 100 num_valid_examples: 100 num_tasks: 1 eval_type: RMS_ERROR}" ^
  --max_experiments=5 ^
  --random_seed=1001 ^
  --sufficient_fitness=0.9999 > %LOGFILE% 2>&1

type %LOGFILE%

echo.
echo ============================================================
echo Run complete. Log saved to: %LOGFILE%
echo ============================================================
goto :eof

:run_full
echo.
echo [1/2] Building run_search_experiment (optimized, full search)...
echo.
bazel-6.5.0.exe build -c opt ^
  --copt=-DMAX_SCALAR_ADDRESSES=4 ^
  --copt=-DMAX_VECTOR_ADDRESSES=3 ^
  --copt=-DMAX_MATRIX_ADDRESSES=1 ^
  :run_search_experiment
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo.
echo [2/2] Running longer search: discovering ML algorithm from scratch...
echo       (Expected runtime: 30+ minutes)
echo.

if not exist logs mkdir logs

set LOGFILE=logs\full_run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

echo Log file: %LOGFILE%
echo.

bazel-out\x64_windows-opt\bin\run_search_experiment.exe ^
  --search_experiment_spec="search_tasks {tasks {scalar_linear_regression_task {} features_size: 4 num_train_examples: 100 num_valid_examples: 100 num_tasks: 10 eval_type: RMS_ERROR}} setup_size_init: 10 predict_size_init: 6 learn_size_init: 8 setup_ops: SCALAR_SUM_OP setup_ops: SCALAR_DIFF_OP setup_ops: SCALAR_PRODUCT_OP setup_ops: SCALAR_DIVISION_OP setup_ops: SCALAR_ABS_OP setup_ops: SCALAR_RECIPROCAL_OP setup_ops: SCALAR_EXP_OP setup_ops: SCALAR_LOG_OP setup_ops: SCALAR_CONST_SET_OP setup_ops: SCALAR_UNIFORM_SET_OP setup_ops: SCALAR_GAUSSIAN_SET_OP setup_ops: SCALAR_VECTOR_PRODUCT_OP setup_ops: SCALAR_BROADCAST_OP setup_ops: VECTOR_SUM_OP setup_ops: VECTOR_DIFF_OP setup_ops: VECTOR_PRODUCT_OP setup_ops: VECTOR_DIVISION_OP setup_ops: VECTOR_ABS_OP setup_ops: VECTOR_RECIPROCAL_OP setup_ops: VECTOR_HEAVYSIDE_OP setup_ops: VECTOR_INNER_PRODUCT_OP setup_ops: VECTOR_OUTER_PRODUCT_OP setup_ops: VECTOR_NORM_OP setup_ops: VECTOR_MEAN_OP setup_ops: VECTOR_ST_DEV_OP setup_ops: VECTOR_CONST_SET_OP setup_ops: VECTOR_UNIFORM_SET_OP setup_ops: VECTOR_GAUSSIAN_SET_OP setup_ops: MATRIX_SUM_OP setup_ops: MATRIX_DIFF_OP setup_ops: MATRIX_PRODUCT_OP setup_ops: MATRIX_DIVISION_OP setup_ops: MATRIX_ABS_OP setup_ops: MATRIX_RECIPROCAL_OP setup_ops: MATRIX_VECTOR_PRODUCT_OP setup_ops: MATRIX_NORM_OP setup_ops: MATRIX_TRANSPOSE_OP setup_ops: MATRIX_MATRIX_PRODUCT_OP setup_ops: MATRIX_MEAN_OP setup_ops: MATRIX_ST_DEV_OP setup_ops: MATRIX_CONST_SET_OP setup_ops: MATRIX_UNIFORM_SET_OP setup_ops: MATRIX_GAUSSIAN_SET_OP setup_ops: NO_OP predict_ops: SCALAR_SUM_OP predict_ops: SCALAR_DIFF_OP predict_ops: SCALAR_PRODUCT_OP predict_ops: SCALAR_DIVISION_OP predict_ops: SCALAR_ABS_OP predict_ops: SCALAR_RECIPROCAL_OP predict_ops: SCALAR_EXP_OP predict_ops: SCALAR_LOG_OP predict_ops: SCALAR_CONST_SET_OP predict_ops: SCALAR_UNIFORM_SET_OP predict_ops: VECTOR_SUM_OP predict_ops: VECTOR_DIFF_OP predict_ops: VECTOR_PRODUCT_OP predict_ops: VECTOR_INNER_PRODUCT_OP predict_ops: MATRIX_VECTOR_PRODUCT_OP predict_ops: NO_OP learn_ops: SCALAR_SUM_OP learn_ops: SCALAR_DIFF_OP learn_ops: SCALAR_PRODUCT_OP learn_ops: SCALAR_DIVISION_OP learn_ops: SCALAR_ABS_OP learn_ops: SCALAR_RECIPROCAL_OP learn_ops: SCALAR_EXP_OP learn_ops: SCALAR_LOG_OP learn_ops: SCALAR_CONST_SET_OP learn_ops: SCALAR_UNIFORM_SET_OP learn_ops: SCALAR_GAUSSIAN_SET_OP learn_ops: SCALAR_VECTOR_PRODUCT_OP learn_ops: SCALAR_BROADCAST_OP learn_ops: VECTOR_SUM_OP learn_ops: VECTOR_DIFF_OP learn_ops: VECTOR_PRODUCT_OP learn_ops: VECTOR_INNER_PRODUCT_OP learn_ops: VECTOR_OUTER_PRODUCT_OP learn_ops: VECTOR_NORM_OP learn_ops: MATRIX_VECTOR_PRODUCT_OP learn_ops: MATRIX_SUM_OP learn_ops: MATRIX_DIFF_OP learn_ops: MATRIX_PRODUCT_OP learn_ops: MATRIX_TRANSPOSE_OP learn_ops: MATRIX_MATRIX_PRODUCT_OP learn_ops: NO_OP fec {num_train_examples: 10 num_valid_examples: 10 cache_size: 100000} fitness_combination_mode: MEAN_FITNESS_COMBINATION population_size: 1000 tournament_size: 10 initial_population: RANDOM_ALGORITHM progress_every: 100000 max_train_steps: 200000000 allowed_mutation_types {mutation_types: ALTER_PARAM_MUTATION_TYPE mutation_types: RANDOMIZE_INSTRUCTION_MUTATION_TYPE mutation_types: INSERT_INSTRUCTION_MUTATION_TYPE mutation_types: REMOVE_INSTRUCTION_MUTATION_TYPE mutation_types: TRADE_INSTRUCTION_MUTATION_TYPE mutation_types: RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE} mutate_prob: 0.9" ^
  --select_tasks="tasks {scalar_linear_regression_task {} features_size: 4 num_train_examples: 1000 num_valid_examples: 100 num_tasks: 100 eval_type: RMS_ERROR}" ^
  --final_tasks="tasks {scalar_linear_regression_task {} features_size: 4 num_train_examples: 1000 num_valid_examples: 100 num_tasks: 100 eval_type: RMS_ERROR}" ^
  --max_experiments=0 ^
  --randomize_task_seeds ^
  --sufficient_fitness=0.9999 > %LOGFILE% 2>&1

type %LOGFILE%

echo.
echo ============================================================
echo Run complete. Log saved to: %LOGFILE%
echo ============================================================
goto :eof
