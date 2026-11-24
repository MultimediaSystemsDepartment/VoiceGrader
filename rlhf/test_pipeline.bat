@echo off
REM Full RLHF-ASR training pipeline smoke-test.
REM Mirrors exactly what "Trigger Training Now" fires in the admin GUI.
REM Double-click from Windows Explorer, or run from the project root.
REM
REM Customise behaviour with environment variables before running:
REM   set GENERATE_N=10          (default 200 -- use small number for a quick test)
REM   set GENERATE_N_CANDIDATES=6
REM   set GENERATE_DATASETS=admedvoice
REM   set SKIP_GENERATE=0        (set to 1 to skip pair generation)

cd /d "%~dp0"
setlocal enabledelayedexpansion

REM ── Defaults (override with env vars before running) ──────────────────────
if not defined GENERATE_N           set GENERATE_N=200
if not defined GENERATE_N_CANDIDATES set GENERATE_N_CANDIDATES=6
if not defined GENERATE_DATASETS    set GENERATE_DATASETS=admedvoice
if not defined SKIP_GENERATE        set SKIP_GENERATE=0
if not defined EVAL_MAX_SAMPLES     set EVAL_MAX_SAMPLES=400

REM ── Detect latest reward model checkpoint ────────────────────────────────
set RM_CHECKPOINT_ARG=
if exist "_outputs\reward_model" (
    for /f "delims=" %%d in ('dir /b /ad /o:n "_outputs\reward_model\reward_model_v*" 2^>nul') do (
        set RM_CHECKPOINT_ARG=--rm_checkpoint _outputs\reward_model\%%d
    )
)

REM ── Header ───────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  RLHF-ASR -- Full pipeline test  (%date% %time%)
echo  GENERATE_N=%GENERATE_N%  N_CANDIDATES=%GENERATE_N_CANDIDATES%
if defined EVAL_MAX_SAMPLES if not "%EVAL_MAX_SAMPLES%"=="" (
    echo  EVAL_MAX_SAMPLES=%EVAL_MAX_SAMPLES%  ^(random stable subset, seed=42^)
) else (
    echo  EVAL_MAX_SAMPLES=full test set
)
if defined RM_CHECKPOINT_ARG (
    echo  Reward model: %RM_CHECKPOINT_ARG%
) else (
    echo  Reward model: none ^(beam-rank fallback^)
)
echo ============================================================

set PASS_COUNT=0
set FAIL_COUNT=0
set FAILED_STEPS=

REM ── Helper macro – call as: call :run_step "description" command args... ─
REM  (batch has no real functions; we use labels + goto instead)

REM ── Step 1: Generate pairs ────────────────────────────────────────────────
if "%SKIP_GENERATE%"=="1" (
    echo.
    echo [SKIP] Step 1/6: generate pairs  ^(SKIP_GENERATE=1^)
    goto step2
)
echo.
echo [RUN ] Step 1/6: generate pairs
python3.12 run_generate_pairs.py ^
    --n %GENERATE_N% ^
    --datasets %GENERATE_DATASETS% ^
    --n_candidates %GENERATE_N_CANDIDATES% ^
    %RM_CHECKPOINT_ARG%
call :check_rc "generate pairs"

:step2
REM ── Step 2: Train reward model ───────────────────────────────────────────
echo.
echo [RUN ] Step 2/6: train reward model
python3.12 run_train_reward.py
call :check_rc "train reward model"

:step3
REM ── Step 3: Score rewards ────────────────────────────────────────────────
echo.
echo [RUN ] Step 3/6: score rewards
python3.12 run_score_rewards.py
call :check_rc "score rewards"

:step4
REM ── Step 4: SFT fine-tuning ──────────────────────────────────────────────
echo.
echo [RUN ] Step 4/6: SFT fine-tuning
python3.12 run_sft.py --epochs 3
call :check_rc "SFT fine-tuning"

:step5
REM ── Step 5: DPO fine-tuning ──────────────────────────────────────────────
echo.
echo [RUN ] Step 5/6: DPO fine-tuning
python3.12 run_rlhf.py --resume _outputs\checkpoints\sft
call :check_rc "DPO fine-tuning"

:step6
REM ── Step 6: Evaluate WER ─────────────────────────────────────────────────
echo.
echo [RUN ] Step 6/6: evaluate WER
python3.12 run_evaluate.py --report
call :check_rc "evaluate WER"

REM ── Summary ───────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  SUMMARY
echo  Passed: %PASS_COUNT%   Failed: %FAIL_COUNT%
if not "%FAILED_STEPS%"=="" (
    echo  Failed steps: %FAILED_STEPS%
    echo  Overall: FAIL
) else (
    echo  Overall: PASS
)
echo ============================================================
echo.
pause
exit /b %FAIL_COUNT%

REM ── Subroutine: record pass/fail for the last command ─────────────────────
:check_rc
if %ERRORLEVEL% EQU 0 (
    echo [PASS] %~1
    set /a PASS_COUNT+=1
) else (
    echo [FAIL] %~1  ^(exit code %ERRORLEVEL%^)
    set /a FAIL_COUNT+=1
    if "%FAILED_STEPS%"=="" (
        set FAILED_STEPS=%~1
    ) else (
        set FAILED_STEPS=%FAILED_STEPS% ^| %~1
    )
)
goto :eof
