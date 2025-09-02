@echo off
REM EAGLE Docker Runner Script for Windows
REM This script provides convenient commands to run EAGLE inference in Docker

setlocal enabledelayedexpansion

REM Check if Docker and Docker Compose are installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker first.
    exit /b 1
)

where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    exit /b 1
)

REM Get the directory of this script
set SCRIPT_DIR=%~dp0

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help

if "%COMMAND%"=="build" goto build
if "%COMMAND%"=="run" goto run
if "%COMMAND%"=="shell" goto shell
if "%COMMAND%"=="jupyter" goto jupyter
if "%COMMAND%"=="logs" goto logs
if "%COMMAND%"=="stop" goto stop
if "%COMMAND%"=="cleanup" goto cleanup
if "%COMMAND%"=="help" goto help
goto help

:build
echo [INFO] Building EAGLE Docker image...
cd /d "%SCRIPT_DIR%"
docker-compose build eagle-inference
if %errorlevel% equ 0 (
    echo [SUCCESS] Docker image built successfully!
) else (
    echo [ERROR] Failed to build Docker image!
    exit /b 1
)
goto end

:run
set BASE_MODEL=%2
set EA_MODEL=%3
set BASE_DEVICE=%4
set DRAFT_DEVICE=%5
set MAX_TOKENS=%6
set NUM_RUNS=%7

if "%BASE_MODEL%"=="" set BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
if "%EA_MODEL%"=="" set EA_MODEL=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
if "%BASE_DEVICE%"=="" set BASE_DEVICE=cuda:0
if "%DRAFT_DEVICE%"=="" set DRAFT_DEVICE=cpu
if "%MAX_TOKENS%"=="" set MAX_TOKENS=128
if "%NUM_RUNS%"=="" set NUM_RUNS=1

echo [INFO] Running EAGLE multi-GPU comparison...
echo [INFO] Base model: %BASE_MODEL%
echo [INFO] EA model: %EA_MODEL%
echo [INFO] Base device: %BASE_DEVICE%
echo [INFO] Draft device: %DRAFT_DEVICE%
echo [INFO] Max tokens: %MAX_TOKENS%
echo [INFO] Number of runs: %NUM_RUNS%

cd /d "%SCRIPT_DIR%"
docker-compose run --rm eagle-inference bash -c "python -m eagle.evaluation.compare_multi_gpu --base-model '%BASE_MODEL%' --ea-model '%EA_MODEL%' --base-device '%BASE_DEVICE%' --draft-device '%DRAFT_DEVICE%' --max-new-tokens %MAX_TOKENS% --num-runs %NUM_RUNS% --prompt 'Write a short story about a robot who learns to feel emotions.' --use-eagle3"

if %errorlevel% equ 0 (
    echo [SUCCESS] Comparison completed!
) else (
    echo [ERROR] Comparison failed!
    exit /b 1
)
goto end

:shell
echo [INFO] Starting interactive shell in EAGLE container...
cd /d "%SCRIPT_DIR%"
docker-compose run --rm eagle-inference bash
goto end

:jupyter
echo [INFO] Starting Jupyter Lab...
cd /d "%SCRIPT_DIR%"
docker-compose --profile jupyter up -d eagle-jupyter
echo [SUCCESS] Jupyter Lab started at http://localhost:8888
goto end

:logs
cd /d "%SCRIPT_DIR%"
docker-compose logs -f eagle-inference
goto end

:stop
echo [INFO] Stopping all Docker services...
cd /d "%SCRIPT_DIR%"
docker-compose down
echo [SUCCESS] All services stopped!
goto end

:cleanup
echo [INFO] Cleaning up Docker resources...
cd /d "%SCRIPT_DIR%"
docker-compose down --volumes --remove-orphans
docker system prune -f
echo [SUCCESS] Cleanup completed!
goto end

:help
echo EAGLE Docker Runner for Windows
echo.
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   build                           Build the Docker image
echo   run [base_model] [ea_model]     Run multi-GPU comparison
echo       [base_device] [draft_device] [max_tokens] [num_runs]
echo   shell                           Start interactive shell
echo   jupyter                         Start Jupyter Lab
echo   logs                            Show container logs
echo   stop                            Stop all services
echo   cleanup                         Clean up Docker resources
echo   help                            Show this help message
echo.
echo Examples:
echo   %0 build
echo   %0 run
echo   %0 run meta-llama/Llama-3.1-8B-Instruct yuhuili/EAGLE3-LLaMA3.1-Instruct-8B cuda:0 cpu 256 3
echo   %0 shell
echo   %0 jupyter
echo.
goto end

:end
endlocal