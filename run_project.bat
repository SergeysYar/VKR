@echo off
setlocal

rem Launches the project from the directory where this file is stored.
cd /d "%~dp0"

where uv.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv was not found in PATH.
    echo Install uv and reopen this window, then run this file again.
    echo Installation: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

rem Friendly positional form:
rem   run_project.bat [tabs^|builder^|analysis^|showcase] [port] [SkipSync]
if "%~1"=="" (
    set "RUN_ARGS=-App tabs -Port 8513"
    goto :run
)

if /I "%~1"=="-h" goto :help
if /I "%~1"=="--help" goto :help
if /I "%~1"=="/h" goto :help

rem Pass PowerShell-style parameters unchanged, for example:
rem   run_project.bat -App builder -Port 8514 -SkipSync
if /I "%~1"=="-App" (
    set "RUN_ARGS=%*"
    goto :run
)
if /I "%~1"=="-Port" (
    set "RUN_ARGS=%*"
    goto :run
)
if /I "%~1"=="-SkipSync" (
    set "RUN_ARGS=%*"
    goto :run
)

set "RUN_APP=%~1"
set "RUN_PORT=%~2"
if "%RUN_PORT%"=="" set "RUN_PORT=8513"
set "RUN_ARGS=-App %RUN_APP% -Port %RUN_PORT%"
if /I "%~3"=="SkipSync" set "RUN_ARGS=%RUN_ARGS% -SkipSync"

:run
echo Starting VKR PointCloud Suite...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_showcase_uv.ps1" %RUN_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The application finished with error code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:help
echo Usage:
echo   run_project.bat
echo   run_project.bat tabs 8513
echo   run_project.bat builder 8514 SkipSync
echo   run_project.bat -App analysis -Port 8515 -SkipSync
echo.
echo Apps: tabs, builder, analysis, showcase.
exit /b 0
