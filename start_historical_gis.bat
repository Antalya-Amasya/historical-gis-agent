@echo off
setlocal

title Historical GIS Agent - Local Project
echo Historical GIS Agent - Local Project
echo Project: %~dp0
echo Starting local services. Readiness checks and browser launch follow.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_historical_gis.ps1"
if errorlevel 1 (
    echo.
    echo Historical GIS Agent could not be started.
    pause
    exit /b 1
)

endlocal
