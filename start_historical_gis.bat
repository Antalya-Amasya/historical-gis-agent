@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_historical_gis.ps1"
if errorlevel 1 (
    echo.
    echo Historical GIS Agent could not be started.
    pause
    exit /b 1
)

endlocal
