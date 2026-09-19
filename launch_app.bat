@echo off
rem Double-click to set up and start PixelOpt. All the work is in launch.py;
rem this only finds a Python to run it with. Extra arguments are passed on,
rem for example:  launch_app.bat --check
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 launch.py %*
) else (
    python launch.py %*
)

if errorlevel 1 (
    echo.
    echo PixelOpt did not start. The reason is printed above; logs are in the logs folder.
    pause
)
endlocal
