@echo off
setlocal
cd /d "%~dp0"
"E:\MuJoCo\runtime\Scripts\python.exe" "%~dp0run_sequential_vehicles.py"
if errorlevel 1 (
    echo.
    echo MuJoCo_2 sequential demonstration failed. See the message above.
    pause
)
endlocal
