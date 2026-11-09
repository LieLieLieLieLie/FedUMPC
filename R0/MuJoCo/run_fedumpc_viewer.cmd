@echo off
setlocal
cd /d "%~dp0"
"E:\MuJoCo\runtime\Scripts\python.exe" "%~dp0run_six_algorithms.py" --method FedUMPC --viewer
if errorlevel 1 (
    echo.
    echo MuJoCo live demonstration failed. See the message above.
    pause
)
endlocal
