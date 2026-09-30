@echo off
echo ========================================================
echo   Email Threat Classification - Machine Learning
echo ========================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo Starting Streamlit Application...
echo The web app will launch in your default browser.
echo Press Ctrl+C in this terminal window to stop the application.
echo.

streamlit run app/app.py

if errorlevel 1 (
    echo.
    echo [NOTICE] Streamlit ended or encountered an error.
    pause
)
