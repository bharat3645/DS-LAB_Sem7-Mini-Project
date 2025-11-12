@echo off
REM Crime Analytics Dashboard - Quick Start Script for Windows

echo ========================================
echo Crime Analytics Dashboard - Starting...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if required CSV files exist
if not exist "districtwise-missing-persons-2017-2022.csv" (
    echo [ERROR] Missing file: districtwise-missing-persons-2017-2022.csv
    echo Please ensure the CSV file is in the current directory.
    pause
    exit /b 1
)

if not exist "districtwise-ipc-crime-by-juveniles-2017-onwards.csv" (
    echo [ERROR] Missing file: districtwise-ipc-crime-by-juveniles-2017-onwards.csv
    echo Please ensure the CSV file is in the current directory.
    pause
    exit /b 1
)

echo [OK] Data files found
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -q streamlit pandas numpy matplotlib seaborn scikit-learn plotly
    echo [OK] Packages installed
) else (
    echo [OK] Streamlit is already installed
)

echo.
echo Starting Streamlit application...
echo.
echo Dashboard will open in your browser automatically
echo If not, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Run streamlit
streamlit run streamlit_app.py
