@echo off
title Deep Learning Model Trainer
echo 🧠 Starting Deep Learning Model Trainer...
echo.
echo Please wait while the application loads...
echo.

cd /d "%~dp0"
python Deep_Learning_GUI_App.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error: Could not start the application.
    echo.
    echo Possible solutions:
    echo 1. Make sure Python is installed
    echo 2. Install required packages by running: pip install tensorflow matplotlib pandas scikit-learn
    echo.
    pause
)
