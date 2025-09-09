@echo off
title Student Performance Predictor
echo Starting Streamlit App...
start "" /min cmd /k "cd /d %~dp0 && streamlit run streamlit_standalone.py"

echo.
echo The app will open in your default browser shortly...
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
pause
