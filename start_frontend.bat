@echo off
cd /d "%~dp0"
echo Starting SafeDrive Predictor Frontend...
streamlit run frontend/app.py --server.port 8501
pause
