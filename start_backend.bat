@echo off
cd /d "%~dp0"
echo Starting SafeDrive Predictor...
start "" http://localhost:8080
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
pause
