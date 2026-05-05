@echo off
cd /d "%~dp0"
echo Starting SafeDrive Predictor Backend...
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
pause
