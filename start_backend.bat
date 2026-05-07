@echo off
cd /d "%~dp0"
echo Starting SafeDrive Predictor backend...
start "SafeDrive Backend" cmd /c python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080
echo Waiting for server to start...
timeout /t 5 /nobreak >nul
start "" http://localhost:8080
echo.
echo Server is running at http://localhost:8080
echo Keep the "SafeDrive Backend" window open. Close it to stop the server.
pause
