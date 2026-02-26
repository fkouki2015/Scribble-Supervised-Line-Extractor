start "Backend" cmd /k "cd /d %~dp0backend && uvicorn server:app --reload --host 0.0.0.0 --port 8000"
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 3 /nobreak >nul
start "" "http://localhost:5173"
