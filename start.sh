#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/backend"
uvicorn server:app --reload --host 0.0.0.0 --port 8000 &

cd "$SCRIPT_DIR/frontend"
npm run dev &

sleep 3

if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:5173"
elif command -v open &>/dev/null; then
    open "http://localhost:5173"
fi

wait
