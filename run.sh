#!/usr/bin/env bash
set -e

PORT="${PORT:-8501}"

streamlit run app.py --server.port "$PORT" --server.headless true &
APP_PID=$!

sleep 2
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:${PORT}" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "http://localhost:${PORT}" >/dev/null 2>&1 || true
fi

echo "Streamlit running at http://localhost:${PORT} (pid ${APP_PID})"
wait ${APP_PID}
