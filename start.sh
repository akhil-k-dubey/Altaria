#!/usr/bin/env bash
# Stark Holo-CAD Quick Launcher
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000

# Check if port 8000 is already active
if ss -tuln | grep -q ":${PORT} "; then
  echo "[*] Holo-CAD server is already active on http://localhost:${PORT}"
  xdg-open "http://localhost:${PORT}" 2>/dev/null || true
  exit 0
fi

echo "[*] Starting Holo-CAD + Local Qwen2.5-VL server on http://localhost:${PORT}..."
/home/akhil/Documents/impel/.venv/bin/python "${DIR}/backend/qwen_service.py" "${PORT}" &
sleep 2
xdg-open "http://localhost:${PORT}" 2>/dev/null || true
