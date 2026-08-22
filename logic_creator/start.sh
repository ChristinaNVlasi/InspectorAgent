#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SMA Logic Creator — start script
# Starts the Python Logic Engine and opens the UI in the default browser.
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/.venv"
PORT=8765
PID_FILE="$SCRIPT_DIR/.engine.pid"

# ── 1. Resolve Python interpreter ────────────────────────────────────────────
if [[ -f "$VENV/bin/python" ]]; then
  PYTHON="$VENV/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

echo "Using Python: $PYTHON"

# ── 2. Install / verify dependencies ─────────────────────────────────────────
echo "Checking dependencies…"
"$PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── 3. Kill any existing engine on the same port ─────────────────────────────
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Stopping existing engine (PID $OLD_PID)…"
    kill "$OLD_PID"
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

# Also clear anything squatting on the port
SQUATTER=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$SQUATTER" ]]; then
  echo "Clearing port $PORT (PID $SQUATTER)…"
  kill "$SQUATTER" 2>/dev/null || true
  sleep 1
fi

# ── 4. Start the Logic Engine ─────────────────────────────────────────────────
echo "Starting Logic Engine on http://localhost:$PORT …"
cd "$SCRIPT_DIR"
"$PYTHON" logic_engine.py &
ENGINE_PID=$!
echo "$ENGINE_PID" > "$PID_FILE"

# ── 5. Wait until the engine is ready ────────────────────────────────────────
echo -n "Waiting for engine"
for i in $(seq 1 20); do
  sleep 0.5
  if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo " ready."
    break
  fi
  echo -n "."
  if [[ $i -eq 20 ]]; then
    echo ""
    echo "ERROR: engine did not start within 10 s. Check output above." >&2
    exit 1
  fi
done

# ── 6. Open UI in browser ─────────────────────────────────────────────────────
UI_URL="http://localhost:$PORT"
echo "Opening UI: $UI_URL"
if command -v open &>/dev/null; then          # macOS
  open "$UI_URL"
elif command -v xdg-open &>/dev/null; then   # Linux
  xdg-open "$UI_URL"
elif command -v start &>/dev/null; then      # Windows / Git Bash
  start "$UI_URL"
fi

# ── 7. Keep running — Ctrl-C to stop ─────────────────────────────────────────
echo ""
echo "  Logic Engine running  →  http://localhost:$PORT"
echo "  Press Ctrl-C to stop."
echo ""

trap "echo ''; echo 'Stopping…'; kill $ENGINE_PID 2>/dev/null; rm -f $PID_FILE; exit 0" INT TERM

wait "$ENGINE_PID"
rm -f "$PID_FILE"
