#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Start Guidance / Selector Agent — port 2830
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/guidance_agent.log"

mkdir -p "$SCRIPT_DIR/logs"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡  SMA Guidance / Selector Agent"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Oracle UI:  http://localhost:2830"
echo "   Log:        $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Kill existing
if lsof -i :2830 -t &>/dev/null; then
  echo "⚠️  Port 2830 already in use — killing existing process"
  kill "$(lsof -i :2830 -t)" 2>/dev/null || true
  sleep 1
fi

cd "$SCRIPT_DIR"
python guidance_agent.py 2>&1 | tee "$LOG_FILE"
