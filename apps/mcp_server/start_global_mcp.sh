#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Start Global MCP Server — port 3030
#  Combines ALL inspection tools: Knorr PCB + Arcelik-Beko WM
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/logs/global_mcp_server.log"

mkdir -p "$SCRIPT_DIR/logs"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡  SMA Global Inspector MCP Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Port:  3030"
echo "   Log:   $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if already running
if lsof -i :3030 -t &>/dev/null; then
  echo "⚠️  Port 3030 already in use — killing existing process"
  kill "$(lsof -i :3030 -t)" 2>/dev/null || true
  sleep 1
fi

cd "$SCRIPT_DIR"
python global_mcp_server.py 2>&1 | tee "$LOG_FILE"
