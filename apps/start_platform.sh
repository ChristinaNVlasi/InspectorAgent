#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  SMA Inspector — Master Launch Script
#  Starts the complete platform:
#    1.  Global MCP Server            (port 3030)
#    2.  Knorr PCB Agent              (port 2829)
#    3.  Arcelik-Beko WM Agent        (port 2828)
#    3b. BORG Alternator Agent        (port 2827)
#    4.  Guidance Agent               (port 2830)
#  Then opens The Oracle UI in the browser.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

APPS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$APPS/logs"
mkdir -p "$LOGS"

# ── Activate virtualenv if present ───────────────────────────────────────────
VENV="$APPS/../.venv"
if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
  echo "  Using venv: $VENV"
fi

# ── Colour helpers ────────────────────────────────────────────────────────────
CYAN=$'\e[96m'; GOLD=$'\e[33m'; GREEN=$'\e[92m'; RED=$'\e[91m'; RST=$'\e[0m'
BOLD=$'\e[1m'

banner() {
  echo ""
  echo "${GOLD}${BOLD}  ╔══════════════════════════════════════════════════════╗${RST}"
  echo "${GOLD}${BOLD}  ║      ⚡  SMA INDUSTRIAL INSPECTION PLATFORM  ⚡      ║${RST}"
  echo "${GOLD}${BOLD}  ╚══════════════════════════════════════════════════════╝${RST}"
  echo ""
}

log_start()  { echo "${CYAN}  ▶ Starting $1 on port $2 …${RST}"; }
log_ok()     { echo "${GREEN}  ✔ $1 launched${RST}"; }
log_skip()   { echo "${GOLD}  ⚠ Port $2 busy — $1 already running${RST}"; }

kill_port() {
  local port=$1
  local pids i=0
  pids=$(lsof -ti :"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill    2>/dev/null || true
    sleep 1
    # escalate to SIGKILL if still alive
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
  # wait up to 5 s for the OS to release the port
  until ! lsof -i :"$port" -t &>/dev/null; do
    sleep 0.4; ((i++))
    [[ $i -ge 13 ]] && echo "${RED}  ✖ Port $port could not be freed${RST}" && return 1
  done
}

wait_port() {
  local port=$1 timeout=${2:-30} i=0
  until lsof -i :"$port" -t &>/dev/null; do
    sleep 0.5; ((i++))
    [[ $i -ge $((timeout*2)) ]] && echo "${RED}  ✖ Port $port timed out${RST}" && return 1
  done
  return 0
}

# ─────────────────────────────────────────────────────────────────────────────
banner

# Load common env if present
[[ -f "$APPS/.env" ]] && source "$APPS/.env"

# ── Require MCP_API_TOKEN — auto-generate on first run ───────────────────────
if [[ -z "${MCP_API_TOKEN:-}" ]]; then
  echo "${GOLD}  ⚙ MCP_API_TOKEN not found — generating a secure token …${RST}"
  MCP_API_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  # Persist to apps/.env so all future runs pick it up automatically
  echo "MCP_API_TOKEN=${MCP_API_TOKEN}" >> "$APPS/.env"
  echo "${GREEN}  ✔ Token generated and saved to apps/.env${RST}"
  echo "${GOLD}  ⚠ Keep apps/.env secret — never commit it to git${RST}"
fi
echo "${GREEN}  ✔ MCP_API_TOKEN ready${RST}"

# ── 1. Global MCP Server — 3030 ───────────────────────────────────────────────
log_start "Global MCP Server" 3030
kill_port 3030
nohup env MCP_API_TOKEN="$MCP_API_TOKEN" \
  python "$APPS/mcp_server/global_mcp_server.py" \
  > "$LOGS/global_mcp.log" 2>&1 &
MCP_PID=$!
echo "$MCP_PID" > "$LOGS/global_mcp.pid"
wait_port 3030 60 && log_ok "Global MCP Server (pid $MCP_PID)"

# ── 2. Knorr PCB Agent — 2829 ─────────────────────────────────────────────────
log_start "Knorr PCB Agent" 2829
kill_port 2829
nohup env MCP_SERVER_URL=http://localhost:3030 AGENT_PORT=2829 \
  MCP_API_TOKEN="$MCP_API_TOKEN" \
  LM_STUDIO_API_BASE="${LM_STUDIO_API_BASE:-http://20.10.10.152:11434/v1}" \
  LM_STUDIO_MODEL="${LM_STUDIO_MODEL:-openai/qwen3:30b-a3b-q8_0}" \
  LM_STUDIO_API_KEY="${LM_STUDIO_API_KEY:-ollama}" \
  python "$APPS/agent_knorr/adk_agent.py" \
  > "$LOGS/knorr_agent.log" 2>&1 &
KN_PID=$!
echo "$KN_PID" > "$LOGS/knorr_agent.pid"
wait_port 2829 && log_ok "Knorr PCB Agent (pid $KN_PID)"

# ── 3. Arcelik-Beko Agent — 2828 ──────────────────────────────────────────────
log_start "Arcelik-Beko WM Agent" 2828
kill_port 2828
nohup env MCP_SERVER_URL=http://localhost:3030 AGENT_PORT=2828 \
  MCP_API_TOKEN="$MCP_API_TOKEN" \
  LM_STUDIO_API_BASE="${LM_STUDIO_API_BASE:-http://20.10.10.152:11434/v1}" \
  LM_STUDIO_MODEL="${LM_STUDIO_MODEL:-openai/qwen3:30b-a3b-q8_0}" \
  LM_STUDIO_API_KEY="${LM_STUDIO_API_KEY:-ollama}" \
  python "$APPS/agent_arcelik/adk_agent.py" \
  > "$LOGS/arcelik_agent.log" 2>&1 &
AR_PID=$!
echo "$AR_PID" > "$LOGS/arcelik_agent.pid"
wait_port 2828 && log_ok "Arcelik-Beko WM Agent (pid $AR_PID)"

# ── 3b. BORG Alternator Agent — 2827 ──────────────────────────────────────────
log_start "BORG Alternator Agent" 2827
kill_port 2827
nohup env MCP_SERVER_URL=http://localhost:3030 AGENT_PORT=2827 \
  MCP_API_TOKEN="$MCP_API_TOKEN" \
  LM_STUDIO_API_BASE="${LM_STUDIO_API_BASE:-http://20.10.10.152:11434/v1}" \
  LM_STUDIO_MODEL="${LM_STUDIO_MODEL:-openai/qwen3:30b-a3b-q8_0}" \
  LM_STUDIO_API_KEY="${LM_STUDIO_API_KEY:-ollama}" \
  python "$APPS/agent_borg/adk_agent.py" \
  > "$LOGS/borg_agent.log" 2>&1 &
BG_PID=$!
echo "$BG_PID" > "$LOGS/borg_agent.pid"
wait_port 2827 && log_ok "BORG Alternator Agent (pid $BG_PID)"

# ── 3.5. Flask API for Noise & Vision — 5001 ──────────────────────────────────
log_start "Noise & Vision API" 5001
kill_port 5001
pushd "$APPS/wm_models/noise" > /dev/null
nohup python api.py > "$LOGS/noise_api.log" 2>&1 &
API_PID=$!
popd > /dev/null
echo "$API_PID" > "$LOGS/noise_api.pid"
wait_port 5001 && log_ok "Noise & Vision API (pid $API_PID)"

# ── 4. Guidance / Oracle Agent — 2830 ─────────────────────────────────────────
log_start "Guidance / Oracle Agent" 2830
kill_port 2830
nohup env GUIDANCE_PORT=2830 \
          KNORR_AGENT_URL=http://localhost:2829 \
          ARCELIK_AGENT_URL=http://localhost:2828 \
          BORG_AGENT_URL=http://localhost:2827 \
          MCP_SERVER_URL=http://localhost:3030 \
          MCP_API_TOKEN="$MCP_API_TOKEN" \
          LM_STUDIO_API_BASE="${LM_STUDIO_API_BASE:-http://localhost:11434/v1}" \
          LM_STUDIO_MODEL="${LM_STUDIO_MODEL:-openai/qwen3:30b-a3b-q8_0}" \
          LM_STUDIO_API_KEY="${LM_STUDIO_API_KEY:-ollama}" \
  python "$APPS/agents/guidance_agent.py" \
  > "$LOGS/guidance_agent.log" 2>&1 &
GD_PID=$!
echo "$GD_PID" > "$LOGS/guidance_agent.pid"
wait_port 2830 && log_ok "Guidance / Oracle Agent (pid $GD_PID)"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "${GOLD}${BOLD}  ┌──────────────────────────────────────────────────────┐${RST}"
echo "${GOLD}${BOLD}  │                PLATFORM STATUS                       │${RST}"
echo "${GOLD}${BOLD}  ├──────────────────────────────────────────────────────┤${RST}"
echo "${GOLD}  │  ${CYAN}Global MCP Server   ${RST}→  http://localhost:3030           ${GOLD}│${RST}"
echo "${GOLD}  │  ${CYAN}Knorr PCB Agent     ${RST}→  http://localhost:2829           ${GOLD}│${RST}"
echo "${GOLD}  │  ${CYAN}Arcelik WM Agent    ${RST}→  http://localhost:2828           ${GOLD}│${RST}"
echo "${GOLD}  │  ${CYAN}BORG Alternator     ${RST}→  http://localhost:2827           ${GOLD}│${RST}"
echo "${GOLD}  │  ${CYAN}Noise & Vision API  ${RST}→  http://localhost:5001           ${GOLD}│${RST}"
echo "${GOLD}  │  ${GREEN}▶ Oracle UI         ${RST}→  http://localhost:2830           ${GOLD}│${RST}"
echo "${GOLD}${BOLD}  └──────────────────────────────────────────────────────┘${RST}"
echo ""

# ── Open browser ───────────────────────────────────────────────────────────────
sleep 1
if command -v open &>/dev/null; then
  open "http://localhost:2830"
elif command -v xdg-open &>/dev/null; then
  xdg-open "http://localhost:2830"
fi

echo "${GREEN}${BOLD}  ⚡ All systems online. Logs in $LOGS/${RST}"
echo ""

# Keep script alive showing logs (Ctrl-C to exit)
trap 'echo "${RED}  Stopping platform …${RST}"; kill $MCP_PID $KN_PID $AR_PID $BG_PID $API_PID $GD_PID 2>/dev/null; exit 0' INT TERM

wait
