#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  SMA Inspector — Local Start Script
#  Run from the repo root after running setup.sh once:
#    bash local-test/start.sh
#
#  Activates .venv and delegates to apps/start_platform.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
APPS="$REPO/apps"

CYAN=$'\e[96m'; GOLD=$'\e[33m'; GREEN=$'\e[92m'; RED=$'\e[91m'; RST=$'\e[0m'; BOLD=$'\e[1m'

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "${RED}${BOLD}  ✖ Virtual environment not found.${RST}"
  echo "  Run setup first:  bash local-test/setup.sh"
  exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Check Ollama is reachable (best-effort, don't block start)
ENV_FILE="$APPS/.env"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
OLLAMA_BASE="${LM_STUDIO_API_BASE:-http://localhost:11434/v1}"
OLLAMA_HEALTH="${OLLAMA_BASE%/v1}/api/tags"
if ! curl -sf "$OLLAMA_HEALTH" >/dev/null 2>&1; then
  echo "${GOLD}${BOLD}  ⚠ Ollama does not appear to be running at ${OLLAMA_BASE}${RST}"
  echo "  Start it with:  ollama serve"
  echo "  Then pull the model: ollama pull qwen3:30b-a3b-q8_0"
  echo "  Continuing anyway — agents will retry on first call …"
  echo ""
fi

# Delegate to the main platform script
exec bash "$APPS/start_platform.sh"
