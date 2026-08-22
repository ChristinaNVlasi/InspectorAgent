#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  SMA Inspector — Local Stop Script
#  Stops the LOCAL platform only. Does NOT touch the deployed VM.
#  Run from the repo root:
#    bash local-test/stop.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS="$REPO/apps"

CYAN=$'\e[96m'; GREEN=$'\e[92m'; RST=$'\e[0m'; BOLD=$'\e[1m'

echo "${CYAN}${BOLD}Stopping local SMA Inspector platform…${RST}"

if [[ ! -f "$APPS/stop_platform.sh" ]]; then
  echo "stop_platform.sh not found at $APPS/stop_platform.sh"
  exit 1
fi

bash "$APPS/stop_platform.sh"

echo "${GREEN}${BOLD}  ✔ Local platform stopped.${RST}"
