#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  SMA Inspector — First-Time Local Setup
#  Run once from the repo root:
#    bash local-test/setup.sh
#
#  What this does:
#    1. Checks system prerequisites (Python 3.10+, ffmpeg)
#    2. Creates a Python virtual environment at .venv/
#    3. Installs all Python dependencies
#    4. Copies .env.example → apps/.env if not already present
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"

CYAN=$'\e[96m'; GOLD=$'\e[33m'; GREEN=$'\e[92m'; RED=$'\e[91m'; RST=$'\e[0m'; BOLD=$'\e[1m'

step()  { echo "${CYAN}${BOLD}▶ $*${RST}"; }
ok()    { echo "${GREEN}  ✔ $*${RST}"; }
warn()  { echo "${GOLD}  ⚠ $*${RST}"; }
fatal() { echo "${RED}${BOLD}  ✖ $*${RST}"; exit 1; }

echo ""
echo "${GOLD}${BOLD}  ╔══════════════════════════════════════════════════════╗${RST}"
echo "${GOLD}${BOLD}  ║       SMA Inspector — Local Setup                   ║${RST}"
echo "${GOLD}${BOLD}  ╚══════════════════════════════════════════════════════╝${RST}"
echo ""

# ── 1. Python 3.10+ ──────────────────────────────────────────────────────────
step "Checking Python version …"
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    major=${ver%%.*}; minor=${ver##*.}
    if [[ $major -ge 3 && $minor -ge 10 ]]; then
      PYTHON="$candidate"
      ok "Found $PYTHON ($ver)"
      break
    fi
  fi
done
[[ -z "$PYTHON" ]] && fatal "Python 3.10 or higher is required. Install it from https://python.org"

# ── 2. ffmpeg (required for audio processing) ────────────────────────────────
step "Checking ffmpeg …"
if command -v ffmpeg &>/dev/null; then
  ok "ffmpeg found"
else
  warn "ffmpeg not found — audio inspection will not work."
  warn "  macOS : brew install ffmpeg"
  warn "  Ubuntu: sudo apt-get install -y ffmpeg"
fi

# ── 3. Virtual environment ────────────────────────────────────────────────────
step "Creating virtual environment at .venv/ …"
if [[ -d "$VENV" ]]; then
  ok "Virtual environment already exists — skipping creation"
else
  "$PYTHON" -m venv "$VENV"
  ok "Virtual environment created"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV/bin/activate"
ok "Virtual environment activated"

pip install --upgrade pip --quiet

# ── 4. Install dependencies ───────────────────────────────────────────────────
step "Installing Python dependencies …"

REQ_FILES=(
  "apps/agents/requirements.txt"
  "apps/agent_knorr/requirements.txt"
  "apps/agent_arcelik/requirements.txt"
  "apps/agent_borg/requirements.txt"
  "apps/wm_models/noise/requirements.txt"
  "apps/pcb_defect_detector/requirements.txt"
)

for req in "${REQ_FILES[@]}"; do
  full="$REPO/$req"
  if [[ -f "$full" ]]; then
    echo "  Installing $req …"
    pip install -r "$full" --quiet
    ok "$req installed"
  else
    warn "$req not found — skipping"
  fi
done

# Pin packages to resolve cross-dependency conflicts (chromadb vs transformers)
step "Pinning transformers/tokenizers for compatibility …"
pip install "transformers==5.4.0" "tokenizers==0.22.2" "chromadb==0.5.23" --no-deps --quiet
ok "Compatibility pins applied"

# Vision model requirements (nested path)
VISION_REQ="$REPO/apps/wm_models/vision/ai_vision/requirements.txt"
if [[ -f "$VISION_REQ" ]]; then
  echo "  Installing WM vision requirements …"
  pip install -r "$VISION_REQ" --quiet
  ok "WM vision requirements installed"
fi

# ── 5. apps/.env ─────────────────────────────────────────────────────────────
step "Setting up apps/.env …"
ENV_FILE="$REPO/apps/.env"
EXAMPLE="$REPO/local-test/.env.example"

if [[ -f "$ENV_FILE" ]]; then
  ok "apps/.env already exists — not overwriting"
else
  cp "$EXAMPLE" "$ENV_FILE"
  ok "apps/.env created from template"
  warn "Open apps/.env and set LM_STUDIO_API_BASE to your Ollama host if it's not localhost"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "${GREEN}${BOLD}  ✔ Setup complete!${RST}"
echo ""
echo "  Next steps:"
echo "    1. Make sure Ollama is running with: ollama serve"
echo "       and the model pulled:             ollama pull qwen3:30b-a3b-q8_0"
echo "    2. (Optional) Edit apps/.env to point LM_STUDIO_API_BASE at your Ollama host"
echo "    3. Start the platform:"
echo "       ${CYAN}bash local-test/start.sh${RST}"
echo ""
