# SMA Inspector — Local Test Guide

Everything you need to run the full platform on a single machine from scratch.

---

> **Office network / VPN required**
> The LLM models run on the shared office inference server (`20.10.10.152`). Your machine must be **on the office network or connected via corporate VPN** before starting the platform. No local model installation is needed.

---

## Prerequisites

| Requirement | Min version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 or 3.12 recommended |
| ffmpeg | any | Required for audio inspection (WM agent) |
| Office network / VPN | — | Required to reach the shared LLM server |

### Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install -y ffmpeg
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd sma-inspector-agent

# 2. Connect to the office VPN (if working remotely)

# 3. One-time setup — installs all dependencies
bash local-test/setup.sh

# 4. Start the platform
bash local-test/start.sh
```

Open **http://localhost:2830** in your browser.

---

## What setup.sh does

Run it **once**. It:

1. Checks Python 3.10+ is available
2. Warns if ffmpeg is missing
3. Creates `.venv/` at the repo root
4. Installs all Python dependencies from every `requirements.txt`
5. Copies `local-test/.env.example` → `apps/.env` (only if `apps/.env` doesn't exist yet)

---

## What start.sh does

Run it **every time** you want to start the platform. It:

1. Activates `.venv/`
2. Checks the office LLM server is reachable over the network (warns but doesn't block)
3. Calls `apps/start_platform.sh` which:
   - Auto-generates `MCP_API_TOKEN` on first run and saves it to `apps/.env`
   - Kills any stale processes on the required ports
   - Starts all 6 services in the background
   - Opens the UI in your browser

---

## Services & Ports

| Service | Port | Log |
|---|---|---|
| Global MCP Server | 3030 | `apps/logs/global_mcp.log` |
| Knorr PCB Agent | 2829 | `apps/logs/knorr_agent.log` |
| Arcelik-Beko WM Agent | 2828 | `apps/logs/arcelik_agent.log` |
| BORG Alternator Agent | 2827 | `apps/logs/borg_agent.log` |
| Noise & Vision API | 5001 | `apps/logs/noise_api.log` |
| Guidance / Oracle UI | 2830 | `apps/logs/guidance_agent.log` |

---

## Configuration

`apps/.env` is created automatically by `setup.sh`. The default `LM_STUDIO_API_BASE` already points to the shared office inference server:

```bash
# apps/.env
LM_STUDIO_API_BASE=http://20.10.10.152:11434/v1
```

Do **not** change this unless the inference server moves. All other values have correct defaults.

### Using a local model instead

If you want to run models locally (no VPN needed), install [Ollama](https://ollama.com) and pull the models:

```bash
# Install Ollama
brew install ollama          # macOS
# curl -fsSL https://ollama.com/install.sh | sh   # Linux

# Pull the required models (~20 GB total)
ollama pull qwen3:30b-a3b-q8_0      # Main LLM — requires ~32 GB RAM
ollama pull qwen2.5vl:7b-q8_0       # Vision LLM (BORG agent)

# Start Ollama
ollama serve
```

Then update `apps/.env` to point to localhost:

```bash
LM_STUDIO_API_BASE=http://localhost:11434/v1
```

The `MCP_API_TOKEN` is **auto-generated** on first `start.sh` run — you do not need to set it manually.

---

## Stopping the Platform

```bash
bash local-test/stop.sh
```

> This stops the **local** platform only. It has no effect on the deployed VM.

---

## Troubleshooting

**"Virtual environment not found"**
→ Run `bash local-test/setup.sh` first.

**"LLM server is not reachable"**
→ Check that you are on the office network or connected via VPN. The inference server is at `20.10.10.152`.

**Agent starts but tools fail with 401 Unauthorized**
→ `MCP_API_TOKEN` mismatch. Delete `apps/.env`, restart — a fresh token will be generated.

**Port already in use**
→ `start_platform.sh` kills stale processes automatically. If it still fails, run `apps/stop_platform.sh` first then restart.

**LLM timeouts or slow responses**
→ The shared server may be under load. Retry after a moment. If the problem persists, verify VPN connectivity.
