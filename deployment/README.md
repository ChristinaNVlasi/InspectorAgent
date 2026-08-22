# Deployment Guide

**Target VM:** `cvlassi@10.104.100.173`  
**Project root on VM:** `~/sma/agentic/`  
**Python venv:** `~/sma/agentic/.venv`

---

## Prerequisites

```bash
# 1. Install system dependencies
sudo apt-get install -y python3.10 python3.10-venv python3-pip ffmpeg sshpass

# 2. Create project root and venv
mkdir -p ~/sma/agentic
cd ~/sma/agentic
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies (run once per agent dir)
pip install -r apps/mcp_server/../agents/requirements.txt
pip install -r apps/agent_knorr/requirements.txt
pip install -r apps/agent_arcelik/requirements.txt
pip install -r apps/wm_models/noise/requirements.txt
pip install -r apps/wm_models/vision/ai_vision/requirements.txt
pip install -r apps/pcb_defect_detector/requirements.txt
```

---

## Environment Setup

Create `apps/.env` on the VM (never commit this file):

```bash
cat > ~/sma/agentic/apps/.env << 'EOF'
MCP_API_TOKEN=
LM_STUDIO_API_BASE=http://20.10.10.152:11434/v1
LM_STUDIO_MODEL=openai/qwen3:30b-a3b-q8_0
LM_STUDIO_API_KEY=ollama
KNORR_AGENT_URL=http://localhost:2829
ARCELIK_AGENT_URL=http://localhost:2828
BORG_AGENT_URL=http://localhost:2827
MCP_SERVER_URL=http://localhost:3030
EOF
```

> Leave `MCP_API_TOKEN` blank — `start_platform.sh` auto-generates a cryptographically secure token on first run and saves it back to `apps/.env`.

---

## Deploy / Update Code

From your local machine, rsync the project to the VM:

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.venv/' --exclude='venv/' \
  --exclude='apps/logs/' --exclude='apps/*/uploads/' \
  --exclude='.env' \
  /path/to/sma-inspector-agent/ \
  cvlassi@10.104.100.173:~/sma/agentic/
```

---

## Start Platform

```bash
ssh cvlassi@10.104.100.173
cd ~/sma/agentic
source .venv/bin/activate
cd apps && ./start_platform.sh
```

Each service runs in the background. Logs are in `apps/logs/`.

---

## Services & Ports

| Service | Port | Log file |
|---------|------|----------|
| Global MCP Server | 3030 | `apps/logs/global_mcp.log` |
| Knorr PCB Agent | 2829 | `apps/logs/knorr_agent.log` |
| Arcelik WM Agent | 2828 | `apps/logs/arcelik_agent.log` |
| BORG Alternator Agent | 2827 | `apps/logs/borg_agent.log` |
| Noise & Vision API | 5001 | `apps/logs/noise_api.log` |
| Oracle / UI | 2830 | `apps/logs/guidance_agent.log` |

---

## Access

| URL | Description |
|-----|-------------|
| `http://10.104.100.173:2830` | Main inspection UI |
| `http://10.104.100.173:3030/status` | Live system dashboard |
| `http://10.104.100.173:3030/tools/schemas` | MCP tool schemas (public) |
| `http://10.104.100.173:3030/health` | Health check JSON |

---

## Stop Platform

```bash
cd ~/sma/agentic/apps && ./stop_platform.sh
```

---

## Useful Commands

```bash
# Check all services are up
curl http://localhost:3030/health
curl http://localhost:2829/health
curl http://localhost:2828/health
curl http://localhost:2827/health
curl http://localhost:2830/health

# Tail a specific log
tail -f apps/logs/global_mcp.log

# Test MCP auth (read token from .env first)
TOKEN=$(grep MCP_API_TOKEN apps/.env | cut -d= -f2)
curl -H "Authorization: Bearer $TOKEN" http://localhost:3030/tools/schemas
```
