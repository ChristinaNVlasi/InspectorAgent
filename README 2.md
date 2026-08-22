# SMA Inspector — AI-Powered Industrial Inspection Platform

> Multi-agent AI system for automated industrial component inspection

---

## Overview

SMA Inspector is a production-deployed, multi-agent AI platform that routes inspection requests to specialist AI agents depending on the component type. A single web UI serves all three inspection use-cases; an SM Inspector agent handles natural-language routing; specialist agents execute tool calls against a shared MCP server that hosts all inference models.

### Use-cases

| Agent | Component | Technology | Port |
|-------|-----------|------------|------|
| **Knorr PCB Inspector** | EBS7 TCM circuit boards | Two-stage RAG · CLIP embeddings | 2829 |
| **Arcelik-Beko WM Inspector** | Washing machine noise & visual damage | ML audio classifier · CLIP RAG | 2828 |
| **BORG Alternator Inspector** | Automotive alternator remanufacturing | Frame extraction · VLM verification | 2827 |
| **SMA Inspector (Guidance Agent)** | Routing & conversation | Qwen3-30B · Google ADK | 2830 |
| **Global MCP Server** | All tool endpoints | FastMCP · Starlette · Bearer auth | 3030 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Browser  (guidance_ui.html)                    |
│             Single-page UI & Mobile-ready                   |
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP / REST
        ┌──────────────▼──────────────┐
        │  SMA Inspector/Guidance     │  port 2830
        │  Qwen3-30B · Google ADK     │
        │  keyword fast-path routing  │
        └──┬──────────┬──────────┬─── ┘
           │          │          │
    ┌──────▼───┐ ┌────▼────┐ ┌──▼──────────┐
    │  Knorr   │ │Arcelik  │ │    BORG     │
    │PCB Agent │ │WM Agent │ │ Alt. Agent  │
    │  :2829   │ │  :2828  │ │   :2827     │
    └──────┬───┘ └────┬────┘ └──┬──────────┘
           └──────────┴──────────┘
                       │ Bearer token auth
        ┌──────────────▼──────────────┐
        │     Global MCP Server        │  port 3030
        │  9 tools · /tools/schemas   │
        │  GuidedPCBInspector          │
        │  NoiseClassifier             │
        │  RAGComponentInspector       │
        │  AlternatorInspector         │
        └─────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Ollama LLM  (separate VM)  │
        │  qwen3:30b-a3b-q8_0         │
        │  qwen2.5vl:7b-q8_0 (vision) │
        └─────────────────────────────┘
```

---

## Repository Structure

```
sma-inspector-agent/
├── apps/
│   ├── agents/                     # Oracle guidance agent + unified UI
│   │   ├── guidance_agent.py       # FastAPI · Google ADK · LiteLLM
│   │   ├── guidance_ui.html        # Full-stack single-page UI
│   │   ├── requirements.txt
│   │   └── start_guidance.sh
│   ├── agent_knorr/                # Knorr PCB inspection agent
│   │   ├── adk_agent.py
│   │   └── requirements.txt
│   ├── agent_arcelik/              # Arcelik-Beko WM inspection agent
│   │   ├── adk_agent.py
│   │   ├── requirements.txt
│   │   └── .env.example
│   ├── wm_models/                  # WM model libraries (imported by MCP server)
│   │   ├── noise/                  # NoiseClassifier + trained model (.pkl) + audio samples
│   │   └── vision/                 # CLIP RAG vision inspector
│   │       ├── ai_vision/          # Python package (embeddings, rag, models, utils)
│   │       └── parts_images/       # Labeled WM component images (RAG reference DB)
│   ├── agent_borg/                 # BORG alternator inspection agent
│   │   └── adk_agent.py
│   ├── mcp_server/                 # Unified MCP tool server
│   │   ├── global_mcp_server.py    # All 9 tools · auth · /status · /tools/schemas
│   │   └── start_global_mcp.sh
│   ├── pcb_defect_detector/        # PCB model library
│   │   ├── models/
│   │   │   ├── guided_inspector.py       # Two-stage RAG inspector
│   │   │   └── corrosion_cv_detector.py  # CV-based corrosion detection
│   │   ├── embeddings/                   # CLIP embedder
│   │   ├── rag/                          # ChromaDB vector store
│   │   ├── utils/                        # Grid mapper (4×4 A1-D4)
│   │   ├── vector_db/                    # Pre-built first_vb + second_vb databases
│   │   └── data/                         # Training samples (burned / corrosion / OK)
│   ├── PCB/                        # Whole-board classification images (burned / corrosion)
│   ├── start_platform.sh           # Start all services
│   └── stop_platform.sh            # Stop all services
```

---

## Key Features

### MCP Server (`port 3030`)
- **9 namespaced tools**: `pcb_*`, `wm_*`, `alternator_*`
- **Bearer token auth** on all `/tools/*` endpoints (`MCP_API_TOKEN` env var)
- **`GET /tools/schemas`** — public JSON schema for all tools (no auth required)
- **`GET /status`** — live system dashboard with service health, response times, tool catalog

### PCB Inspection — Two-Stage RAG
1. **Stage 1** (`first_vb`): Whole-board CLIP embedding → burned / corrosion / OK classification
2. **Stage 2** (`second_vb`): Cell-level grid localization → exact A1–D4 board position
- Corrosion CV detector as fallback/confirmation layer
- Returns annotated visualization + grid overlay + zoomed defect crop

### WM Inspection
- **Noise diagnosis**: 7-class ML classifier (bearing, pump, motor, springs, counterweight, shock absorber, foot leveling)
- **Visual damage**: CLIP RAG against labeled component images (cabinet, dispenser, front wall, surface)

### BORG Alternator Inspection
- Accepts video (`mp4`/`mov`) or stitched frame images
- VLM verification that upload actually shows an alternator (`qwen2.5vl:7b`)
- Component analysis: pulley, plastic end cover, housing/casting
- Produces cost deduction table (damaged pulley €10, cover €9, casting 50% of deposit)

### UI Highlights
- Dark gold industrial aesthetic · mobile/tablet responsive (`100dvh`)
- **Execution Flow trace** on every response — expandable, shows all agent reasoning steps
- **⟳ New Chat** button on every panel — resets session without page reload
- Real-time typing indicators · formatted markdown responses · confidence bar result cards

---

## Running the Platform

### Prerequisites
- Python 3.10+ with a virtual environment activated
- Ollama running `qwen3:30b-a3b-q8_0` (LLM) and `qwen2.5vl:7b-q8_0` (vision)
- Copy `.env.example` → `.env` in each agent directory and fill in values

### Environment Variables

```bash
# Required by all agents and the MCP server
MCP_API_TOKEN=<your-secure-token>

# LLM backend
LM_STUDIO_BASE=http://<ollama-host>:11434/v1
LM_STUDIO_MODEL=openai/qwen3:30b-a3b-q8_0
LM_STUDIO_API_KEY=ollama

# Agent base URLs (defaults work on a single host)
MCP_SERVER_URL=http://localhost:3030
KNORR_AGENT_URL=http://localhost:2829
ARCELIK_AGENT_URL=http://localhost:2828
BORG_AGENT_URL=http://localhost:2827
```

### Start All Services

```bash
cd apps
./start_platform.sh
```

Or start individually:

```bash
python mcp_server/global_mcp_server.py     # port 3030
python agents/guidance_agent.py            # port 2830
python agent_knorr/adk_agent.py            # port 2829
python agent_arcelik/adk_agent.py          # port 2828
python agent_borg/adk_agent.py             # port 2827
```

### Access Points

| URL | Description |
|-----|-------------|
| `http://localhost:2830` | Main inspection UI |
| `http://localhost:3030/status` | Live system status dashboard |
| `http://localhost:3030/tools/schemas` | MCP tool API reference (public) |
| `http://localhost:3030/health` | MCP health check JSON |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Qwen3-30B-A3B (q8_0) via Ollama |
| Vision LLM | Qwen2.5-VL 7B (q8_0) via Ollama |
| Agent Framework | Google ADK · LiteLLM |
| MCP Protocol | FastMCP · SSE transport (Starlette) |
| PCB Embeddings | OpenAI CLIP (ViT-B/32) · ChromaDB |
| WM Vision | CLIP RAG · ChromaDB vector store |
| WM Audio | scikit-learn classifier · MFCC features |
| API Layer | FastAPI · uvicorn |
| Frontend | Vanilla JS — single HTML file, no build step |
| Auth | Static Bearer token via env var |

---

## API Quick Reference

### Inspect a PCB (authenticated)
```bash
curl -X POST http://localhost:3030/tools/pcb_inspect \
  -H "Authorization: Bearer $MCP_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"image_file_path": "/path/to/board.jpg", "board_type": "EBS7_TCM"}'
```

### Get all tool schemas (public)
```bash
curl http://localhost:3030/tools/schemas
```

### Chat with SMA Inpsector
```bash
curl -X POST http://localhost:2830/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need to inspect a PCB board", "session_id": "demo-1"}'
```

---

*Built by CVL with ♥️*
