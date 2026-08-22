"""
Guidance / Selector Agent -- SMA Inspector  (v3.0 -- LLM-powered with memory)
Understands user intent via LLM conversation and routes to the correct specialist.

SMA Inspector is a Google ADK agent backed by LiteLLM.  It maintains full
per-session conversation memory (no regex, no keyword shortcuts).

Endpoints:
  GET    /                      -> Serve the SMA Inspector UI (guidance_ui.html)
  POST   /chat                  -> Chat with SMA Inspector (LLM + memory)
  GET    /health                -> Health check
  GET    /status                -> System-wide status
  DELETE /session/{session_id}  -> Clear a session history

Port: 2830
"""
import os
import logging
from pathlib import Path
from typing import Optional

import uvicorn
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Google ADK
from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

load_dotenv()

# ------------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
AGENT_PORT        = int(os.getenv("GUIDANCE_PORT", 2830))
AGENT_HOST        = os.getenv("GUIDANCE_HOST", "0.0.0.0")

KNORR_AGENT_URL   = os.getenv("KNORR_AGENT_URL",  "http://localhost:2829")
ARCELIK_AGENT_URL = os.getenv("ARCELIK_AGENT_URL", "http://localhost:2828")
BORG_AGENT_URL    = os.getenv("BORG_AGENT_URL",    "http://localhost:2827")
MCP_SERVER_URL    = os.getenv("MCP_SERVER_URL",    "http://localhost:3030")

LM_STUDIO_BASE    = os.getenv("LM_STUDIO_API_BASE", "http://20.10.10.152:11434/v1")
LM_STUDIO_MODEL   = os.getenv("LM_STUDIO_MODEL",    "openai/qwen3:30b-a3b-q8_0")
LM_STUDIO_KEY     = os.getenv("LM_STUDIO_API_KEY",  "ollama")

# ------------------------------------------------------------------------------
# SMA Inspector System Prompt
# ------------------------------------------------------------------------------
_SMA_INSPECTOR_INSTRUCTION = """
You are SMA Inspector -- the intelligent gateway for the SMA Industrial Inspection Platform.

## Your Mission
Engage users in warm, futuristic conversation, understand what they want to inspect,
and guide them to the right specialist AI system.

## Specialist Systems Available
1. Knorr PCB Inspection System -- detects defects on circuit boards (burned areas,
   corrosion, missing components, solder issues, cracks) using RAG-powered CLIP embeddings.
   Activate when the user mentions: PCB, circuit board, electronic board, Knorr, EBS7,
   solder, corrosion, burnt, missing component, electronics, board defect.

2. Arcelik-Beko Washing Machine Specialist -- diagnoses washing machines via audio/image
   analysis (noise, vibration, bearing wear, pump faults, drum issues).
   Activate when the user mentions: washing machine, washer, Arcelik, Beko, noise,
   vibration, drum, bearing, motor, pump, spin, squeaking, rattling, leaking.

3. BORG Automotive Alternator Inspection -- inspects returned alternators via video for
   remanufacturing acceptance. Analyses pulley, cover (plastic end cap), and housing/casting
   for damage. Applies cost deductions: damaged pulley €10, damaged cover €9, damaged casting
   = 50% deposit reduction.
   Activate when the user mentions: alternator, BORG, pulley, cover, housing, casting,
   remanufacturing, alternator inspection, generator, car part, refund, deposit.

## Routing Signal
When you decide to route the user, include ONE of these exact tokens anywhere in your reply:
  [ROUTE:pcb]              -- to activate the Knorr PCB system
  [ROUTE:washing_machine]  -- to activate the Arcelik-Beko system
  [ROUTE:borg]             -- to activate the BORG Alternator system

If it is not yet clear which system the user needs, do NOT include a route token.
Ask a clarifying question instead.

## Personality & Style
- Sophisticated, warm, and futuristic -- the user should feel they are talking to an
  advanced industrial AI.
- When routing, express genuine excitement about activating the specialist.
- Keep responses concise (2-4 sentences).
- Remember everything said earlier in this conversation and refer back to it when relevant.

## Memory
You have full memory of this conversation. Use it to avoid repeating yourself and to
refine your understanding as the user shares more information.

/no_think
"""


# ------------------------------------------------------------------------------
# GuidanceAgent -- ADK-powered SMA Inspector with per-session memory
# ------------------------------------------------------------------------------

class GuidanceAgent:
    """
    LLM-backed SMA Inspector agent using Google ADK + LiteLLM.
    Maintains per-session conversation history; no regex/keyword routing shortcuts.
    """

    def __init__(self):
        logger.info("Initialising SMA Inspector Guidance Agent (ADK + LiteLLM)")
        logger.info(f"  LLM endpoint : {LM_STUDIO_BASE}")
        logger.info(f"  Model        : {LM_STUDIO_MODEL}")

        # Per-session conversation log (for context injection into each turn)
        self.conversation_history: dict = {}

        # ADK session service + runner (no MCP tools -- pure conversation)
        self.session_service = InMemorySessionService()
        self.agent = Agent(
            name="sma_inspector_guidance",
            model=LiteLlm(
                model=LM_STUDIO_MODEL,
                api_key=LM_STUDIO_KEY,
                api_base=LM_STUDIO_BASE,
            ),
            description="SMA Inspector -- SMA Industrial Inspection gateway",
            instruction=_SMA_INSPECTOR_INSTRUCTION,
        )
        self.runner = Runner(
            agent=self.agent,
            app_name="sma_inspector_app",
            session_service=self.session_service,
        )
        logger.info("SMA Inspector agent ready")

    # ---- helpers -------------------------------------------------------------

    def _extract_intent(self, text: str) -> str:
        """Read the routing token the LLM embeds in its reply."""
        if "[ROUTE:pcb]" in text:
            return "pcb"
        if "[ROUTE:washing_machine]" in text:
            return "washing_machine"
        if "[ROUTE:borg]" in text:
            return "borg"
        return "unknown"

    def _strip_route_token(self, text: str) -> str:
        """Remove internal routing tokens and think blocks before sending text to the user."""
        import re
        # Strip <think>...</think> blocks (Qwen3 thinking mode)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Strip plain-text internal monologue: loop-remove consecutive reasoning paragraphs
        _THINKING_START = re.compile(
            r'^(Okay[,. ]|Alright[,. ]|Let me\b|I need to\b|The user\b|So the user\b|'
            r'Looking at\b|I should\b|Since they\b|Wait,|Hmm,|Actually, I|But I\b|'
            r'Now, I|Alternatively,|But the user\b)',
            re.IGNORECASE,
        )
        paras = re.split(r'\n\n+', text)
        while paras and _THINKING_START.match(paras[0].strip()):
            paras.pop(0)
        text = '\n\n'.join(paras)
        return (
            text
            .replace("[ROUTE:pcb]", "")
            .replace("[ROUTE:washing_machine]", "")
            .replace("[ROUTE:borg]", "")
            .strip()
        )

    def _build_enriched_message(self, session_id: str, user_message: str) -> str:
        """
        Prepend recent conversation history so the LLM always has context,
        even across ADK runner resets.
        """
        history = self.conversation_history.get(session_id, [])
        recent = history[-10:]  # last 5 exchanges
        if not recent:
            return user_message
        lines = ["[Conversation so far:]"]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "SMA Inspector"
            lines.append(f"{role}: {msg['content'][:300]}")
        lines.append("[End of history]\n\nCurrent message: " + user_message)
        return "\n".join(lines)

    # ---- keyword fast-path (before LLM) ----------------------------------------

    _PCB_KEYWORDS = {
        "pcb", "circuit board", "circuit", "board", "knorr", "ebs7", "ebs", "tcm",
        "solder", "corrosion", "burnt", "burned", "missing component", "electronics",
        "board defect", "pcb inspection", "printed circuit",
    }
    _WM_KEYWORDS = {
        "washing machine", "washer", "arcelik", "beko", "noise", "vibration",
        "bearing", "motor", "pump", "drum", "spin", "squeaking", "rattling",
        "leaking", "washing", "laundry", "machine noise",
    }
    _BORG_KEYWORDS = {
        "alternator", "borg", "pulley", "cover", "housing", "casting",
        "remanufacturing", "generator", "car part", "refund", "deposit",
        "alternator inspection",
    }

    def _keyword_intent(self, text: str) -> str:
        """Return a hard-wired intent if any keyword matches, else empty string."""
        t = text.lower()
        if any(k in t for k in self._PCB_KEYWORDS):
            return "pcb"
        if any(k in t for k in self._WM_KEYWORDS):
            return "washing_machine"
        if any(k in t for k in self._BORG_KEYWORDS):
            return "borg"
        return ""

    # ---- main entry point ----------------------------------------------------

    async def process(self, user_message: str, session_id: str) -> dict:
        """
        Run the user message through the SMA Inspector LLM.
        Returns {"response": str, "intent": str}.
        """
        enriched = self._build_enriched_message(session_id, user_message)

        # Ensure ADK session exists
        try:
            await self.session_service.create_session(
                app_name="sma_inspector_app",
                user_id="sma_inspector_user",
                session_id=session_id,
            )
        except Exception:
            pass  # session already exists

        content = types.Content(
            role="user",
            parts=[types.Part(text=enriched)],
        )

        full_response = ""
        try:
            async for event in self.runner.run_async(
                user_id="sma_inspector_user",
                session_id=session_id,
                new_message=content,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            full_response += part.text
                if event.is_final_response():
                    break
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            full_response = (
                "I'm momentarily unable to reach my reasoning engine. "
                "Please tell me: are you looking to inspect a PCB circuit board "
                "or diagnose a washing machine?"
            )

        full_response = full_response.strip()

        # Extract routing intent and clean the response text
        intent         = self._extract_intent(full_response)
        clean_response = self._strip_route_token(full_response)

        # Persist to per-session conversation history
        history = self.conversation_history.setdefault(session_id, [])
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": clean_response})
        if len(history) > 30:
            history[:] = history[-30:]

        logger.info(f"[{session_id}] intent={intent} | {user_message[:60]}")

        _ROUTE_LABELS = {
            "pcb":             "→ Knorr PCB Inspection Agent (port 2829)",
            "washing_machine": "→ Arcelik-Beko Washing Machine Agent (port 2828)",
            "borg":            "→ BORG Alternator Inspection Agent (port 2827)",
        }
        trace = [
            {"step": 1, "type": "user_input",     "content": user_message},
            {"step": 2, "type": "llm_reasoning",  "content": f"LLM ({LM_STUDIO_MODEL}) processed message and determined intent: '{intent}'"},
            {"step": 3, "type": "routing",        "content": _ROUTE_LABELS.get(intent, "→ Handled directly by Oracle (no specialist needed)")},
        ]
        return {"response": clean_response, "intent": intent, "trace": trace}

    def clear_session(self, session_id: str):
        self.conversation_history.pop(session_id, None)


# ------------------------------------------------------------------------------
# Singleton agent instance (created at module load)
# ------------------------------------------------------------------------------
_sma_inspector = GuidanceAgent()

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(title="SMA Guidance Agent", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_FILE = Path(__file__).parent / "guidance_ui.html"


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response:     str
    intent:       str            # "pcb" | "washing_machine" | "unknown"
    redirect_url: Optional[str] = None
    agent_url:    Optional[str] = None
    session_id:   str
    trace:        Optional[list] = None


@app.get("/")
async def root():
    if UI_FILE.exists():
        return FileResponse(UI_FILE, media_type="text/html")

@app.get("/logo")
async def logo():
    logo_file = Path(__file__).parent / "Logo_CORE-IC-White.png"
    if not logo_file.exists():
        # Try workspace root
        logo_file = Path(__file__).parent.parent.parent / "apps" / "Logo_CORE-IC-White.png"
    if not logo_file.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(str(logo_file), media_type="image/png")
    raise HTTPException(status_code=404, detail="UI file not found")


@app.get("/health")
async def health():
    return {
        "status":        "healthy",
        "service":       "SMA Inspector Guidance Agent",
        "version":       "3.0.0",
        "llm_backend":   "ADK + LiteLLM",
        "port":          AGENT_PORT,
        "knorr_agent":   KNORR_AGENT_URL,
        "arcelik_agent": ARCELIK_AGENT_URL,
        "borg_agent":    BORG_AGENT_URL,
        "mcp_server":    MCP_SERVER_URL,
    }


@app.get("/status")
async def system_status():
    """Probe all system components."""
    results = {}
    checks = [
        ("mcp_server",    f"{MCP_SERVER_URL}/health"),
        ("knorr_agent",   f"{KNORR_AGENT_URL}/health"),
        ("arcelik_agent", f"{ARCELIK_AGENT_URL}/health"),
        ("borg_agent",    f"{BORG_AGENT_URL}/health"),
    ]
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in checks:
            try:
                r = await client.get(url)
                results[name] = {"status": "online", "code": r.status_code}
            except Exception as e:
                results[name] = {"status": "offline", "error": str(e)}
    return results


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a message through the SMA Inspector LLM and return routing information."""
    session_id = req.session_id or "default"

    result   = await _sma_inspector.process(req.message, session_id)
    intent   = result["intent"]
    response = result["response"]
    trace    = result.get("trace")

    agent_url = None
    if intent == "pcb":
        agent_url = KNORR_AGENT_URL
    elif intent == "washing_machine":
        agent_url = ARCELIK_AGENT_URL
    elif intent == "borg":
        agent_url = BORG_AGENT_URL

    return ChatResponse(
        response=response,
        intent=intent,
        redirect_url=None,
        agent_url=agent_url,
        session_id=session_id,
        trace=trace,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _sma_inspector.clear_session(session_id)
    return {"cleared": session_id}


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SMA Inspector / Guidance Agent  (v3.0 -- LLM + memory)")
    logger.info(f"  Port          : {AGENT_PORT}")
    logger.info(f"  LLM           : {LM_STUDIO_MODEL}")
    logger.info(f"  Knorr agent   : {KNORR_AGENT_URL}")
    logger.info(f"  Arcelik agent : {ARCELIK_AGENT_URL}")
    logger.info(f"  MCP server    : {MCP_SERVER_URL}")
    logger.info("=" * 60)
    uvicorn.run(app, host=AGENT_HOST, port=AGENT_PORT, log_level="info")