"""
BORG Alternator Inspection Agent  (v1.0)
Inspects alternator units via video for remanufacturing acceptance.
Agent runs on port 2827.
"""
import os
import re
import sys
import logging
import tempfile
import base64
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.mcp_tool import MCPToolset, SseConnectionParams

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API token for MCP authentication — read from environment, never logged or hardcoded
_MCP_TOKEN = os.environ.get("MCP_API_TOKEN", "")
if not _MCP_TOKEN:
    logging.getLogger(__name__).critical(
        "FATAL: MCP_API_TOKEN environment variable is not set. "
        "Set it before starting: export MCP_API_TOKEN=<token>."
    )
    sys.exit(1)
_MCP_AUTH_HEADERS = {"Authorization": f"Bearer {_MCP_TOKEN}"}

# ── Cost table ────────────────────────────────────────────────────────────────
DEDUCTION_TABLE = {
    "pulley":  {"label": "Damaged pulley",         "deduction": 10.00,  "is_percentage": False},
    "cover":   {"label": "Damaged cover",           "deduction": 9.00,   "is_percentage": False},
    "casting": {"label": "Damaged casting (cracks/fractures)", "deduction": 50.0, "is_percentage": True},
}
# Base deposit for percentage calculations
BASE_DEPOSIT = float(os.getenv("DEPOSIT_AMOUNT", 100.00))
MIN_VIDEO_DURATION = float(os.getenv("MIN_VIDEO_DURATION_SEC", 2.0))


class BorgAlternatorAgent:
    """ADK-powered BORG alternator inspection agent."""

    SYSTEM_PROMPT = f"""You are BORG Inspector — a specialist AI for alternator remanufacturing assessment at BORG Automotive.

## Your Role
You receive alternator inspection videos and provide a professional remanufacturing assessment.
You report on three components: pulley, cover (plastic end cover), and housing/casting.

## Inspection Results Format
When an inspection is complete, present results clearly:
- State overall acceptance: "Alternator is accepted for remanufacturing"
- List component status for: Pulley | Cover | Housing/Casting
- Use clear indicators: ✅ OK / ⚠️ Damaged

## Cost Deductions (when asked for estimate)
Apply these deductions from the deposit:
- Damaged pulley          = €10.00 deduction
- Damaged cover           = €9.00 deduction
- Damaged casting (cracks/fractures) = 50% reduction of deposit refund
- Component OK            = no deduction

## Routing & Tools
You have access to the `alternator_inspect` MCP tool. Use it when a video file path is provided.

## Style
- Professional, precise, and clear.
- When presenting results always show a structured component table.
- Keep non-inspection conversation concise (2-3 sentences).
- When asked for cost estimate, itemise each deduction clearly.
- Remember the current session's inspection results and refer to them.
/no_think"""

    def __init__(self, mcp_server_url: str = "http://localhost:3030"):
        self.mcp_server_url = mcp_server_url
        self.lm_studio_api_base = os.getenv("LM_STUDIO_API_BASE", "http://20.10.10.152:11434/v1")
        self.lm_studio_model    = os.getenv("LM_STUDIO_MODEL",    "openai/qwen3:30b-a3b-q8_0")
        self.api_key            = os.getenv("LM_STUDIO_API_KEY",  "ollama")

        logger.info("🔧 Initialising BORG Alternator Inspection Agent")
        logger.info(f"📡 MCP Server : {self.mcp_server_url}")
        logger.info(f"🧠 LLM        : {self.lm_studio_model}")

        self.mcp_tools = self._init_mcp_tools()
        self.conversation_history: dict = {}
        self.session_service = InMemorySessionService()
        self.agent  = self._create_agent()
        self.runner = Runner(
            agent=self.agent,
            app_name="borg_inspection_app",
            session_service=self.session_service,
        )
        logger.info("✅ BORG Agent initialised")

    def _init_mcp_tools(self):
        try:
            sse_url = f"{self.mcp_server_url}/sse"
            mcp_toolset = MCPToolset(connection_params=SseConnectionParams(url=sse_url, headers=_MCP_AUTH_HEADERS))
            logger.info(f"✅ Connected to MCP server at {sse_url}")
            return mcp_toolset
        except Exception as e:
            logger.error(f"❌ MCP connection failed: {e}")
            raise

    def _create_agent(self):
        return Agent(
            name="borg_alternator_inspector",
            model=LiteLlm(
                model=self.lm_studio_model,
                api_key=self.api_key,
                api_base=self.lm_studio_api_base,
            ),
            description="BORG alternator inspection agent for remanufacturing assessment",
            instruction=self.SYSTEM_PROMPT,
            tools=[self.mcp_tools],
        )

    def _build_enriched_message(self, session_id: str, user_message: str) -> str:
        history = self.conversation_history.get(session_id, [])
        recent  = history[-10:]
        if not recent:
            return user_message
        lines = ["[Conversation so far:]"]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "BORG Inspector"
            lines.append(f"{role}: {msg['content'][:300]}")
        lines.append("[End of history]\n\nCurrent message: " + user_message)
        return "\n".join(lines)

    async def process(self, user_message: str, session_id: str, file_path: str = None) -> dict:
        # Inject file path into message if provided
        if file_path:
            user_message = (
                f"{user_message}\n\n[VIDEO FILE PATH: {file_path}]"
                "\nPlease run the alternator_inspect tool on this video file."
            )

        enriched = self._build_enriched_message(session_id, user_message)

        try:
            await self.session_service.create_session(
                app_name="borg_inspection_app",
                user_id="borg_user",
                session_id=session_id,
            )
        except Exception:
            pass

        content = types.Content(
            role="user",
            parts=[types.Part(text=enriched)],
        )

        full_response = ""
        tool_result   = None

        try:
            async for event in self.runner.run_async(
                user_id="borg_user",
                session_id=session_id,
                new_message=content,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            full_response += part.text
                        # Capture tool call results
                        if hasattr(part, "function_response") and part.function_response:
                            fn = part.function_response
                            if fn.name == "alternator_inspect":
                                tool_result = fn.response
                if event.is_final_response():
                    break
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            full_response = "I encountered an issue processing your request. Please try again."

        # Strip <think> blocks and plain-text reasoning monologue
        full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        _THINKING_START = re.compile(
            r'^(Okay[,. ]|Alright[,. ]|Let me\b|I need to\b|The user\b|So the user\b|'
            r'Looking at\b|I should\b|Since they\b|Wait,|Hmm,|Actually, I|But I\b|'
            r'Now, I|Alternatively,|But the user\b)',
            re.IGNORECASE,
        )
        paras = re.split(r'\n\n+', full_response)
        while paras and _THINKING_START.match(paras[0].strip()):
            paras.pop(0)
        full_response = '\n\n'.join(paras).strip()

        history = self.conversation_history.setdefault(session_id, [])
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > 30:
            history[:] = history[-30:]

        return {"response": full_response, "tool_result": tool_result}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_video_duration(path: str) -> Optional[float]:
    """Return video duration in seconds using cv2, falling back to moviepy."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and frames and frames > 0:
            return frames / fps
    except Exception:
        pass
    try:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(path) as clip:
            return clip.duration
    except Exception:
        pass
    return None


# ── FastAPI server ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import shutil
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    agent_port     = int(os.getenv("AGENT_PORT", 2827))
    agent_host     = os.getenv("AGENT_HOST", "0.0.0.0")
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:3030")

    logger.info("=" * 60)
    logger.info("BORG Alternator Inspection Agent  (v1.0)")
    logger.info(f"  Port       : {agent_port}")
    logger.info(f"  MCP Server : {mcp_server_url}")
    logger.info("=" * 60)

    agent = BorgAlternatorAgent(mcp_server_url=mcp_server_url)

    # Temp upload dir
    uploads_dir = Path(tempfile.gettempdir()) / "sma_inspector_borg"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="BORG Alternator Inspection Agent", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        message:    str
        session_id: Optional[str] = None
        file_path:  Optional[str] = None

    class ChatResponse(BaseModel):
        response:    str
        session_id:  Optional[str] = None
        tool_result: Optional[dict] = None
        trace:       Optional[list] = None

    @app.get("/")
    async def root():
        return {"service": "BORG Alternator Inspection Agent", "status": "running", "port": agent_port}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent_port": agent_port, "mcp_server": mcp_server_url}

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Accept video or image file, validate, return path."""
        from datetime import datetime
        try:
            suffix = Path(file.filename).suffix.lower()
            IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
            VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
            if suffix not in IMAGE_EXTS | VIDEO_EXTS:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a video or image.")

            # Images: save and return immediately (no duration check needed)
            if suffix in IMAGE_EXTS:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                safe_name = f"borg_{timestamp}{suffix}"
                dest      = uploads_dir / safe_name
                with open(dest, "wb") as buf:
                    shutil.copyfileobj(file.file, buf)
                abs_path = str(dest.resolve())
                logger.info(f"✅ Image uploaded: {abs_path}")
                return {"success": True, "file_path": abs_path, "filename": safe_name, "duration": None}


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            safe_name = f"borg_{timestamp}{suffix}"
            dest      = uploads_dir / safe_name

            with open(dest, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            # Check video duration
            abs_path = str(dest.resolve())
            duration = _get_video_duration(abs_path)

            if duration is not None and duration < MIN_VIDEO_DURATION:
                dest.unlink(missing_ok=True)
                return {
                    "success":  False,
                    "too_short": True,
                    "duration": round(duration, 2),
                    "message":  f"Video is too short ({duration:.1f}s). Please provide a video of at least {MIN_VIDEO_DURATION:.0f} seconds.",
                }

            logger.info(f"✅ Video uploaded: {abs_path} ({duration:.1f}s)" if duration else f"✅ Video uploaded: {abs_path}")
            return {
                "success":   True,
                "file_path": abs_path,
                "filename":  safe_name,
                "duration":  round(duration, 2) if duration else None,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        import httpx
        session_id = req.session_id or "borg_default"

        # ── Fast path: video file provided → call MCP directly, skip LLM ──────
        if req.file_path:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(
                        f"{mcp_server_url}/tools/alternator_inspect",
                        json={"video_file_path": req.file_path, "deposit_amount": 50.0},
                        headers=_MCP_AUTH_HEADERS,
                    )
                tool_result = r.json()
            except Exception as e:
                logger.error(f"Direct MCP call failed: {e}")
                tool_result = {"success": False, "error": str(e)}
            finally:
                # Delete file immediately — never store user data
                try:
                    Path(req.file_path).unlink(missing_ok=True)
                    logger.info(f"🗑️ Deleted upload: {req.file_path}")
                except Exception:
                    pass

            # Build a quick text summary without touching the LLM
            if tool_result.get("success"):
                comps = tool_result.get("components", {})
                lines = ["**Alternator inspection complete.**\n"]
                for k, c in comps.items():
                    name   = c.get("label", k.capitalize())
                    status = c.get("status", "unknown")
                    if status == "ok":
                        lines.append(f"✅ {name} is OK")
                    elif status == "damaged":
                        lines.append(f"⚠️ {name} is damaged")
                    else:
                        lines.append(f"❓ {name} — unknown")
                deductions = tool_result.get("deductions", [])
                if deductions:
                    lines.append("\n💰 **Cost deductions:**")
                    for d in deductions:
                        lines.append(f"  • {d['label']}: −€{d['deduction']:.2f}")
                    lines.append(f"\n**Total deduction:** −€{tool_result.get('total_deduction', 0):.2f}")
                    lines.append(f"**Estimated refund:** €{tool_result.get('estimated_refund', 0):.2f}")
                else:
                    lines.append("\n✅ No deductions — unit in good condition.")
                summary = "\n".join(lines)
            else:
                err = tool_result.get("error", "Unknown error")
                summary = f"⚠️ Inspection could not be completed: {err}"

            # Build execution trace
            if tool_result.get("success"):
                comps_summary = ", ".join(
                    f"{k}: {v.get('status','?')}" for k, v in tool_result.get("components", {}).items()
                )
                trace = [
                    {"step": 1, "type": "user_input",     "content": f"Alternator file received: {Path(req.file_path).name}"},
                    {"step": 2, "type": "agent_decision", "content": "File path provided — calling alternator_inspect tool directly (fast path, LLM skipped)"},
                    {"step": 3, "type": "tool_call",      "tool": "alternator_inspect", "content": f"alternator_inspect(video_file_path=..., deposit_amount=50.0)"},
                    {"step": 4, "type": "tool_result",    "content": f"accepted=True, {comps_summary}, deduction=€{tool_result.get('total_deduction',0):.2f}, refund=€{tool_result.get('estimated_refund',0):.2f}"},
                    {"step": 5, "type": "final_response", "content": "Component summary + cost breakdown returned to user"},
                ]
            else:
                trace = [
                    {"step": 1, "type": "user_input",  "content": "Alternator file received"},
                    {"step": 2, "type": "tool_call",   "tool": "alternator_inspect", "content": "alternator_inspect called"},
                    {"step": 3, "type": "tool_result", "content": f"error: {tool_result.get('error', 'unknown')}"},
                ]

            return ChatResponse(response=summary, session_id=session_id, tool_result=tool_result, trace=trace)

        # ── Normal path: text-only → go through LLM ──────────────────────────
        result = await agent.process(req.message, session_id)
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            tool_result=result.get("tool_result"),
        )

    uvicorn.run(app, host=agent_host, port=agent_port, log_level="info")
