"""
ADK Agent for Knorr PCB Inspection System
Uses LM Studio LLM and connects to MCP Server on port 3029
Agent runs on port 2829
"""
import os
import re
import sys
import logging
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Google ADK imports
from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.mcp_tool import MCPToolset, SseConnectionParams

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


class KnorrPCBInspectionAgent:
    """
    ADK Agent for Knorr PCB inspection.
    Orchestrates PCB defect detection, preprocessing, and grid localization via MCP tools.
    """
    
    def __init__(
        self,
        mcp_server_url: str = "http://localhost:3030",
        lm_studio_api_base: str = None,
        lm_studio_model: str = None,
        api_key: str = None
    ):
        """
        Initialize the inspection agent.
        
        Args:
            mcp_server_url: URL of the MCP inspection server
            lm_studio_api_base: LM Studio API endpoint
            lm_studio_model: Model name to use
            api_key: API key for LM Studio
        """
        self.mcp_server_url = mcp_server_url
        self.lm_studio_api_base = lm_studio_api_base or os.getenv('LM_STUDIO_API_BASE', 'http://20.10.10.152:11434/v1')
        self.lm_studio_model = lm_studio_model or os.getenv('LM_STUDIO_MODEL', 'ollama/qwen3:235b')
        self.api_key = api_key or os.getenv('LM_STUDIO_API_KEY', 'ollama')
        
        logger.info("🤖 Initializing Knorr PCB Inspection Agent")
        logger.info(f"📡 MCP Server: {self.mcp_server_url}")
        logger.info(f"🧠 LLM: {self.lm_studio_model}")
        
        # Initialize MCP tools
        self.mcp_tools = self._initialize_mcp_tools()
        
        # Conversation history storage - keyed by session_id
        # Each session stores: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.conversation_history = {}
        
        # Create session service
        self.session_service = InMemorySessionService()
        
        # Create the agent and runner
        self.agent = self._create_agent()
        self.runner = Runner(
            agent=self.agent,
            app_name="pcb_inspection_app",
            session_service=self.session_service
        )
        
        logger.info("✅ Agent initialized successfully")
    
    def _initialize_mcp_tools(self):
        """Initialize MCP tools from the inspection server"""
        try:
            # Connect to MCP server using SSE connection with Bearer token auth
            sse_url = f"{self.mcp_server_url}/sse"
            connection_params = SseConnectionParams(url=sse_url, headers=_MCP_AUTH_HEADERS)
            mcp_toolset = MCPToolset(connection_params=connection_params)
            logger.info(f"✅ Connected to MCP server at {sse_url}")
            return mcp_toolset
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {e}")
            logger.warning("⚠️ Make sure Global MCP server is running on port 3030")
            raise
    
    def _create_agent(self):
        """Create the ADK agent with tools"""
        instruction = """You are a PCB inspection AI assistant.

CAPABILITIES:
- Inspect PCB images for defects (burned, corrosion, missing parts, cracks, solder issues)
- Preprocess PCB images
- Localize defects using grid coordinates
- Get database statistics

TOOLS AVAILABLE (Global MCP Server port 3030):
- pcb_inspect(image_file_path, board_type)      — full defect detection
- pcb_preprocess(image_file_path, method)        — background removal & crop
- pcb_localize_defect(defect_type, grid_cell)    — grid coordinate lookup
- pcb_get_status()                               — model health check
- pcb_database_stats()                           — RAG database statistics

CRITICAL INSTRUCTIONS:
1. When user provides an image path, call pcb_inspect tool with that path.
2. After receiving the tool result, respond with a SHORT structured summary (2-4 lines max):
   - Status (OK / NOT_OK)
   - Defect type and location (if any)
   - Confidence percentage
   Do NOT describe or repeat raw JSON, base64 data, or field names.
3. For follow-up questions about a PREVIOUS inspection, DO NOT call tools again — answer from conversation history.
4. Only call tools for NEW images or NEW data requests.
5. Be concise. Never output more than 5 sentences.

/no_think"""

        try:
            agent = Agent(
                name="knorr_pcb_inspector",
                model=LiteLlm(
                    model=self.lm_studio_model,
                    api_key=self.api_key,
                    api_base=self.lm_studio_api_base,
                ),
                description="PCB defect inspection assistant",
                instruction=instruction,
                tools=[self.mcp_tools]
            )
            logger.info("✅ ADK agent created with MCP tools")
            return agent
        except Exception as e:
            logger.error(f"❌ Failed to create agent: {e}")
            raise
    
    async def process_request(self, user_message: str, session_id: Optional[str] = None):
        """
        Process a user request through the agent with conversation history.
        
        Args:
            user_message: User's message or question
            session_id: Optional session ID for conversation continuity
        
        Returns:
            Agent's response
        """
        try:
            logger.info(f"📝 Processing request: {user_message[:100]}...")
            
            # Use session_id or default
            if not session_id:
                session_id = "default_session"
            
            # Initialize conversation history for this session if needed
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # Get recent conversation history (last 6 messages = 3 exchanges)
            recent_history = self.conversation_history[session_id][-6:]
            
            # Build context-enriched message with history
            if recent_history:
                history_text = "\n\n[Previous conversation in this session:\n"
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = msg["content"][:200]  # Limit to 200 chars per message
                    history_text += f"{role}: {content}\n"
                history_text += "]\n\nCurrent question: "
                enriched_message = history_text + user_message
            else:
                enriched_message = user_message
            
            # Ensure session exists
            try:
                await self.session_service.create_session(
                    app_name="pcb_inspection_app",
                    user_id="default_user",
                    session_id=session_id
                )
            except:
                pass  # Session might already exist
            
            # Create content with enriched message
            content = types.Content(
                role='user',
                parts=[types.Part(text=enriched_message)]
            )
            
            # Run agent and collect response + tool calls
            full_response = ""
            tool_calls = []   # [{name, args}] — what tools the LLM decided to call

            async for event in self.runner.run_async(
                user_id="default_user",
                session_id=session_id,
                new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            full_response += part.text

                        # Capture function_call parts (LLM's tool invocation decisions)
                        if hasattr(part, 'function_call') and part.function_call:
                            try:
                                fc = part.function_call
                                tool_calls.append({
                                    'name': getattr(fc, 'name', 'unknown'),
                                    'args': dict(getattr(fc, 'args', {}) or {}),
                                })
                                logger.info(f"🔧 LLM called tool: {getattr(fc, 'name', 'unknown')}")
                            except Exception as e:
                                logger.warning(f"Could not capture function_call: {e}")

                if event.is_final_response():
                    break
            
            # Store in conversation history (original message, not enriched)
            self.conversation_history[session_id].append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
            })
            
            # Keep conversation history from growing too large (last 20 messages = 10 exchanges)
            if len(self.conversation_history[session_id]) > 20:
                self.conversation_history[session_id] = self.conversation_history[session_id][-20:]
            
            logger.info(f"✅ Request processed successfully (history: {len(self.conversation_history[session_id])} messages)")
            
            # Return both text and tool results
            _clean = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
            # Strip plain thinking paragraphs that Qwen3 emits outside <think> tags
            _THINKING_START = re.compile(
                r'^(Okay[,. ]|Alright[,. ]|Let me\b|I need to\b|The user\b|So the user\b|'
                r'Looking at\b|I should\b|Since they\b|Wait,|Hmm,|Actually[, ]|But I\b|'
                r'Now[, ]|Alternatively,|But the user\b|First[, ]|The provided|'
                r'Based on the|I remember|I can see|I notice|Let\'s|So I|'
                r'In this case|In summary|To summarize|For this)',
                re.IGNORECASE,
            )
            _paras = re.split(r'\n\n+', _clean)
            while _paras and _THINKING_START.match(_paras[0].strip()):
                _paras.pop(0)
            result = {"response": '\n\n'.join(_paras).strip()}

            # Pass tool_calls so /chat knows which tools the LLM invoked (and with what args)
            if tool_calls:
                result['tool_calls'] = tool_calls
                logger.info(f"✅ Tool calls captured: {[tc['name'] for tc in tool_calls]}")

            return result
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}", exc_info=True)
            return {"response": f"Error: {str(e)}"}
    
    def run_interactive(self):
        """Run the agent in interactive CLI mode"""
        logger.info("🎯 Starting interactive mode")
        logger.info("Type 'quit' or 'exit' to stop\n")
        
        session_id = "interactive_session"
        
        print("=" * 70)
        print("🔍 Knorr PCB Inspection Agent - Interactive Mode")
        print("=" * 70)
        print("\nI can help you inspect PCB boards for defects:")
        print("  • RAG-based PCB defect detection")
        print("  • Image preprocessing (AI-powered cropping)")
        print("  • Grid-based defect localization")
        print("  • Database statistics and queries")
        print("\nExamples:")
        print("  - 'Inspect PCB image at /path/to/pcb.jpg'")
        print("  - 'Preprocess the image /path/to/pcb.jpg'")
        print("  - 'Localize defect BURNED at grid B1'")
        print("  - 'Get database statistics'")
        print("  - 'What defects can you detect?'")
        print("\n" + "=" * 70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Process the request (synchronous wrapper for async)
                import asyncio
                result = asyncio.run(self.process_request(user_input, session_id))
                
                # Extract text response
                if isinstance(result, dict):
                    response_text = result.get('response', str(result))
                else:
                    response_text = str(result)
                
                print(f"\nAgent: {response_text}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    import sys
    
    # Get configuration from environment
    agent_port = int(os.getenv('AGENT_PORT', '2829'))
    agent_host = os.getenv('AGENT_HOST', '0.0.0.0')
    mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:3030')
    
    logger.info("=" * 70)
    logger.info("🚀 Knorr PCB Inspection Agent")
    logger.info("=" * 70)
    logger.info(f"Agent Port: {agent_port}")
    logger.info(f"Agent Host: {agent_host}")
    logger.info(f"MCP Server: {mcp_server_url}")
    logger.info("=" * 70)
    
    try:
        # Initialize the agent
        agent = KnorrPCBInspectionAgent(mcp_server_url=mcp_server_url)
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            # Run in interactive CLI mode
            agent.run_interactive()
        else:
            # Run as API server
            logger.info(f"🌐 Starting API server on {agent_host}:{agent_port}")
            logger.info("💡 Use --interactive flag for CLI mode")
            
            # Import FastAPI components
            from fastapi import FastAPI, HTTPException, UploadFile, File
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel
            import uvicorn
            import shutil
            import tempfile
            from datetime import datetime
            
            app = FastAPI(
                title="Knorr PCB Inspection Agent API",
                description="ADK Agent for PCB defect detection",
                version="1.0.0"
            )
            
            # Add CORS middleware to allow web interface
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Allow all origins for development
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Use OS temp dir so files are not persisted in the project
            uploads_dir = Path(tempfile.gettempdir()) / "sma_inspector"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            class ChatRequest(BaseModel):
                message: str
                session_id: Optional[str] = None
                image_path: Optional[str] = None
            
            class InspectionRequest(BaseModel):
                message: str
                session_id: Optional[str] = None
            
            class InspectionResponse(BaseModel):
                response: str
                session_id: Optional[str] = None
            
            @app.get("/")
            async def root():
                return {
                    "service": "Knorr PCB Inspection Agent",
                    "status": "running",
                    "version": "1.0.0",
                    "mcp_server": mcp_server_url,
                    "capabilities": [
                        "pcb_defect_detection",
                        "image_preprocessing",
                        "grid_localization",
                        "database_statistics"
                    ]
                }
            
            @app.get("/health")
            async def health():
                return {
                    "status": "healthy",
                    "agent_port": agent_port,
                    "mcp_server": mcp_server_url
                }
            
            # Note: Standalone UI removed. Use Oracle UI at http://localhost:2830
            
            @app.post("/clear_history")
            async def clear_history(request: dict):
                """Clear conversation history for a session"""
                try:
                    session_id = request.get("session_id", "default_session")
                    if session_id in agent.conversation_history:
                        del agent.conversation_history[session_id]
                        logger.info(f"🗑️ Cleared history for session: {session_id}")
                        return {"success": True, "message": f"History cleared for session {session_id}"}
                    else:
                        return {"success": True, "message": "No history found for this session"}
                except Exception as e:
                    logger.error(f"Error clearing history: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.post("/chat")
            async def chat(request: ChatRequest):
                """Chat endpoint - processes user messages with optional image path"""
                try:
                    import re
                    import httpx

                    # Build message with image path if provided
                    message = request.message
                    if request.image_path:
                        message = f"{message}\nImage path: {request.image_path}"

                    # Detect image path — used for fast-path and cleanup
                    path_match = re.search(r'(/[^\n]+?\.(jpg|jpeg|png|JPG|JPEG|PNG))', message)
                    file_path = path_match.group(1) if path_match else None

                    # ── PCB FAST-PATH ─────────────────────────────────────────
                    # When an image is present, call the MCP REST endpoint directly
                    # (no waiting for the LLM tool-call round-trip).
                    # The LLM still receives the structured result as context.
                    if file_path and Path(file_path).exists():
                        try:
                            mcp_url = mcp_server_url.replace("localhost", "127.0.0.1")
                            async with httpx.AsyncClient(timeout=60.0) as client:
                                img_resp = await client.post(
                                    f"{mcp_url}/tools/pcb_inspect",
                                    json={"image_file_path": file_path, "board_type": "EBS7_TCM"},
                                    headers=_MCP_AUTH_HEADERS,
                                )
                            if img_resp.status_code == 200:
                                tool_data = img_resp.json()
                                status      = tool_data.get("status", "UNKNOWN")
                                confidence  = tool_data.get("confidence", 0)
                                conf_pct    = int(confidence * 100) if confidence <= 1 else int(confidence)
                                defect_info = tool_data.get("defect_info")

                                if status == "OK":
                                    agent_text = f"✅ PCB inspection complete. Status: OK — No defects detected. The board is in good condition. Confidence: {conf_pct}%."
                                elif defect_info:
                                    defect_type = defect_info.get("defect_type", "unknown")
                                    location    = defect_info.get("location", "unknown")
                                    description = defect_info.get("description", "")
                                    agent_text = (
                                        f"⚠️ PCB inspection complete. Status: NOT_OK — {defect_type.upper()} defect detected"
                                        f" at location {location}. Confidence: {conf_pct}%."
                                        + (f" {description}" if description else "")
                                    )
                                else:
                                    agent_text = f"PCB inspection complete. Status: {status}. Confidence: {conf_pct}%."

                                # Store result in agent conversation history so LLM has context
                                session_id = request.session_id or "default_session"
                                if session_id not in agent.conversation_history:
                                    agent.conversation_history[session_id] = []
                                agent.conversation_history[session_id].append({"role": "user",      "content": message})
                                agent.conversation_history[session_id].append({"role": "assistant", "content": agent_text})

                                trace = [
                                    {"step": 1, "type": "user_input",    "content": f"PCB image received: {Path(file_path).name}"},
                                    {"step": 2, "type": "tool_call",     "tool": "pcb_inspect", "content": f"pcb_inspect(image_file_path={file_path!r}, board_type='EBS7_TCM') — direct call"},
                                    {"step": 3, "type": "tool_result",   "content": f"status={status}, confidence={conf_pct}%"},
                                    {"step": 4, "type": "final_response","content": "Structured result + images returned to user"},
                                ]

                                try:
                                    Path(file_path).unlink(missing_ok=True)
                                except Exception:
                                    pass

                                return JSONResponse({
                                    "response":    agent_text,
                                    "tool_result": tool_data,
                                    "trace":       trace,
                                    "session_id":  session_id,
                                })
                        except Exception as e:
                            logger.warning(f"PCB fast-path failed, falling back to LLM: {e}")
                    # ── END FAST-PATH ─────────────────────────────────────────

                    # No image (or fast-path failed) → LLM handles the message
                    result = await agent.process_request(
                        message,
                        request.session_id
                    )

                    # If the LLM still happened to call pcb_inspect, enrich response
                    tool_calls = result.get("tool_calls", []) if isinstance(result, dict) else []
                    pcb_call = next((tc for tc in tool_calls if tc["name"] == "pcb_inspect"), None)
                    if pcb_call:
                        file_path = pcb_call["args"].get("image_file_path") or file_path
                        if file_path and Path(file_path).exists():
                            try:
                                mcp_url = mcp_server_url.replace("localhost", "127.0.0.1")
                                async with httpx.AsyncClient(timeout=60.0) as client:
                                    img_resp = await client.post(
                                        f"{mcp_url}/tools/pcb_inspect",
                                        json={"image_file_path": file_path, "board_type": "EBS7_TCM"},
                                        headers=_MCP_AUTH_HEADERS,
                                    )
                                    if img_resp.status_code == 200:
                                        tool_data = img_resp.json()
                                        status      = tool_data.get("status", "UNKNOWN")
                                        confidence  = tool_data.get("confidence", 0)
                                        conf_pct    = int(confidence * 100) if confidence <= 1 else int(confidence)
                                        defect_info = tool_data.get("defect_info")

                                        if status == "OK":
                                            agent_text = f"✅ PCB inspection complete. Status: OK — No defects detected. The board is in good condition. Confidence: {conf_pct}%."
                                        elif defect_info:
                                            defect_type = defect_info.get("defect_type", "unknown")
                                            location    = defect_info.get("location", "unknown")
                                            description = defect_info.get("description", "")
                                            agent_text = (
                                                f"⚠️ PCB inspection complete. Status: NOT_OK — {defect_type.upper()} defect detected"
                                                f" at location {location}. Confidence: {conf_pct}%."
                                                + (f" {description}" if description else "")
                                            )
                                        else:
                                            agent_text = f"PCB inspection complete. Status: {status}. Confidence: {conf_pct}%."

                                        trace = [
                                            {"step": 1, "type": "user_input",    "content": f"PCB image received: {Path(file_path).name}"},
                                            {"step": 2, "type": "llm_reasoning", "content": f"LLM ({agent.lm_studio_model}) processed message and triggered pcb_inspect tool"},
                                            {"step": 3, "type": "tool_call",     "tool": "pcb_inspect", "content": f"pcb_inspect(image_file_path={file_path!r}, board_type='EBS7_TCM')"},
                                            {"step": 4, "type": "tool_result",   "content": f"status={status}, confidence={conf_pct}%"},
                                            {"step": 5, "type": "final_response","content": "Structured result + images returned to user"},
                                        ]
                                        result["response"]    = agent_text
                                        result["tool_result"] = tool_data
                                        result["trace"]       = trace
                            except Exception as e:
                                logger.warning(f"Image fetch failed: {e}")

                    # Delete uploaded file after processing
                    if file_path:
                        try:
                            Path(file_path).unlink(missing_ok=True)
                            logger.info(f"🗑️ Deleted upload: {file_path}")
                        except Exception:
                            pass

                    # Handle both dict and string responses
                    if isinstance(result, dict):
                        response_data = result.copy()
                        response_data['session_id'] = request.session_id or "default_session"
                        return JSONResponse(response_data)
                    else:
                        return JSONResponse({
                            "response": str(result),
                            "session_id": request.session_id or "default_session"
                        })
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.post("/upload")
            async def upload_image(file: UploadFile = File(...)):
                """Upload endpoint - saves image file and returns path"""
                try:
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_extension = Path(file.filename).suffix
                    unique_filename = f"pcb_{timestamp}{file_extension}"
                    file_path = uploads_dir / unique_filename
                    
                    # Save file
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    logger.info(f"✅ File uploaded: {file_path}")
                    
                    return JSONResponse({
                        "success": True,
                        "file_path": str(file_path.absolute()),
                        "filename": unique_filename
                    })
                except Exception as e:
                    logger.error(f"Upload error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.post("/inspect_direct")
            async def inspect_direct(file: UploadFile = File(...)):
                """Direct inspection endpoint - uploads and inspects image, returns full structured results with images"""
                try:
                    import httpx
                    
                    # Save uploaded file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_extension = Path(file.filename).suffix
                    unique_filename = f"pcb_{timestamp}{file_extension}"
                    file_path = uploads_dir / unique_filename
                    
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    logger.info(f"✅ File uploaded for direct inspection: {file_path}")
                    
                    # Call MCP server's inspect_pcb tool directly
                    # The MCP server should have a REST endpoint we can call
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        # Try calling MCP server's tool endpoint
                        mcp_response = await client.post(
                            f"{mcp_server_url}/tools/pcb_inspect",
                            json={
                                "image_file_path": str(file_path.absolute()),
                                "board_type": "EBS7_TCM"
                            },
                            headers=_MCP_AUTH_HEADERS,
                        )
                        
                        if mcp_response.status_code == 200:
                            result = mcp_response.json()
                            logger.info(f"✅ MCP inspection result: Status={result.get('status')}")
                            return JSONResponse(result)
                        else:
                            # Fall back to agent-based inspection
                            logger.warning(f"MCP direct call failed ({mcp_response.status_code}), using agent...")
                            message = f"Please inspect this PCB image at path: {file_path.absolute()}. Use the pcb_inspect tool."
                            response_text = await agent.process_request(message, f"inspection_{timestamp}")
                            return JSONResponse({
                                "success": True,
                                "response": response_text,
                                "file_path": str(file_path.absolute())
                            })
                    
                except Exception as e:
                    logger.error(f"Direct inspection error: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.post("/inspect", response_model=InspectionResponse)
            async def inspect(request: InspectionRequest):
                """Legacy inspect endpoint for backward compatibility"""
                try:
                    result = await agent.process_request(
                        request.message,
                        request.session_id
                    )
                    # Extract text response
                    if isinstance(result, dict):
                        response_text = result.get('response', str(result))
                    else:
                        response_text = str(result)
                    
                    return InspectionResponse(
                        response=response_text,
                        session_id=request.session_id
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Run the server
            uvicorn.run(
                app,
                host=agent_host,
                port=agent_port,
                log_level="info"
            )
    
    except Exception as e:
        logger.error(f"❌ Failed to start agent: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
