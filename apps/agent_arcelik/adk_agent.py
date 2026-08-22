"""
ADK Agent for Arcelik-Beko Inspection System
Uses LM Studio LLM and connects to MCP Server on port 3028
Agent runs on port 2828
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
    logger.critical(
        "FATAL: MCP_API_TOKEN environment variable is not set. "
        "Set it before starting: export MCP_API_TOKEN=<token>."
    )
    sys.exit(1)
_MCP_AUTH_HEADERS = {"Authorization": f"Bearer {_MCP_TOKEN}"}


class ArcelikBekoInspectionAgent:
    """
    ADK Agent for Arcelik-Beko washing machine inspection.
    Orchestrates noise diagnosis and vision damage detection via MCP tools.
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
        self.lm_studio_model = lm_studio_model or os.getenv('LM_STUDIO_MODEL', 'openai/qwen3:30b-a3b-q8_0')
        self.api_key = api_key or os.getenv('LM_STUDIO_API_KEY', 'ollama')
        
        logger.info("🤖 Initializing Arcelik-Beko Inspection Agent")
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
            app_name="inspection_app",
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
        instruction = """You are an expert AI assistant for Arcelik-Beko washing machine inspection and diagnosis.

Your capabilities:
1. **Noise Diagnosis**: Analyze audio recordings to identify mechanical issues
2. **Visual Damage Detection**: Analyze images of components to detect damage

TOOLS AVAILABLE (Global MCP Server port 3030):
- wm_diagnose_noise(audio_file_path, model_id)   — diagnose washing machine noise from audio
- wm_detect_damage(image_file_path, component_hint) — detect damage in component images
- wm_get_status()                                 — model health check

CRITICAL INSTRUCTIONS:
1. Always use the EXACT tool names above (wm_ prefix is required).
2. For follow-up questions about a PREVIOUS inspection, DO NOT call tools again — answer using earlier conversation context.
3. Only call tools when inspecting a NEW file or fetching fresh status.
4. Be concise, professional, and structured.
5. **FILE PATH RULE (HIGHEST PRIORITY):** If the message starts with `[FILE PATH: /some/absolute/path]`,
   you MUST extract that exact absolute path and pass it as `audio_file_path` or `image_file_path`
   to the appropriate tool. NEVER use just a bare filename — always use the full absolute path provided.

/no_think"""

        try:
            agent = Agent(
                name="arcelik_beko_inspector",
                model=LiteLlm(
                    model=self.lm_studio_model,
                    api_key=self.api_key,
                    api_base=self.lm_studio_api_base,
                ),
                description="Washing machine inspection assistant",
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
        Process a user request through the agent.
        
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
                    content_preview = msg["content"][:200]  # Limit to 200 chars per message
                    history_text += f"{role}: {content_preview}\n"
                history_text += "]\n\nCurrent question: "
                enriched_message = history_text + user_message
            else:
                enriched_message = user_message
            
            # Ensure session exists
            try:
                await self.session_service.create_session(
                    app_name="inspection_app",
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
            
            # Run agent and collect response + tool results
            full_response = ""
            tool_results = []
            async for event in self.runner.run_async(
                user_id="default_user",
                session_id=session_id,
                new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            full_response += part.text
                        
                        # Capture tool results (function responses)
                        if hasattr(part, 'function_response') and part.function_response:
                            try:
                                func_resp = part.function_response
                                tool_name = func_resp.name if hasattr(func_resp, 'name') else 'unknown'
                                if hasattr(func_resp, 'response'):
                                    tool_results.append({
                                        'name': tool_name,
                                        'response': func_resp.response
                                    })
                            except Exception as te:
                                logger.warning(f"Could not capture tool result: {te}")
                
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
            
            # Return both text and tool results (consistent dict format like Knorr agent)
            clean = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
            _THINKING_START = re.compile(
                r'^(Okay[,. ]|Alright[,. ]|Let me\b|I need to\b|The user\b|So the user\b|'
                r'Looking at\b|I should\b|Since they\b|Wait,|Hmm,|Actually, I|But I\b|'
                r'Now, I|Alternatively,|But the user\b)',
                re.IGNORECASE,
            )
            _paras = re.split(r'\n\n+', clean)
            while _paras and _THINKING_START.match(_paras[0].strip()):
                _paras.pop(0)
            result = {"response": '\n\n'.join(_paras).strip()}
            if tool_results:
                for tool_result in tool_results:
                    if tool_result['name'] in ('wm_diagnose_noise', 'wm_detect_damage'):
                        result['tool_result'] = tool_result['response']
                        logger.info(f"✅ Tool result included: {type(tool_result['response'])}")
                        break
            # Build execution trace
            trace = [{"step": 1, "type": "user_input", "content": user_message}]
            if tool_results:
                for idx, tr in enumerate(tool_results):
                    trace.append({"step": len(trace) + 1, "type": "tool_call",
                                  "tool": tr['name'], "content": f"{tr['name']}(...)"})
                    trace.append({"step": len(trace) + 1, "type": "tool_result",
                                  "content": f"Tool '{tr['name']}' returned result"})
            else:
                trace.append({"step": 2, "type": "agent_decision",
                              "content": "No tool call needed — LLM answered from knowledge"})
            trace.append({"step": len(trace) + 1, "type": "final_response",
                          "content": "Structured diagnosis returned to user"})
            result['trace'] = trace
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
        print("🔧 Arcelik-Beko Inspection Agent - Interactive Mode")
        print("=" * 70)
        print("\nI can help you diagnose washing machine issues:")
        print("  • Noise diagnosis from audio recordings")
        print("  • Visual damage detection from images")
        print("\nExamples:")
        print("  - 'Diagnose the noise in audio file /path/to/sound.wav'")
        print("  - 'Check for damage in image /path/to/photo.jpg'")
        print("  - 'What issues can you detect?'")
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
                
                # process_request returns a dict with at minimum a 'response' key
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
    agent_port = int(os.getenv('AGENT_PORT', '2828'))
    agent_host = os.getenv('AGENT_HOST', '0.0.0.0')
    mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:3030')
    
    logger.info("=" * 70)
    logger.info("🚀 Arcelik-Beko Inspection Agent")
    logger.info("=" * 70)
    logger.info(f"Agent Port: {agent_port}")
    logger.info(f"Agent Host: {agent_host}")
    logger.info(f"MCP Server: {mcp_server_url}")
    logger.info("=" * 70)
    
    try:
        # Initialize the agent
        agent = ArcelikBekoInspectionAgent(mcp_server_url=mcp_server_url)
        
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
            from pydantic import BaseModel
            import uvicorn
            import shutil
            import tempfile
            from datetime import datetime

            # Uploads directory — use OS temp so files are not persisted in the project
            uploads_dir = Path(tempfile.gettempdir()) / "sma_inspector"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            app = FastAPI(
                title="Arcelik-Beko Inspection Agent API",
                description="ADK Agent for washing machine inspection",
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
            
            class InspectionRequest(BaseModel):
                message: str
                session_id: Optional[str] = None
                # Absolute path returned by /upload — if provided, it is
                # injected into the message so the LLM passes the real path
                # to the MCP tool, avoiding the bare-filename lookup failure.
                file_path: Optional[str] = None
            
            class InspectionResponse(BaseModel):
                response: str
                session_id: Optional[str] = None
                tool_result: Optional[dict] = None
                trace: Optional[list] = None
            
            @app.get("/")
            async def root():
                return {
                    "service": "Arcelik-Beko Inspection Agent",
                    "status": "running",
                    "version": "1.0.0",
                    "mcp_server": mcp_server_url,
                    "capabilities": [
                        "noise_diagnosis",
                        "damage_detection"
                    ]
                }
            
            @app.get("/health")
            async def health():
                return {
                    "status": "healthy",
                    "agent_port": agent_port,
                    "mcp_server": mcp_server_url
                }

            @app.post("/upload")
            async def upload_file(file: UploadFile = File(...)):
                """
                Upload an audio or image file, save it with a unique name, and
                return the absolute path.  The UI sends this path in the next
                /inspect call so the MCP tool always receives an absolute path.
                """
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    suffix = Path(file.filename).suffix.lower()
                    safe_name = f"wm_{timestamp}{suffix}"
                    dest = uploads_dir / safe_name

                    with open(dest, "wb") as buf:
                        shutil.copyfileobj(file.file, buf)

                    abs_path = str(dest.resolve())
                    logger.info(f"✅ File uploaded: {abs_path}")
                    return {"success": True, "file_path": abs_path, "filename": safe_name}
                except Exception as e:
                    logger.error(f"Upload error: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/inspect", response_model=InspectionResponse)
            async def inspect(request: InspectionRequest):
                try:
                    message = request.message
                    # If the UI supplied an absolute file path from /upload,
                    # prepend it so the LLM uses the real path for the MCP tool.
                    if request.file_path:
                        message = (
                            f"[FILE PATH: {request.file_path}]\n"
                            + message
                        )
                        logger.info(f"📎 File path injected into message: {request.file_path}")

                    result = await agent.process_request(
                        message,
                        request.session_id
                    )

                    # Delete file immediately after processing — never store user data
                    if request.file_path:
                        try:
                            Path(request.file_path).unlink(missing_ok=True)
                            logger.info(f"🗑️ Deleted upload: {request.file_path}")
                        except Exception:
                            pass
                    # process_request always returns a dict
                    if isinstance(result, dict):
                        return InspectionResponse(
                            response=result.get('response', ''),
                            session_id=request.session_id,
                            tool_result=result.get('tool_result'),
                            trace=result.get('trace'),
                        )
                    return InspectionResponse(
                        response=str(result),
                        session_id=request.session_id,
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
