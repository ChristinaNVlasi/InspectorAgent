# Known Issues and Debugging Guide

## Issue Summary

### The 404 Mystery
The repair guide tool was successfully:
- ✅ Registered in MCP server (`/tools/schemas` showed it)
- ✅ Deployed to server (files verified on disk)
- ✅ Server restarted (PID changed confirming new process)
- ❌ **BUT**: Returns 404 "Not Found" when POST request sent

### What We Tried

1. **Simplified Visualizations** (452 lines)
   - Removed complex step-specific diagram generation
   - Used single overview diagram for all steps
   - Still returned 404

2. **Multiple Server Restarts**
   - Killed process with `fuser -k 3030/tcp`
   - Killed with `pkill -f 'global_mcp_server.py'`
   - Manually verified new PID after restart
   - Tool still returned 404

3. **File Verification**
   - Confirmed files deployed with `rsync`
   - Verified line counts with `wc -l`
   - Checked code with `grep` for key functions
   - Used `python -m py_compile` to verify syntax
   - Files were correct on server

4. **Tool Registration Check**
   ```bash
   curl http://localhost:3030/tools/schemas | python3 -m json.tool | grep 'pcb_generate_repair_guide'
   ```
   - ✅ Tool appeared in schemas
   - ✅ Tool name was correct
   - ❌ POST to tool returned 404

### Error Patterns

#### Server Logs
```
INFO:     127.0.0.1:49598 - "POST /tools/pcb_generate_repair_guide HTTP/1.1" 404 Not Found
```

#### Client Error
```
curl -X POST http://localhost:3030/tools/pcb_generate_repair_guide ...
# Returns: Not Found (empty response)
```

#### Intermittent Issue
Sometimes also saw:
```
Response ended prematurely
```
This suggests the server might be crashing during tool execution.

## Root Cause Hypotheses

### Theory 1: Import Errors
The tool imports may be failing silently:
```python
from apps.pcb_defect_detector.models.repair_knowledge_base import generate_repair_guide
from apps.pcb_defect_detector.models.repair_visualizer import create_2d_repair_diagram
```

**Why this could cause 404:**
- FastMCP might skip tool registration if imports fail
- But tool still appears in schemas (contradicts this)

**To Debug:**
- Add logging in tool function to see if it's ever called
- Check for import errors in server startup logs
- Test imports in Python shell on server

### Theory 2: FastMCP Route Mismatch
The MCP server uses FastMCP which might have routing issues:
- Tool registered in schemas correctly
- But HTTP route not created properly
- Could be FastMCP bug or configuration issue

**To Debug:**
- Check FastMCP version
- Try registering simpler tool to test framework
- Look at other working tools for differences

### Theory 3: Memory/Code Caching
Server might be caching old code despite restart:
- Python module cache
- `.pyc` files from old version
- Uvicorn hot reload issues

**To Debug:**
- Clear all `__pycache__` directories
- Use `python -B` flag to skip `.pyc` files
- Restart with fresh Python interpreter

### Theory 4: PIL/Pillow Dependency
The visualizer uses PIL for image generation:
```python
from PIL import Image, ImageDraw, ImageFont
```

**Why this could cause crash:**
- PIL might not be installed on server
- Version mismatch causing import failure
- Missing system libraries (fonts, etc.)

**To Debug:**
- Test `from PIL import Image` on server
- Check if Pillow is in requirements.txt
- Try tool without include_diagrams=true

## Recommended Debugging Approach

### Phase 1: Verify Environment
```bash
# SSH to server
ssh cvlassi@10.104.100.173

# Test imports
cd ~/sma/agentic/apps
source ~/sma/agentic/.venv/bin/activate
python -c "from pcb_defect_detector.models.repair_visualizer import create_2d_repair_diagram; print('Import OK')"
python -c "from PIL import Image; print('PIL OK')"
```

### Phase 2: Add Logging
Add debug logging to the tool function:
```python
@mcp.tool()
def pcb_generate_repair_guide(...):
    print("=" * 50)
    print("REPAIR GUIDE TOOL CALLED")
    print(f"Args: defect_type={defect_type}, cells={defect_cells}")
    print("=" * 50)
    
    # Rest of function...
```

Check if this ever appears in logs:
```bash
tail -f ~/sma/agentic/apps/logs/global_mcp.log
```

### Phase 3: Test Minimal Version
Create ultra-minimal version:
```python
@mcp.tool()
def pcb_generate_repair_guide(defect_type: str, defect_cells: list[str]) -> dict:
    """Minimal test version"""
    return {
        "success": True,
        "message": "Tool is working!",
        "defect_type": defect_type,
        "defect_cells": defect_cells
    }
```

If this works, gradually add back complexity.

### Phase 4: Check FastMCP Issues
```bash
# Check FastMCP version
pip show fastmcp

# Look for similar issues
# Check if other tools with complex return types work
```

## Alternative Approaches

### Option 1: Separate API Endpoint
Instead of MCP tool, create dedicated FastAPI endpoint:
```python
@app.post("/api/repair-guide")
async def generate_repair_guide_endpoint(request: RepairGuideRequest):
    # Same logic but as regular API endpoint
    pass
```

**Pros:**
- More control over routing
- Easier debugging
- Standard FastAPI patterns

**Cons:**
- Not integrated with MCP protocol
- Agents need different integration

### Option 2: Return Simple Data, Render Elsewhere
Don't generate images in tool:
```python
@mcp.tool()
def pcb_generate_repair_guide(...):
    # Return only text data
    # Let frontend generate visualizations
    return {
        "success": True,
        "steps": [...],
        "defect_cells": ["C4"],  # Frontend renders this
        "diagram_data": {
            "grid_size": [4, 4],
            "highlighted_cells": [{"row": 2, "col": 3}]
        }
    }
```

**Pros:**
- Simpler tool
- Faster response
- Frontend has more control

**Cons:**
- Duplicate visualization logic
- Less portable

### Option 3: Pre-generate and Cache Diagrams
Generate diagrams at startup or on demand, store in files:
```python
# Generate diagrams once
diagrams_cache = {
    "C4_burned": "data:image/png;base64,...",
    ...
}

@mcp.tool()
def pcb_generate_repair_guide(...):
    diagram = diagrams_cache.get(f"{defect_cells[0]}_{defect_type}")
    # Return cached diagram
```

**Pros:**
- Fast response
- Pre-validated images
- No runtime PIL issues

**Cons:**
- Limited flexibility
- Large cache size

## Files to Check

When debugging, examine these files:
```
~/sma/agentic/apps/logs/global_mcp.log          # Server logs
~/sma/agentic/apps/mcp_server/global_mcp_server.py   # Tool registration
~/sma/agentic/apps/pcb_defect_detector/models/repair_visualizer.py
~/sma/agentic/apps/pcb_defect_detector/models/repair_knowledge_base.py
```

## Success Criteria

Tool is working when:
1. ✅ Appears in `/tools/schemas`
2. ✅ POST returns 200 status
3. ✅ Response has `"success": true`
4. ✅ Steps array has 6 items
5. ✅ Diagrams are valid base64 PNG strings
6. ✅ No errors in server logs

## Next Steps for Implementation

1. Start with **minimal version** (no imports, return hardcoded data)
2. Add **imports one at a time**, testing after each
3. Add **text generation** (steps without diagrams)
4. Add **simple diagram** (basic grid, no PIL initially)
5. Add **PIL diagrams** last
6. Test **thoroughly** at each step before proceeding

## Contact Information

If continuing this work:
- Original implementation: May 15-18, 2026
- Last known status: Tool reverted, code preserved here
- Test server: cvlassi@10.104.100.173
- MCP port: 3030
- Token: e9f02900efbe37708596ebed51f4ce3d44f3e92dee112fdf3617bafa9fd96028
