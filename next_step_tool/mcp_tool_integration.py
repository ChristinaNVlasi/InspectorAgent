"""
MCP Tool Integration Code for PCB Repair Guide

This file shows how to integrate the repair guide tool into global_mcp_server.py
Copy the relevant sections into your MCP server implementation.
"""

# ============================================================================
# IMPORTS - Add these to the top of global_mcp_server.py
# ============================================================================

from apps.pcb_defect_detector.models.repair_knowledge_base import (
    generate_repair_guide,
    get_defect_notes
)
from apps.pcb_defect_detector.models.repair_visualizer import (
    create_2d_repair_diagram,
    generate_all_step_diagrams
)


# ============================================================================
# MCP TOOL DEFINITION - Add this to your @mcp.tool() decorated functions
# ============================================================================

@mcp.tool()
def pcb_generate_repair_guide(
    defect_type: str,
    defect_cells: list[str],
    include_diagrams: bool = True
) -> dict:
    """
    Generate a comprehensive PCB repair guide with step-by-step instructions
    
    This tool creates detailed repair guides for PCB defects, including:
    - 6-step repair process (Preparation, Inspection, Removal, Application, Testing, Verification)
    - Detailed instructions for each step
    - Required tools and estimated time
    - Safety warnings
    - Visual diagrams showing defect locations (optional)
    
    Args:
        defect_type: Type of defect - must be one of:
            - "burned": Burned or heat-damaged components/traces
            - "corrosion": Corroded pads or components
            - "cold_solder": Cold or insufficient solder joints
            - "missing_component": Missing components
        defect_cells: List of affected PCB grid cells (e.g., ["C4", "B2"])
                     Grid uses letters A-D for rows and numbers 1-4 for columns
        include_diagrams: Whether to generate visual diagrams (default: True)
    
    Returns:
        dict: Comprehensive repair guide with structure:
            {
                "success": bool,
                "defect_type": str,
                "defect_cells": list,
                "repair_guide": {
                    "steps": [
                        {
                            "step_number": int,
                            "title": str,
                            "description": str,
                            "instructions": list[str],
                            "tools_needed": list[str],
                            "estimated_time": str,
                            "safety_warnings": list[str]
                        },
                        ...  # 6 steps total
                    ],
                    "visuals": {
                        "overview_diagram": str,  # base64 PNG
                        "step_diagrams": [
                            {"step": int, "diagram": str},  # base64 PNG per step
                            ...
                        ]
                    } | None
                }
            }
    
    Example:
        >>> result = pcb_generate_repair_guide(
        ...     defect_type="burned",
        ...     defect_cells=["C4"],
        ...     include_diagrams=True
        ... )
        >>> print(result['repair_guide']['steps'][0]['title'])
        'Preparation'
    """
    try:
        # Validate defect type
        valid_types = ["burned", "corrosion", "cold_solder", "missing_component"]
        if defect_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid defect_type. Must be one of: {', '.join(valid_types)}",
                "valid_types": valid_types
            }
        
        # Validate defect cells
        if not defect_cells or not isinstance(defect_cells, list):
            return {
                "success": False,
                "error": "defect_cells must be a non-empty list of grid cell identifiers"
            }
        
        # Generate repair guide with steps
        guide_data = generate_repair_guide(defect_type, defect_cells)
        
        # Generate visual diagrams if requested
        visuals = None
        if include_diagrams:
            try:
                # Generate overview diagram
                overview_diagram = create_2d_repair_diagram(
                    defect_type=defect_type,
                    defect_cells=defect_cells
                )
                
                # Generate step-by-step diagrams
                step_diagrams = generate_all_step_diagrams(
                    defect_type=defect_type,
                    defect_cells=defect_cells
                )
                
                visuals = {
                    "overview_diagram": overview_diagram,
                    "step_diagrams": step_diagrams
                }
            except Exception as e:
                print(f"Warning: Failed to generate diagrams: {e}")
                # Continue without diagrams rather than failing completely
                visuals = None
        
        # Build response
        repair_guide = {
            "steps": guide_data["steps"],
            "visuals": visuals
        }
        
        # Add defect-specific notes if available
        if guide_data.get("defect_notes"):
            repair_guide["defect_notes"] = guide_data["defect_notes"]
        
        return {
            "success": True,
            "defect_type": defect_type,
            "defect_cells": defect_cells,
            "repair_guide": repair_guide
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============================================================================
# TESTING - How to test the tool
# ============================================================================

def test_repair_guide_tool():
    """Test the repair guide tool"""
    
    # Test case 1: Burned defect with diagrams
    result1 = pcb_generate_repair_guide(
        defect_type="burned",
        defect_cells=["C4"],
        include_diagrams=True
    )
    print("Test 1 - Burned defect:")
    print(f"  Success: {result1['success']}")
    print(f"  Steps: {len(result1.get('repair_guide', {}).get('steps', []))}")
    print(f"  Has diagrams: {'visuals' in result1.get('repair_guide', {})}")
    
    # Test case 2: Corrosion without diagrams
    result2 = pcb_generate_repair_guide(
        defect_type="corrosion",
        defect_cells=["B2", "B3"],
        include_diagrams=False
    )
    print("\nTest 2 - Corrosion defect:")
    print(f"  Success: {result2['success']}")
    print(f"  Affected cells: {result2.get('defect_cells')}")
    
    # Test case 3: Invalid defect type
    result3 = pcb_generate_repair_guide(
        defect_type="invalid_type",
        defect_cells=["C4"],
        include_diagrams=False
    )
    print("\nTest 3 - Invalid defect type:")
    print(f"  Success: {result3['success']}")
    print(f"  Error: {result3.get('error')}")


if __name__ == "__main__":
    test_repair_guide_tool()


# ============================================================================
# CURL TESTING COMMAND
# ============================================================================

"""
Test the deployed tool with curl:

curl -X POST http://localhost:3030/tools/pcb_generate_repair_guide \
  -H 'Authorization: Bearer e9f02900efbe37708596ebed51f4ce3d44f3e92dee112fdf3617bafa9fd96028' \
  -H 'Content-Type: application/json' \
  -d '{
    "defect_type": "burned",
    "defect_cells": ["C4"],
    "include_diagrams": true
  }' | python3 -m json.tool
"""
