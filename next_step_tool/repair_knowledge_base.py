"""
PCB Repair Knowledge Base
Contains repair procedures, steps, and defect-specific knowledge
"""

REPAIR_STEPS = {
    "general": [
        {
            "step_number": 1,
            "title": "Preparation",
            "description": "Prepare the workspace and gather necessary tools",
            "instructions": [
                "Power off the PCB completely and disconnect all power sources",
                "Set up a well-lit, static-free workspace with ESD protection",
                "Gather all required tools and replacement components",
                "Document the current state with photographs",
                "Review the defect location and plan the repair approach"
            ],
            "tools_needed": [
                "ESD wrist strap and mat",
                "Soldering iron (adjustable temperature)",
                "Magnifying glass or microscope",
                "Isopropyl alcohol (>90%)",
                "Lint-free cloths",
                "Flux pen",
                "Solder wick or desoldering pump"
            ],
            "estimated_time": "5-10 minutes",
            "safety_warnings": [
                "Always wear ESD protection when handling PCBs",
                "Ensure adequate ventilation when soldering",
                "Never work on powered circuits",
                "Use appropriate heat protection for surrounding components"
            ]
        },
        {
            "step_number": 2,
            "title": "Inspection",
            "description": "Thoroughly inspect the defect and surrounding area",
            "instructions": [
                "Use magnification to examine the defect location closely",
                "Check for secondary damage in surrounding areas",
                "Test continuity of traces leading to the defect",
                "Document all findings with detailed notes",
                "Identify the root cause of the defect if possible"
            ],
            "tools_needed": [
                "Magnifying glass or microscope (10x-40x)",
                "Multimeter for continuity testing",
                "Flashlight for better visibility",
                "Camera for documentation"
            ],
            "estimated_time": "10-15 minutes",
            "safety_warnings": [
                "Ensure PCB is fully discharged before testing",
                "Use appropriate probe tips to avoid damaging pads"
            ]
        },
        {
            "step_number": 3,
            "title": "Removal",
            "description": "Safely remove damaged component or material",
            "instructions": [
                "Apply flux to the affected area",
                "Heat the solder joint to working temperature (typically 350-400°C)",
                "Use solder wick or desoldering pump to remove excess solder",
                "Gently remove the damaged component with tweezers",
                "Clean the pad area with isopropyl alcohol",
                "Verify pad integrity after removal"
            ],
            "tools_needed": [
                "Soldering iron with appropriate tip",
                "Solder wick or desoldering pump",
                "Flux pen",
                "Precision tweezers",
                "Isopropyl alcohol",
                "Cotton swabs or lint-free cloths"
            ],
            "estimated_time": "15-20 minutes",
            "safety_warnings": [
                "Avoid overheating the PCB - work quickly and efficiently",
                "Protect nearby components with heat shields if necessary",
                "Never pull components with force - ensure solder is fully melted"
            ]
        },
        {
            "step_number": 4,
            "title": "Application",
            "description": "Apply new component or repair damaged area",
            "instructions": [
                "Verify the replacement component matches specifications",
                "Apply fresh flux to the cleaned pad",
                "Position the new component carefully with proper alignment",
                "Apply solder at the correct temperature and amount",
                "Ensure proper solder joint formation (concave fillet)",
                "Clean excess flux with isopropyl alcohol"
            ],
            "tools_needed": [
                "Replacement component (verified match)",
                "Soldering iron",
                "High-quality solder (lead-free or leaded as required)",
                "Flux pen",
                "Tweezers for component placement",
                "Isopropyl alcohol for cleaning"
            ],
            "estimated_time": "10-15 minutes",
            "safety_warnings": [
                "Verify component polarity before soldering",
                "Use appropriate solder temperature for the component type",
                "Avoid cold solder joints - ensure proper heat transfer"
            ]
        },
        {
            "step_number": 5,
            "title": "Testing",
            "description": "Test the repair and verify functionality",
            "instructions": [
                "Visually inspect all solder joints under magnification",
                "Test continuity with a multimeter",
                "Check for short circuits between adjacent pads",
                "Measure resistance and voltage at test points if applicable",
                "Power on the PCB in a controlled manner",
                "Verify the repaired area functions correctly"
            ],
            "tools_needed": [
                "Multimeter",
                "Power supply (adjustable)",
                "Oscilloscope (if available)",
                "Magnifying glass"
            ],
            "estimated_time": "15-20 minutes",
            "safety_warnings": [
                "Start with low voltage when first powering on",
                "Monitor for unusual heat or odors during testing",
                "Be prepared to disconnect power immediately if issues arise"
            ]
        },
        {
            "step_number": 6,
            "title": "Verification",
            "description": "Final verification and documentation",
            "instructions": [
                "Perform full functional test of the PCB",
                "Document the repair with photographs",
                "Update maintenance records",
                "Apply conformal coating if required",
                "Perform final quality check",
                "Label the PCB with repair date if applicable"
            ],
            "tools_needed": [
                "Camera for documentation",
                "Conformal coating (if required)",
                "Labels and markers"
            ],
            "estimated_time": "5-10 minutes",
            "safety_warnings": [
                "Ensure all flux residue is cleaned before coating",
                "Allow adequate curing time for conformal coating"
            ]
        }
    ]
}

DEFECT_SPECIFIC_NOTES = {
    "burned": {
        "additional_checks": [
            "Inspect surrounding components for heat damage",
            "Check for carbonized PCB material that may need removal",
            "Verify power supply stability to prevent future burning",
            "Test surrounding traces for damage"
        ],
        "common_causes": [
            "Overcurrent condition",
            "Component failure",
            "Poor thermal management",
            "Power supply surge"
        ],
        "prevention": "Install current limiting or use higher rated components"
    },
    "corrosion": {
        "additional_checks": [
            "Identify source of moisture or contamination",
            "Check for corrosion spread to nearby areas",
            "Test insulation resistance after cleaning",
            "Consider protective coating after repair"
        ],
        "common_causes": [
            "Moisture ingress",
            "Chemical contamination",
            "Galvanic corrosion",
            "Poor environmental protection"
        ],
        "prevention": "Apply conformal coating and ensure proper enclosure sealing"
    },
    "cold_solder": {
        "additional_checks": [
            "Inspect all joints in the same area for similar issues",
            "Verify soldering iron temperature was appropriate",
            "Check for contamination on pads before repair",
            "Test mechanical strength of joint after reflow"
        ],
        "common_causes": [
            "Insufficient heat during soldering",
            "Contaminated surfaces",
            "Improper solder type",
            "Component movement during cooling"
        ],
        "prevention": "Use proper soldering temperature and technique, ensure clean surfaces"
    },
    "missing_component": {
        "additional_checks": [
            "Verify correct component part number",
            "Check for damage to pads where component was missing",
            "Test surrounding circuit for damage",
            "Verify component orientation and polarity"
        ],
        "common_causes": [
            "Manufacturing defect",
            "Mechanical shock or vibration",
            "Thermal stress",
            "Poor solder joint quality"
        ],
        "prevention": "Improve solder quality and consider conformal coating or potting"
    }
}

def get_repair_steps():
    """Get the general repair steps"""
    return REPAIR_STEPS["general"]

def get_defect_notes(defect_type):
    """Get additional notes specific to defect type"""
    return DEFECT_SPECIFIC_NOTES.get(defect_type, {})

def generate_repair_guide(defect_type, defect_cells):
    """
    Generate a complete repair guide for a specific defect
    
    Args:
        defect_type: Type of defect (burned, corrosion, cold_solder, missing_component)
        defect_cells: List of affected PCB grid cells (e.g., ['C4', 'D3'])
    
    Returns:
        Dictionary with repair steps and defect-specific information
    """
    steps = get_repair_steps()
    defect_notes = get_defect_notes(defect_type)
    
    # Enhance steps with defect-specific information
    enhanced_steps = []
    for step in steps:
        enhanced_step = step.copy()
        
        # Add defect location context to inspection step
        if step["step_number"] == 2:
            enhanced_step["instructions"].insert(
                0, 
                f"Focus inspection on grid cells: {', '.join(defect_cells)}"
            )
        
        # Add defect-specific checks where relevant
        if step["step_number"] == 2 and defect_notes:
            additional_checks = defect_notes.get("additional_checks", [])
            if additional_checks:
                enhanced_step["instructions"].extend([
                    f"DEFECT-SPECIFIC: {check}" for check in additional_checks
                ])
        
        enhanced_steps.append(enhanced_step)
    
    return {
        "defect_type": defect_type,
        "affected_cells": defect_cells,
        "steps": enhanced_steps,
        "defect_notes": defect_notes
    }
