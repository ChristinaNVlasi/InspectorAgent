# Example API Responses

## Successful Response with Diagrams

```json
{
  "success": true,
  "defect_type": "burned",
  "defect_cells": ["C4"],
  "repair_guide": {
    "steps": [
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
          "Focus inspection on grid cells: C4",
          "Use magnification to examine the defect location closely",
          "Check for secondary damage in surrounding areas",
          "Test continuity of traces leading to the defect",
          "Document all findings with detailed notes",
          "Identify the root cause of the defect if possible",
          "DEFECT-SPECIFIC: Inspect surrounding components for heat damage",
          "DEFECT-SPECIFIC: Check for carbonized PCB material that may need removal",
          "DEFECT-SPECIFIC: Verify power supply stability to prevent future burning",
          "DEFECT-SPECIFIC: Test surrounding traces for damage"
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
    ],
    "visuals": {
      "overview_diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]",
      "step_diagrams": [
        {
          "step": 1,
          "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]"
        },
        {
          "step": 2,
          "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]"
        },
        {
          "step": 3,
          "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]"
        },
        {
          "step": 4,
          "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]"
        },
        {
          "step": 5,
          "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]"
        },
        {
          "step": 6,
          "diagram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...[truncated]"
        }
      ]
    },
    "defect_notes": {
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
    }
  }
}
```

## Response Without Diagrams

```json
{
  "success": true,
  "defect_type": "corrosion",
  "defect_cells": ["B2", "B3"],
  "repair_guide": {
    "steps": [
      {
        "step_number": 1,
        "title": "Preparation",
        "description": "Prepare the workspace and gather necessary tools",
        "instructions": [...],
        "tools_needed": [...],
        "estimated_time": "5-10 minutes",
        "safety_warnings": [...]
      }
      // ... 6 steps total
    ],
    "visuals": null,
    "defect_notes": {
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
    }
  }
}
```

## Error Response - Invalid Defect Type

```json
{
  "success": false,
  "error": "Invalid defect_type. Must be one of: burned, corrosion, cold_solder, missing_component",
  "valid_types": [
    "burned",
    "corrosion",
    "cold_solder",
    "missing_component"
  ]
}
```

## Error Response - Invalid Cell Format

```json
{
  "success": false,
  "error": "defect_cells must be a non-empty list of grid cell identifiers"
}
```

## Error Response - Server Exception

```json
{
  "success": false,
  "error": "name 'create_2d_repair_diagram' is not defined",
  "traceback": "Traceback (most recent call last):\n  File \"...\", line 123, in pcb_generate_repair_guide\n    diagram = create_2d_repair_diagram(...)\nNameError: name 'create_2d_repair_diagram' is not defined"
}
```

## Minimal Response (Testing)

For testing, a minimal working response could be:

```json
{
  "success": true,
  "message": "Tool is working!",
  "defect_type": "burned",
  "defect_cells": ["C4"]
}
```

## Frontend Display Example

How the frontend might display this data:

```html
<div class="repair-guide">
  <h2>PCB Repair Guide - BURNED Defect at C4</h2>
  
  <!-- Overview Diagram -->
  <div class="overview">
    <img src="data:image/png;base64,iVBOR..." alt="PCB Overview">
  </div>
  
  <!-- Steps -->
  <div class="steps">
    <div class="step">
      <h3>Step 1: Preparation (5-10 minutes)</h3>
      <p>Prepare the workspace and gather necessary tools</p>
      
      <!-- Step-specific diagram -->
      <img src="data:image/png;base64,iVBOR..." alt="Step 1 Diagram">
      
      <h4>Instructions:</h4>
      <ol>
        <li>Power off the PCB completely and disconnect all power sources</li>
        <li>Set up a well-lit, static-free workspace with ESD protection</li>
        <!-- ... -->
      </ol>
      
      <h4>⚠️ Safety Warnings:</h4>
      <ul>
        <li>Always wear ESD protection when handling PCBs</li>
        <!-- ... -->
      </ul>
      
      <h4>🛠️ Tools Needed:</h4>
      <ul>
        <li>ESD wrist strap and mat</li>
        <!-- ... -->
      </ul>
    </div>
    
    <!-- Repeat for steps 2-6 -->
  </div>
  
  <!-- Defect-Specific Notes -->
  <div class="defect-notes">
    <h3>Additional Notes for BURNED Defects</h3>
    <p><strong>Prevention:</strong> Install current limiting or use higher rated components</p>
    
    <h4>Common Causes:</h4>
    <ul>
      <li>Overcurrent condition</li>
      <li>Component failure</li>
      <!-- ... -->
    </ul>
  </div>
</div>
```

## cURL Test Commands

### Test with diagrams:
```bash
curl -X POST http://localhost:3030/tools/pcb_generate_repair_guide \
  -H 'Authorization: Bearer e9f02900efbe37708596ebed51f4ce3d44f3e92dee112fdf3617bafa9fd96028' \
  -H 'Content-Type: application/json' \
  -d '{
    "defect_type": "burned",
    "defect_cells": ["C4"],
    "include_diagrams": true
  }' | python3 -m json.tool
```

### Test without diagrams:
```bash
curl -X POST http://localhost:3030/tools/pcb_generate_repair_guide \
  -H 'Authorization: Bearer e9f02900efbe37708596ebed51f4ce3d44f3e92dee112fdf3617bafa9fd96028' \
  -H 'Content-Type: application/json' \
  -d '{
    "defect_type": "corrosion",
    "defect_cells": ["B2", "B3"],
    "include_diagrams": false
  }' | python3 -m json.tool
```

### Quick success check:
```bash
curl -X POST http://localhost:3030/tools/pcb_generate_repair_guide \
  -H 'Authorization: Bearer e9f02900efbe37708596ebed51f4ce3d44f3e92dee112fdf3617bafa9fd96028' \
  -H 'Content-Type: application/json' \
  -d '{"defect_type": "burned", "defect_cells": ["C4"], "include_diagrams": false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('✅ Success:', d.get('success')); print('Steps:', len(d.get('repair_guide', {}).get('steps', [])))"
```

Expected output:
```
✅ Success: True
Steps: 6
```
