"""
PCB Repair Visualizer
Generates 2D visual diagrams for repair guides

SIMPLIFIED VERSION - Works reliably with basic grid visualization
"""

from PIL import Image, ImageDraw, ImageFont
import base64
import io
from typing import List, Tuple

# PCB Grid Configuration for EBS7 TCM
GRID_ROWS = 4  # A, B, C, D
GRID_COLS = 4  # 1, 2, 3, 4
GRID_LETTERS = ['A', 'B', 'C', 'D']

# Visual constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 640
CELL_SIZE = 120
GRID_START_X = 200
GRID_START_Y = 100
CELL_PADDING = 10

# Colors
COLOR_BACKGROUND = (240, 240, 245)
COLOR_GRID_LINE = (100, 100, 120)
COLOR_CELL_NORMAL = (255, 255, 255)
COLOR_CELL_DEFECT = (255, 100, 100)
COLOR_TEXT = (20, 20, 40)
COLOR_TITLE = (40, 80, 160)


def parse_cell(cell_str: str) -> Tuple[int, int]:
    """
    Parse cell string like 'C4' into row and column indices
    Returns: (row_index, col_index) e.g., 'C4' -> (2, 3)
    """
    cell_str = cell_str.upper().strip()
    if len(cell_str) < 2:
        raise ValueError(f"Invalid cell format: {cell_str}")
    
    row_letter = cell_str[0]
    col_number = cell_str[1:]
    
    if row_letter not in GRID_LETTERS:
        raise ValueError(f"Invalid row letter: {row_letter}")
    
    try:
        col_num = int(col_number)
        if col_num < 1 or col_num > GRID_COLS:
            raise ValueError(f"Column number out of range: {col_num}")
    except ValueError:
        raise ValueError(f"Invalid column number: {col_number}")
    
    row_index = GRID_LETTERS.index(row_letter)
    col_index = col_num - 1
    
    return (row_index, col_index)


def create_2d_repair_diagram(
    defect_type: str,
    defect_cells: List[str],
    step_number: int = None,
    step_description: str = None
) -> str:
    """
    Create a simple 2D grid diagram highlighting defect locations
    
    SIMPLIFIED VERSION: Always returns the same overview diagram for all steps
    This version proved to be more reliable than complex step-specific visualizations
    
    Args:
        defect_type: Type of defect (burned, corrosion, etc.)
        defect_cells: List of affected grid cells (e.g., ['C4', 'D3'])
        step_number: Which repair step (1-6) - UNUSED in simplified version
        step_description: Description of the step - UNUSED in simplified version
    
    Returns:
        Base64 encoded PNG image string with data URI prefix
    """
    # Always return simple overview diagram (step-specific visuals removed)
    return _create_overview_diagram(defect_type, defect_cells)


def _create_overview_diagram(defect_type: str, defect_cells: List[str]) -> str:
    """
    Create a simple overview diagram showing the PCB grid with highlighted defects
    
    This is the WORKING version that proved reliable in production testing
    """
    # Create image
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        cell_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()
    
    # Draw title
    title = f"PCB Repair Guide - {defect_type.upper()} Defect"
    draw.text((IMAGE_WIDTH // 2, 30), title, fill=COLOR_TITLE, 
              font=title_font, anchor="mm")
    
    # Parse defect cell positions
    defect_positions = []
    for cell in defect_cells:
        try:
            pos = parse_cell(cell)
            defect_positions.append(pos)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    # Draw PCB grid
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Calculate cell position
            x = GRID_START_X + col * (CELL_SIZE + CELL_PADDING)
            y = GRID_START_Y + row * (CELL_SIZE + CELL_PADDING)
            
            # Determine if this cell has a defect
            is_defect = (row, col) in defect_positions
            cell_color = COLOR_CELL_DEFECT if is_defect else COLOR_CELL_NORMAL
            
            # Draw cell
            draw.rectangle(
                [x, y, x + CELL_SIZE, y + CELL_SIZE],
                fill=cell_color,
                outline=COLOR_GRID_LINE,
                width=2
            )
            
            # Draw cell label
            cell_label = f"{GRID_LETTERS[row]}{col + 1}"
            draw.text(
                (x + CELL_SIZE // 2, y + CELL_SIZE // 2),
                cell_label,
                fill=COLOR_TEXT,
                font=cell_font,
                anchor="mm"
            )
            
            # If defect, add marker
            if is_defect:
                # Draw red X
                padding = 20
                draw.line(
                    [(x + padding, y + padding), 
                     (x + CELL_SIZE - padding, y + CELL_SIZE - padding)],
                    fill=(200, 0, 0),
                    width=4
                )
                draw.line(
                    [(x + CELL_SIZE - padding, y + padding),
                     (x + padding, y + CELL_SIZE - padding)],
                    fill=(200, 0, 0),
                    width=4
                )
    
    # Draw row labels (A, B, C, D)
    for row in range(GRID_ROWS):
        y = GRID_START_Y + row * (CELL_SIZE + CELL_PADDING) + CELL_SIZE // 2
        draw.text(
            (GRID_START_X - 40, y),
            GRID_LETTERS[row],
            fill=COLOR_TEXT,
            font=label_font,
            anchor="mm"
        )
    
    # Draw column labels (1, 2, 3, 4)
    for col in range(GRID_COLS):
        x = GRID_START_X + col * (CELL_SIZE + CELL_PADDING) + CELL_SIZE // 2
        draw.text(
            (x, GRID_START_Y - 40),
            str(col + 1),
            fill=COLOR_TEXT,
            font=label_font,
            anchor="mm"
        )
    
    # Add legend
    legend_y = GRID_START_Y + GRID_ROWS * (CELL_SIZE + CELL_PADDING) + 30
    
    # Defect indicator
    draw.rectangle(
        [GRID_START_X, legend_y, GRID_START_X + 30, legend_y + 30],
        fill=COLOR_CELL_DEFECT,
        outline=COLOR_GRID_LINE,
        width=2
    )
    draw.text(
        (GRID_START_X + 40, legend_y + 15),
        f"Defect Location: {', '.join(defect_cells)}",
        fill=COLOR_TEXT,
        font=label_font,
        anchor="lm"
    )
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


def generate_all_step_diagrams(
    defect_type: str,
    defect_cells: List[str]
) -> List[dict]:
    """
    Generate diagrams for all 6 repair steps
    
    In the simplified version, this returns the same overview diagram for each step
    This proved more reliable than generating unique diagrams per step
    
    Args:
        defect_type: Type of defect
        defect_cells: List of affected cells
    
    Returns:
        List of dicts with step number and diagram for each step
    """
    step_diagrams = []
    
    # Generate same diagram for all 6 steps (simplified approach)
    overview_diagram = _create_overview_diagram(defect_type, defect_cells)
    
    for step_num in range(1, 7):
        step_diagrams.append({
            "step": step_num,
            "diagram": overview_diagram
        })
    
    return step_diagrams


# Example usage and testing
if __name__ == "__main__":
    # Test diagram generation
    defect_cells = ["C4", "D3"]
    defect_type = "burned"
    
    # Generate overview diagram
    diagram = create_2d_repair_diagram(defect_type, defect_cells)
    print(f"Generated diagram (length: {len(diagram)} chars)")
    print(f"Preview: {diagram[:100]}...")
    
    # Generate all step diagrams
    all_diagrams = generate_all_step_diagrams(defect_type, defect_cells)
    print(f"\nGenerated {len(all_diagrams)} step diagrams")
    for step_diag in all_diagrams:
        print(f"  Step {step_diag['step']}: {len(step_diag['diagram'])} chars")
