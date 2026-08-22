"""
PCB Grid Coordinate Mapper

Maps alphanumeric grid coordinates (A1-D8) to pixel regions on PCB images.

Grid Structure:
- Rows: A, B, C, D (4 rows)
- Columns: 1-8 (8 columns)
- Two PCB views:
  * Left view: Columns 1-4 (A1 to D4) - 16 squares
  * Right view: Columns 5-8 (A5 to D8) - 16 squares
"""
import re
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


class PCBGridMapper:
    """Maps grid coordinates to pixel regions on PCB images"""
    
    # Grid structure
    ROWS = ['A', 'B', 'C', 'D']  # 4 rows
    COLS = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 columns
    
    # View definitions
    LEFT_VIEW_COLS = [1, 2, 3, 4]  # Left PCB view
    RIGHT_VIEW_COLS = [5, 6, 7, 8]  # Right PCB view
    
    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        """
        Initialize grid mapper
        
        Args:
            image_width: Width of PCB image in pixels
            image_height: Height of PCB image in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # Grid dimensions will be calculated dynamically
        self._update_grid_dimensions()
    
    def _update_grid_dimensions(self):
        """
        Update grid dimensions based on image aspect ratio
        - Wide images (4:3 or wider): Two 4×4 views side-by-side (8 columns total)
        - Square/tall images: Single 4×4 view (columns 1-4 and 5-8 map to same grid)
        
        IMPORTANT: Each cropped image is divided into exactly 4x4 grid squares
        using the actual image dimensions (W/4 × H/4 per cell)
        """
        aspect_ratio = self.image_width / self.image_height if self.image_height > 0 else 1.0
        
        # If image is wide (aspect >= 1.3, like 4:3), treat as two views side-by-side
        if aspect_ratio >= 1.3:
            # TWO 4×4 views side-by-side (left and right)
            self.is_dual_view = True
            
            # Each cell spans exactly W/8 width and H/4 height
            self.cell_width = self.image_width / 8.0   # 8 columns total
            self.cell_height = self.image_height / 4.0  # 4 rows
            
            # Each view is 4 columns wide
            self.view_width = self.cell_width * 4
            self.view_height = self.cell_height * 4
            
            logger.debug(f"Grid: DUAL view (8 cols), cell={self.cell_width:.1f}×{self.cell_height:.1f}px, aspect={aspect_ratio:.2f}")
        else:
            # SINGLE 4×4 view (square or tall image) - most common after cropping
            self.is_dual_view = False
            
            # Divide image into exactly 4×4 grid using actual dimensions
            # Each cell is W/4 wide and H/4 tall
            self.cell_width = self.image_width / 4.0   # Exact division by 4 columns
            self.cell_height = self.image_height / 4.0  # Exact division by 4 rows
            
            # Single view covers the entire image
            self.view_width = self.image_width
            self.view_height = self.image_height
            
            logger.debug(f"Grid: SINGLE 4×4 view, cell={self.cell_width:.1f}×{self.cell_height:.1f}px, image={self.image_width}×{self.image_height}px")
        
        # For backward compatibility (some code may use cell_size)
        self.cell_size = min(self.cell_width, self.cell_height)
    
    def parse_location(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Parse grid location string (e.g., "B1", "D8")
        
        Args:
            location: Grid coordinate string (e.g., "B1", "D8")
            
        Returns:
            Dictionary with row, column, and view info, or None if invalid
        """
        if not location or location.lower() in ['none', 'unknown']:
            return None
        
        # Parse format: Letter (A-D) + Number (1-8)
        match = re.match(r'^([A-D])([1-8])$', location.upper())
        if not match:
            logger.warning(f"Invalid location format: {location}")
            return None
        
        row = match.group(1)
        col = int(match.group(2))
        
        # Determine which view (left or right)
        if col in self.LEFT_VIEW_COLS:
            view = 'left'
            view_col = col  # 1-4
        else:  # col in RIGHT_VIEW_COLS
            view = 'right'
            view_col = col - 4  # 5-8 becomes 1-4
        
        return {
            'row': row,
            'col': col,
            'view': view,
            'view_col': view_col,
            'row_index': self.ROWS.index(row),
            'col_index': col - 1
        }
    
    def get_cell_bbox(self, location: str, image: Optional[Image.Image] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box (x1, y1, x2, y2) for a grid cell
        
        Args:
            location: Grid coordinate string (e.g., "B1", "D8")
            image: Optional PIL Image to get actual dimensions
            
        Returns:
            Tuple of (x1, y1, x2, y2) pixel coordinates, or None if invalid
        """
        loc_info = self.parse_location(location)
        if not loc_info:
            return None
        
        # Update dimensions if image provided
        if image:
            self.image_width = image.width
            self.image_height = image.height
            self._update_grid_dimensions()
        
        # Determine layout based on location:
        # - Columns 1-4 (left view): map directly to 4×4 grid
        # - Columns 5-8 (right view): map directly to 4×4 grid
        # The location tells us which view this image represents!
        
        row_idx = loc_info['row_index']  # 0-3 (A, B, C, D)
        col = loc_info['col']  # 1-8
        
        if self.is_dual_view:
            # Wide image with two views side-by-side
            # Left view (cols 1-4) starts at x=0
            # Right view (cols 5-8) starts at x=view_width
            view_col_idx = loc_info['view_col'] - 1  # 0-3 within the view
            
            if loc_info['view'] == 'left':
                base_x = 0
            else:  # right view
                base_x = self.view_width
            
            x1 = base_x + (view_col_idx * self.cell_width)
            y1 = row_idx * self.cell_height
        else:
            # Single view image (square/tall) - CROPPED IMAGE
            # This is a single 4×4 grid representing either:
            #   - Left view (A1-D4) if defect location is in cols 1-4
            #   - Right view (A5-D8) if defect location is in cols 5-8
            # 
            # We map the view_col (1-4) to the 4 columns of the cropped image
            view_col_idx = loc_info['view_col'] - 1  # 0-3
            
            # Calculate pixel position in the 4×4 grid
            # Column: 0-3 maps to x positions: 0, W/4, 2W/4, 3W/4
            # Row: 0-3 maps to y positions: 0, H/4, 2H/4, 3H/4
            x1 = view_col_idx * self.cell_width
            y1 = row_idx * self.cell_height
        
        x2 = x1 + self.cell_width
        y2 = y1 + self.cell_height
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def mark_defect_area(
        self, 
        image: Image.Image, 
        location: str,
        color: str = 'red',
        line_width: int = 10,  # Increased from 5 to 10 for more visible marking
        label: Optional[str] = None,
        fill_alpha: int = 40,  # Increased from 15 to 40 for more visible fill
        padding_percent_x: float = 0.80,  # Expand bbox by 80% on left/right sides (extra width)
        padding_percent_y: float = 0.05  # Expand bbox by 5% on top/bottom sides (minimal height)
    ) -> Image.Image:
        """
        Mark defect area on PCB image with bounding box
        
        Args:
            image: PIL Image of PCB
            location: Grid coordinate (e.g., "B1", "D8")
            color: Color for marking (default: red)
            line_width: Width of bounding box lines
            label: Optional label text to display
            fill_alpha: Alpha transparency for fill (0-255)
            padding_percent_x: Percentage to expand bbox beyond cell horizontally (0.25 = 25% padding)
            padding_percent_y: Percentage to expand bbox beyond cell vertically (0.10 = 10% padding)
            
        Returns:
            New PIL Image with marked area
        """
        bbox = self.get_cell_bbox(location, image)
        if not bbox:
            logger.warning(f"Could not mark location: {location}")
            return image.copy()
        
        # Create a copy of the image
        marked_img = image.copy()
        draw = ImageDraw.Draw(marked_img, 'RGBA')
        
        x1, y1, x2, y2 = bbox
        
        # Add padding to expand the marked area - more in width, less in height
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_percent_x)
        pad_y = int(height * padding_percent_y)
        
        # Expand bbox with padding, keeping within image bounds
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image.width, x2 + pad_x)
        y2 = min(image.height, y2 + pad_y)
        
        # Draw semi-transparent fill
        fill_color = self._hex_to_rgba(color, fill_alpha)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        # Draw thick border
        for i in range(line_width):
            draw.rectangle(
                [x1 + i, y1 + i, x2 - i, y2 - i],
                outline=color,
                width=1
            )
        
        # Add label if provided
        if label:
            try:
                # Try to use a larger font - increased for better visibility
                font_size = max(60, int(self.cell_height * 0.20))  # Increased from 0.15 to 0.20
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add label with background
            label_text = f"{location}: {label}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position label above the box
            label_x = x1 + (x2 - x1 - text_width) // 2
            label_y = max(10, y1 - text_height - 10)
            
            # Draw label background
            padding = 10
            draw.rectangle(
                [label_x - padding, label_y - padding, 
                 label_x + text_width + padding, label_y + text_height + padding],
                fill=(0, 0, 0, 200)
            )
            
            # Draw label text
            draw.text((label_x, label_y), label_text, fill='white', font=font)
        
        return marked_img
    
    def zoom_to_area(
        self, 
        image: Image.Image, 
        location: str,
        zoom_factor: float = 1.5,
        target_size: Tuple[int, int] = (800, 800),
        padding_percent_x: float = 0.40,  # More padding horizontally (40% for extra width)
        padding_percent_y: float = 0.10  # Less padding vertically (10% for correct height)
    ) -> Optional[Image.Image]:
        """
        Zoom into defect area
        
        Args:
            image: PIL Image of PCB
            location: Grid coordinate (e.g., "B1", "D8")
            zoom_factor: How much to expand beyond cell boundaries (1.0 = exact cell)
            target_size: Desired output size
            padding_percent_x: Horizontal padding to expand beyond cell (0.25 = 25%)
            padding_percent_y: Vertical padding to expand beyond cell (0.10 = 10%)
            
        Returns:
            Zoomed PIL Image, or None if invalid location
        """
        bbox = self.get_cell_bbox(location, image)
        if not bbox:
            return None
        
        x1, y1, x2, y2 = bbox
        
        # Add padding to the cell before zooming - more in width, less in height
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_percent_x)
        pad_y = int(height * padding_percent_y)
        
        # Expand bbox with padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image.width, x2 + pad_x)
        y2 = min(image.height, y2 + pad_y)
        
        # Recalculate dimensions with padding
        width = x2 - x1
        height = y2 - y1
        
        # Calculate center and dimensions
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Apply zoom factor
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Calculate new bounds (keep within image)
        crop_x1 = max(0, center_x - new_width // 2)
        crop_y1 = max(0, center_y - new_height // 2)
        crop_x2 = min(image.width, center_x + new_width // 2)
        crop_y2 = min(image.height, center_y + new_height // 2)
        
        # Crop and resize
        zoomed = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        zoomed = zoomed.resize(target_size, Image.Resampling.LANCZOS)
        
        return zoomed
    
    def _hex_to_rgba(self, hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        """Convert hex color to RGBA tuple"""
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        
        # Handle color names
        color_map = {
            'red': 'FF0000',
            'green': '00FF00',
            'blue': '0000FF',
            'yellow': 'FFFF00',
            'orange': 'FFA500',
            'purple': '800080'
        }
        hex_color = color_map.get(hex_color.lower(), hex_color)
        
        # Convert to RGB
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b, alpha)
        
        return (255, 0, 0, alpha)  # Default to red
    
    def get_view_info(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Get information about which PCB view contains the location
        
        Args:
            location: Grid coordinate (e.g., "B1", "D8")
            
        Returns:
            Dictionary with view information including specific location
        """
        loc_info = self.parse_location(location)
        if not loc_info:
            return None
        
        if loc_info['view'] == 'left':
            return {
                'location': location.upper(),
                'view': 'left',
                'view_name': 'Left PCB View',
                'columns': '1-4',
                'grid_range': 'A1 to D4',
                'description': f'Defect located at {location.upper()} on the left side of the PCB'
            }
        else:
            return {
                'location': location.upper(),
                'view': 'right',
                'view_name': 'Right PCB View',
                'columns': '5-8',
                'grid_range': 'A5 to D8',
                'description': f'Defect located at {location.upper()} on the right side of the PCB'
            }
    
    def draw_grid_overlay(self, image: Image.Image, line_color: str = 'cyan', 
                          line_width: int = 2, show_labels: bool = True) -> Image.Image:
        """
        Draw 4×4 grid overlay on image to show cell boundaries
        
        Args:
            image: PIL Image to draw grid on
            line_color: Color for grid lines
            line_width: Width of grid lines
            show_labels: Whether to show cell labels (A1, B2, etc.)
            
        Returns:
            New image with grid overlay
        """
        # Update dimensions for this image
        self.image_width = image.width
        self.image_height = image.height
        self._update_grid_dimensions()
        
        # Create copy
        gridded = image.copy()
        draw = ImageDraw.Draw(gridded, 'RGBA')
        
        # Draw vertical lines (5 lines for 4 columns: 0, 1/4, 2/4, 3/4, 1)
        for i in range(5):
            x = i * self.cell_width
            draw.line([(x, 0), (x, image.height)], fill=line_color, width=line_width)
        
        # Draw horizontal lines (5 lines for 4 rows)
        for i in range(5):
            y = i * self.cell_height
            draw.line([(0, y), (image.width, y)], fill=line_color, width=line_width)
        
        # Add cell labels if requested
        if show_labels:
            try:
                font_size = max(20, int(self.cell_height * 0.08))
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()
            
            # Label each cell
            for row_idx, row in enumerate(self.ROWS):
                for col_idx in range(4):
                    # Calculate center of cell
                    x_center = (col_idx + 0.5) * self.cell_width
                    y_center = (row_idx + 0.5) * self.cell_height
                    
                    # Create label (e.g., "A1")
                    # Note: this shows the grid as if it's a single view
                    # The actual location mapping depends on which view this image represents
                    label = f"{row}{col_idx + 1}"
                    
                    # Draw label with background
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    
                    label_x = int(x_center - text_w / 2)
                    label_y = int(y_center - text_h / 2)
                    
                    # Semi-transparent background
                    padding = 5
                    draw.rectangle(
                        [label_x - padding, label_y - padding,
                         label_x + text_w + padding, label_y + text_h + padding],
                        fill=(0, 0, 0, 100)
                    )
                    draw.text((label_x, label_y), label, fill='white', font=font)
        
        return gridded
