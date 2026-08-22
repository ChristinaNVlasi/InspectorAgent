"""
Grid-Based PCB Image Segmenter

Segments PCB images into grid cells (A1-D4, A5-D8) for precise defect localization.
This module handles the splitting of PCB images into individual grid cells for 
independent inspection and RAG-based similarity search.
"""
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GridSegmenter:
    """Segments PCB images into grid cells for localized inspection"""
    
    # Grid layout configuration
    ROWS = ['A', 'B', 'C', 'D']  # 4 rows
    COLS_LEFT = [1, 2, 3, 4]     # Left view columns
    COLS_RIGHT = [5, 6, 7, 8]    # Right view columns
    
    def __init__(self):
        """Initialize grid segmenter"""
        logger.info("Initialized GridSegmenter for 4x4 grid layout")
    
    def segment_image(
        self, 
        image: Image.Image,
        view_type: str = 'auto'
    ) -> Dict[str, Image.Image]:
        """
        Segment PCB image into grid cells
        
        Args:
            image: PIL Image of PCB (background already removed/cropped)
            view_type: 'left' (A1-D4), 'right' (A5-D8), 'auto', or 'dual'
            
        Returns:
            Dictionary mapping grid coordinates to cell images
            e.g., {'A1': PIL.Image, 'A2': PIL.Image, ..., 'D4': PIL.Image}
        """
        logger.info(f"Segmenting image {image.size} into grid cells (view: {view_type})")
        
        width, height = image.size
        aspect_ratio = width / height
        
        # Determine view type automatically if needed
        if view_type == 'auto':
            # Wide images (≥1.5:1) are treated as dual view (two 4x4 grids side-by-side)
            # Square/portrait images are single view (one 4x4 grid)
            if aspect_ratio >= 1.5:
                view_type = 'dual'
                logger.info(f"Auto-detected DUAL view (aspect {aspect_ratio:.2f})")
            else:
                # For single view, we need to know if it's left or right
                # Default to left view for now (will be determined by context/metadata)
                view_type = 'left'
                logger.info(f"Auto-detected SINGLE view (aspect {aspect_ratio:.2f})")
        
        # Segment based on view type
        if view_type == 'dual':
            return self._segment_dual_view(image)
        elif view_type == 'left':
            return self._segment_single_view(image, 'left')
        elif view_type == 'right':
            return self._segment_single_view(image, 'right')
        else:
            raise ValueError(f"Unknown view_type: {view_type}")
    
    def _segment_single_view(
        self, 
        image: Image.Image, 
        view: str
    ) -> Dict[str, Image.Image]:
        """
        Segment single view (4x4 grid) into cells
        
        Args:
            image: PIL Image containing one 4x4 grid
            view: 'left' (columns 1-4) or 'right' (columns 5-8)
            
        Returns:
            Dictionary mapping grid coordinates to cell images
        """
        width, height = image.size
        
        # Calculate cell dimensions (divide into 4x4 grid)
        cell_width = width / 4
        cell_height = height / 4
        
        logger.debug(f"Single view segmentation: {width}x{height} → {cell_width:.1f}x{cell_height:.1f} per cell")
        
        # Determine column numbers based on view
        cols = self.COLS_LEFT if view == 'left' else self.COLS_RIGHT
        
        # Extract each cell
        cells = {}
        for row_idx, row in enumerate(self.ROWS):
            for col_idx, col in enumerate(cols):
                # Calculate bounding box for this cell
                x1 = int(col_idx * cell_width)
                y1 = int(row_idx * cell_height)
                x2 = int((col_idx + 1) * cell_width)
                y2 = int((row_idx + 1) * cell_height)
                
                # Crop cell from image
                cell_image = image.crop((x1, y1, x2, y2))
                
                # Store with grid coordinate
                grid_coord = f"{row}{col}"
                cells[grid_coord] = cell_image
                
                logger.debug(f"Extracted cell {grid_coord}: bbox=({x1},{y1},{x2},{y2})")
        
        logger.info(f"Segmented {len(cells)} cells from single {view} view")
        return cells
    
    def _segment_dual_view(self, image: Image.Image) -> Dict[str, Image.Image]:
        """
        Segment dual view (two 4x4 grids side-by-side) into cells
        
        Args:
            image: PIL Image containing two 4x4 grids (left A1-D4, right A5-D8)
            
        Returns:
            Dictionary mapping all grid coordinates (A1-D8) to cell images
        """
        width, height = image.size
        
        # Split image in half horizontally
        half_width = width // 2
        
        logger.debug(f"Dual view segmentation: {width}x{height} → 2 views of {half_width}x{height}")
        
        # Extract left and right views
        left_view = image.crop((0, 0, half_width, height))
        right_view = image.crop((half_width, 0, width, height))
        
        # Segment each view
        left_cells = self._segment_single_view(left_view, 'left')
        right_cells = self._segment_single_view(right_view, 'right')
        
        # Combine both views
        all_cells = {**left_cells, **right_cells}
        
        logger.info(f"Segmented {len(all_cells)} cells from dual view (32 total)")
        return all_cells
    
    def get_cell_bounds(
        self, 
        image_size: Tuple[int, int],
        grid_coord: str,
        view_type: str = 'auto'
    ) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates for a specific grid cell
        
        Args:
            image_size: (width, height) of the full PCB image
            grid_coord: Grid coordinate (e.g., 'B1', 'D8')
            view_type: 'left', 'right', 'dual', or 'auto'
            
        Returns:
            Tuple (x1, y1, x2, y2) of cell bounding box
        """
        width, height = image_size
        aspect_ratio = width / height
        
        # Auto-detect view type
        if view_type == 'auto':
            view_type = 'dual' if aspect_ratio >= 1.5 else 'left'
        
        # Parse grid coordinate
        if not grid_coord or len(grid_coord) < 2:
            raise ValueError(f"Invalid grid coordinate: {grid_coord}")
        
        row = grid_coord[0].upper()
        col = int(grid_coord[1:])
        
        if row not in self.ROWS:
            raise ValueError(f"Invalid row: {row}")
        if col not in (self.COLS_LEFT + self.COLS_RIGHT):
            raise ValueError(f"Invalid column: {col}")
        
        row_idx = self.ROWS.index(row)
        
        # Calculate bounds based on view type
        if view_type == 'dual':
            # Two 4x4 grids side-by-side
            cell_width = width / 8  # 8 columns total
            cell_height = height / 4
            
            col_idx = col - 1  # 0-7
        else:
            # Single 4x4 grid
            cell_width = width / 4
            cell_height = height / 4
            
            # Map col to 0-3 index
            if view_type == 'left':
                col_idx = col - 1 if col in self.COLS_LEFT else None
            else:  # right
                col_idx = col - 5 if col in self.COLS_RIGHT else None
            
            if col_idx is None:
                raise ValueError(f"Column {col} not valid for {view_type} view")
        
        # Calculate bounding box
        x1 = int(col_idx * cell_width)
        y1 = int(row_idx * cell_height)
        x2 = int((col_idx + 1) * cell_width)
        y2 = int((row_idx + 1) * cell_height)
        
        return (x1, y1, x2, y2)
    
    def visualize_grid(
        self, 
        image: Image.Image,
        view_type: str = 'auto',
        line_color: str = 'yellow',
        line_width: int = 3,
        show_labels: bool = True
    ) -> Image.Image:
        """
        Draw grid lines and labels on image for visualization
        
        Args:
            image: PIL Image to draw on
            view_type: 'left', 'right', 'dual', or 'auto'
            line_color: Color of grid lines
            line_width: Width of grid lines
            show_labels: Whether to show grid coordinate labels
            
        Returns:
            PIL Image with grid overlay
        """
        from PIL import ImageDraw, ImageFont
        
        # Create a copy to avoid modifying original
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        width, height = image.size
        aspect_ratio = width / height
        
        # Auto-detect view type
        if view_type == 'auto':
            view_type = 'dual' if aspect_ratio >= 1.5 else 'left'
        
        # Determine number of columns
        num_cols = 8 if view_type == 'dual' else 4
        num_rows = 4
        
        cell_width = width / num_cols
        cell_height = height / num_rows
        
        # Draw vertical lines
        for i in range(1, num_cols):
            x = int(i * cell_width)
            draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
        
        # Draw horizontal lines
        for i in range(1, num_rows):
            y = int(i * cell_height)
            draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
        
        # Draw border
        draw.rectangle([(0, 0), (width-1, height-1)], outline=line_color, width=line_width)
        
        # Add labels if requested
        if show_labels:
            try:
                # Try to load a font
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
            
            # Determine which columns to label
            if view_type == 'dual':
                cols = self.COLS_LEFT + self.COLS_RIGHT
            elif view_type == 'left':
                cols = self.COLS_LEFT
            else:
                cols = self.COLS_RIGHT
            
            # Add grid coordinate labels
            for row_idx, row in enumerate(self.ROWS):
                for col_idx, col in enumerate(cols):
                    label = f"{row}{col}"
                    
                    # Calculate label position (center of cell)
                    x = int((col_idx + 0.5) * cell_width)
                    y = int((row_idx + 0.5) * cell_height)
                    
                    # Draw text with black background for visibility
                    bbox = draw.textbbox((x, y), label, font=font)
                    draw.rectangle(bbox, fill='black')
                    draw.text((x, y), label, fill='white', font=font, anchor='mm')
        
        logger.info(f"Generated grid visualization ({view_type} view)")
        return img_copy
    
    @staticmethod
    def parse_defect_location_from_path(image_path: str) -> Optional[str]:
        """
        Extract grid location from defect image path
        
        Example paths:
            - .../NOT_OK/burned_b1/image.png → 'B1'
            - .../NOT_OK/CORROSION_D8/image.png → 'D8'
        
        Args:
            image_path: Path to defect image
            
        Returns:
            Grid coordinate (e.g., 'B1', 'D8') or None if not found
        """
        import re
        from pathlib import Path
        
        # Get folder name
        path_obj = Path(image_path)
        folder_name = path_obj.parent.name.upper()
        
        # Look for pattern like "B1" or "D8" in folder name
        match = re.search(r'([A-D])([1-8])', folder_name)
        if match:
            row = match.group(1)
            col = match.group(2)
            return f"{row}{col}"
        
        return None
