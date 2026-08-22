"""
Enhanced Database Builder for Grid-Based PCB Defect Detection

This script builds a vector database with grid-cell level embeddings for precise
defect localization. Each defect image is segmented into grid cells, and embeddings
are stored with location metadata (A1-D8).
"""
import sys
from pathlib import Path
import logging
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from embeddings.clip_embedder import CLIPEmbedder
from rag.vector_store import PCBVectorStore
from utils.grid_segmenter import GridSegmenter
from preprocessing.pcb_extractor import PCBExtractor
from config import (
    PCB_DATA_DIR, 
    MODEL_CONFIG, 
    VECTOR_DB_CONFIG, 
    PCB_CONFIG,
    PREPROCESSING_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridBasedDatabaseBuilder:
    """Build vector database with grid-cell level embeddings"""
    
    def __init__(self, enable_grid_segmentation: bool = True):
        """
        Initialize database builder
        
        Args:
            enable_grid_segmentation: If True, segment images into grid cells;
                                     if False, use whole-image embeddings
        """
        logger.info("Initializing Grid-Based Database Builder...")
        
        self.enable_grid_segmentation = enable_grid_segmentation
        
        # Initialize components
        self.embedder = CLIPEmbedder(
            model_name=MODEL_CONFIG["clip_model"],
            device=MODEL_CONFIG["device"]
        )
        
        self.vector_store = PCBVectorStore(
            persist_directory=VECTOR_DB_CONFIG["persist_directory"],
            collection_name=VECTOR_DB_CONFIG["collection_name"]
        )
        
        if enable_grid_segmentation:
            self.grid_segmenter = GridSegmenter()
            self.pcb_extractor = PCBExtractor(
                min_area_ratio=PREPROCESSING_CONFIG["min_area_ratio"],
                padding=PREPROCESSING_CONFIG["padding"]
            )
            logger.info("Grid segmentation ENABLED - will process at cell level")
        else:
            self.grid_segmenter = None
            self.pcb_extractor = None
            logger.info("Grid segmentation DISABLED - using whole-image processing")
    
    def build_database(self, data_dir: Path, reset: bool = False):
        """
        Build vector database from PCB images
        
        Args:
            data_dir: Directory containing OK/ and NOT_OK/ folders
            reset: If True, clear existing database
        """
        logger.info(f"Building database from: {data_dir}")
        logger.info(f"Reset database: {reset}")
        logger.info(f"Grid segmentation: {self.enable_grid_segmentation}")
        
        if reset:
            logger.warning("Resetting database...")
            self.vector_store.reset()
        
        # Process OK images
        ok_dir = data_dir / "OK"
        if ok_dir.exists():
            logger.info(f"\n{'='*60}")
            logger.info("Processing OK images...")
            logger.info(f"{'='*60}")
            self._process_directory(ok_dir, status="OK")
        else:
            logger.warning(f"OK directory not found: {ok_dir}")
        
        # Process NOT_OK images (defects)
        not_ok_dir = data_dir / "NOT_OK"
        if not_ok_dir.exists():
            logger.info(f"\n{'='*60}")
            logger.info("Processing NOT_OK images (defects)...")
            logger.info(f"{'='*60}")
            
            # Process each defect type folder
            for defect_folder in not_ok_dir.iterdir():
                if defect_folder.is_dir() and not defect_folder.name.startswith('.'):
                    logger.info(f"\n  Defect folder: {defect_folder.name}")
                    self._process_directory(
                        defect_folder,
                        status="NOT_OK",
                        defect_folder_name=defect_folder.name
                    )
        else:
            logger.warning(f"NOT_OK directory not found: {not_ok_dir}")
        
        # Log final stats
        total_count = self.vector_store.collection.count()
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Database build complete!")
        logger.info(f"   Total entries: {total_count}")
        logger.info(f"   Grid mode: {self.enable_grid_segmentation}")
        logger.info(f"{'='*60}")
    
    def _process_directory(
        self, 
        directory: Path,
        status: str,
        defect_folder_name: str = None
    ):
        """
        Process all images in a directory
        
        Args:
            directory: Directory containing images
            status: "OK" or "NOT_OK"
            defect_folder_name: Name of defect folder (for NOT_OK images)
        """
        # Find all image files
        image_extensions = PCB_CONFIG["image_extensions"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in {directory}")
            return
        
        logger.info(f"Found {len(image_files)} images")
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"Processing {directory.name}"):
            try:
                self._process_image(
                    image_path,
                    status=status,
                    defect_folder_name=defect_folder_name
                )
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}", exc_info=True)
    
    def _process_image(
        self,
        image_path: Path,
        status: str,
        defect_folder_name: str = None
    ):
        """
        Process a single image: segment into grid cells and create embeddings
        
        Args:
            image_path: Path to image file
            status: "OK" or "NOT_OK"
            defect_folder_name: Defect folder name (e.g., "burned_b1", "CORROSION_D8")
        """
        # Load image
        image = Image.open(image_path)
        
        # Extract defect metadata
        defect_type, defect_location, severity = self._parse_defect_metadata(
            defect_folder_name
        )
        
        if self.enable_grid_segmentation:
            # GRID-BASED PROCESSING
            self._process_image_grid_based(
                image,
                image_path,
                status,
                defect_type,
                defect_location,
                severity
            )
        else:
            # WHOLE-IMAGE PROCESSING (legacy)
            self._process_image_whole(
                image,
                image_path,
                status,
                defect_type,
                defect_location,
                severity
            )
    
    def _process_image_grid_based(
        self,
        image: Image.Image,
        image_path: Path,
        status: str,
        defect_type: str,
        defect_location: str,
        severity: str
    ):
        """
        Process image with grid segmentation
        
        For OK images: Segment into 16 cells, embed each
        For defect images: Segment into 16 cells, embed defect cell
        
        Args:
            image: PIL Image
            image_path: Image file path
            status: OK or NOT_OK
            defect_type: Type of defect (if NOT_OK)
            defect_location: Grid location of defect (e.g., 'B1')
            severity: Defect severity
        """
        # Preprocess: remove background
        if self.pcb_extractor and PREPROCESSING_CONFIG["enabled"]:
            try:
                processed_image, crop_info = self.pcb_extractor.extract_pcb(
                    image,
                    method=PREPROCESSING_CONFIG["method"]
                )
                if crop_info.get('detected', False):
                    image = processed_image
            except Exception as e:
                logger.warning(f"Background removal failed for {image_path}: {e}")
        
        # Determine view type based on defect location or image aspect ratio
        view_type = 'auto'
        if defect_location:
            # If defect is in columns 1-4, it's left view
            # If defect is in columns 5-8, it's right view
            col = int(defect_location[1:]) if len(defect_location) > 1 else 1
            view_type = 'left' if col <= 4 else 'right'
        
        # Segment into grid cells
        cell_images = self.grid_segmenter.segment_image(image, view_type)
        
        # Process cells based on status
        if status == "OK":
            # For OK images: embed ALL cells (to build healthy baseline)
            for grid_coord, cell_image in cell_images.items():
                self._add_cell_to_database(
                    cell_image=cell_image,
                    image_path=image_path,
                    grid_coord=grid_coord,
                    status="OK",
                    defect_type=None,
                    severity=None
                )
        else:
            # For defect images: embed ONLY the defective cell
            if defect_location and defect_location in cell_images:
                self._add_cell_to_database(
                    cell_image=cell_images[defect_location],
                    image_path=image_path,
                    grid_coord=defect_location,
                    status="NOT_OK",
                    defect_type=defect_type,
                    severity=severity
                )
            else:
                logger.warning(
                    f"Defect location {defect_location} not found in cells for {image_path}"
                )
    
    def _process_image_whole(
        self,
        image: Image.Image,
        image_path: Path,
        status: str,
        defect_type: str,
        defect_location: str,
        severity: str
    ):
        """
        Process whole image without grid segmentation (legacy mode)
        
        Args:
            image: PIL Image
            image_path: Image file path
            status: OK or NOT_OK
            defect_type: Type of defect
            defect_location: Grid location (stored but not used for segmentation)
            severity: Defect severity
        """
        # Generate embedding for whole image
        embedding = self.embedder.encode_image(str(image_path))
        
        # Prepare metadata
        metadata = {
            "status": status,
            "filename": image_path.name,
            "path": str(image_path),
            "location": defect_location if defect_location else "unknown"
        }
        
        if status == "NOT_OK":
            metadata.update({
                "defect_type": defect_type,
                "severity": severity,
                "description": f"{defect_type} defect at {defect_location}"
            })
        
        # Add to database
        self.vector_store.add_images(
            embeddings=np.array([embedding]),
            metadatas=[metadata],
            image_paths=[str(image_path)],
            ids=[f"img_{image_path.stem}"]
        )
    
    def _add_cell_to_database(
        self,
        cell_image: Image.Image,
        image_path: Path,
        grid_coord: str,
        status: str,
        defect_type: str = None,
        severity: str = None
    ):
        """
        Add a single grid cell to the database
        
        Args:
            cell_image: PIL Image of the grid cell
            image_path: Original image path
            grid_coord: Grid coordinate (e.g., 'B1')
            status: OK or NOT_OK
            defect_type: Type of defect (if NOT_OK)
            severity: Defect severity (if NOT_OK)
        """
        import tempfile
        
        # Save cell to temporary file for embedding
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
                cell_image.save(temp_path)
            
            # Generate embedding
            embedding = self.embedder.encode_image(temp_path)
            
        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()
        
        # Prepare metadata
        metadata = {
            "status": status,
            "location": grid_coord,
            "source_image": str(image_path),
            "filename": image_path.name,
        }
        
        if status == "NOT_OK":
            metadata.update({
                "defect_type": defect_type or "unknown",
                "severity": severity or "medium",
                "description": f"{defect_type or 'defect'} at {grid_coord}"
            })
        
        # Create unique ID
        cell_id = f"{image_path.stem}_{grid_coord}"
        
        # Add to database
        self.vector_store.add_images(
            embeddings=np.array([embedding]),
            metadatas=[metadata],
            image_paths=[str(image_path)],
            ids=[cell_id]
        )
    
    def _parse_defect_metadata(
        self, 
        defect_folder_name: str
    ) -> tuple[str, str, str]:
        """
        Parse defect metadata from folder name
        
        Examples:
            - "burned_b1" → ("burned", "B1", "high")
            - "CORROSION_D8" → ("corrosion", "D8", "medium")
        
        Args:
            defect_folder_name: Name of defect folder
            
        Returns:
            (defect_type, location, severity)
        """
        if not defect_folder_name:
            return ("unknown", None, "medium")
        
        # Extract location using grid segmenter utility
        location = GridSegmenter.parse_defect_location_from_path(
            f"/fake/path/{defect_folder_name}/image.png"
        )
        
        # Extract defect type from folder name
        folder_lower = defect_folder_name.lower()
        
        # Map defect types
        defect_type_map = {
            'burned': ('burned', 'high'),
            'burn': ('burned', 'high'),
            'fire': ('burned', 'high'),
            'corrosion': ('corrosion', 'medium'),
            'corroded': ('corrosion', 'medium'),
            'rust': ('corrosion', 'medium'),
            'missing': ('missing', 'high'),
            'absent': ('missing', 'high'),
            'crack': ('crack', 'medium'),
            'fracture': ('crack', 'medium'),
            'solder': ('solder', 'medium'),
            'bridge': ('solder', 'medium'),
            'short': ('short', 'high'),
        }
        
        defect_type = "unknown"
        severity = "medium"
        
        for keyword, (dtype, sev) in defect_type_map.items():
            if keyword in folder_lower:
                defect_type = dtype
                severity = sev
                break
        
        return (defect_type, location, severity)


def main():
    """Main entry point for database building"""
    parser = argparse.ArgumentParser(
        description="Build vector database for grid-based PCB defect detection"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(PCB_DATA_DIR),
        help='Directory containing OK/ and NOT_OK/ folders'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database before building'
    )
    parser.add_argument(
        '--no-grid',
        action='store_true',
        help='Disable grid segmentation (use whole-image embeddings)'
    )
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = GridBasedDatabaseBuilder(
        enable_grid_segmentation=not args.no_grid
    )
    
    # Build database
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    builder.build_database(data_dir, reset=args.reset)


if __name__ == "__main__":
    main()
