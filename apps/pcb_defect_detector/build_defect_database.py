"""
Defect-Only Database Builder for PCB Inspection

This script builds a vector database containing ONLY defect examples (corrosion, burned).
New images are compared against these defects using CLIP similarity:
- Similarity < 0.8 (80%) = OK (no defect detected)
- Similarity >= 0.8 (80%) = DEFECT (corrosion or burned detected)
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
from config import PCB_DATA_DIR, MODEL_CONFIG, VECTOR_DB_CONFIG, PCB_CONFIG, PREPROCESSING_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefectDatabaseBuilder:
    """Build vector database with ONLY defect examples for similarity comparison"""
    
    def __init__(self):
        """Initialize defect database builder"""
        logger.info("Initializing Defect-Only Database Builder...")
        
        # Initialize components
        self.embedder = CLIPEmbedder(MODEL_CONFIG["clip_model"])
        self.vector_store = PCBVectorStore(
            persist_directory=VECTOR_DB_CONFIG["persist_directory"],
            collection_name=VECTOR_DB_CONFIG["collection_name"]
        )
        self.grid_segmenter = GridSegmenter()  # Uses default 4x4 grid
        self.pcb_extractor = PCBExtractor(
            min_area_ratio=PREPROCESSING_CONFIG.get("min_area_ratio", 0.15),
            padding=PREPROCESSING_CONFIG.get("padding", 10)
        )
        
        logger.info("Defect-Only Database Builder initialized")
    
    def build_database(self, data_dir: Path, reset: bool = False):
        """
        Build database from DEFECT images only
        
        Args:
            data_dir: Path to data directory containing NOT_OK folder
            reset: If True, reset the database before building
        """
        logger.info(f"Building defect database from: {data_dir}")
        logger.info(f"Reset database: {reset}")
        
        if reset:
            logger.warning("Resetting database...")
            self.vector_store.reset()
        
        # Process ONLY defect images from NOT_OK folder
        not_ok_dir = data_dir / "NOT_OK"
        
        if not not_ok_dir.exists():
            logger.error(f"NOT_OK directory not found: {not_ok_dir}")
            return
        
        logger.info("=" * 60)
        logger.info("Processing DEFECT images (corrosion, burned)...")
        logger.info("=" * 60)
        
        defect_count = 0
        
        # Process each defect type folder
        for defect_folder in sorted(not_ok_dir.iterdir()):
            if not defect_folder.is_dir():
                continue
            
            defect_type = defect_folder.name.lower()
            logger.info(f"\n  Defect type: {defect_type}")
            
            # Get all images
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(list(defect_folder.glob(ext)))
            
            logger.info(f"  Found {len(image_files)} images")
            
            # Process each defect image
            for img_path in tqdm(image_files, desc=f"Processing {defect_type}"):
                try:
                    # Load image
                    image = Image.open(img_path).convert("RGB")
                    
                    # Extract PCB region (crop background)
                    extracted_img, _ = self.pcb_extractor.extract_pcb(image, method='ai')
                    
                    # Segment into grid cells
                    cells = self.grid_segmenter.segment_image(extracted_img)
                    
                    # Try to extract grid location from filename
                    filename = img_path.stem
                    grid_location = self._extract_grid_location(filename)
                    
                    if grid_location and grid_location in cells:
                        # Store only the specific defect cell
                        cell_img = cells[grid_location]
                        embedding = self.embedder.encode_image(cell_img)
                        
                        metadata = {
                            "image_path": str(img_path),
                            "defect_type": defect_type,
                            "grid_location": grid_location,
                            "label": "defect",
                            "original_filename": filename
                        }
                        
                        self.vector_store.add_images(
                            embeddings=np.array([embedding]),
                            metadatas=[metadata],
                            image_paths=[str(img_path)]
                        )
                        defect_count += 1
                    else:
                        # If no grid location in filename, store whole extracted image
                        embedding = self.embedder.encode_image(extracted_img)
                        
                        metadata = {
                            "image_path": str(img_path),
                            "defect_type": defect_type,
                            "grid_location": "unknown",
                            "label": "defect",
                            "original_filename": filename
                        }
                        
                        self.vector_store.add_images(
                            embeddings=np.array([embedding]),
                            metadatas=[metadata],
                            image_paths=[str(img_path)]
                        )
                        defect_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Defect database build complete!")
        logger.info(f"   Total defect examples stored: {defect_count}")
        logger.info(f"   Similarity threshold: < 0.8 = OK, >= 0.8 = DEFECT")
        logger.info("=" * 60)
    
    def _extract_grid_location(self, filename: str) -> str:
        """
        Extract grid location from filename (e.g., 'burned_b1' -> 'B1')
        
        Args:
            filename: Image filename
            
        Returns:
            Grid location (e.g., 'B1') or None
        """
        import re
        
        # Pattern: letter followed by number (case insensitive)
        # Examples: b1, B1, d8, D8
        match = re.search(r'([a-dA-D])(\d)', filename)
        if match:
            letter = match.group(1).upper()
            number = match.group(2)
            return f"{letter}{number}"
        
        return None


def main():
    parser = argparse.ArgumentParser(description="Build defect-only vector database")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PCB_DATA_DIR,
        help="Path to data directory"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database before building"
    )
    
    args = parser.parse_args()
    
    builder = DefectDatabaseBuilder()
    builder.build_database(args.data_dir, reset=args.reset)


if __name__ == "__main__":
    main()
