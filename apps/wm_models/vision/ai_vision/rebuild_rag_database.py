#!/usr/bin/env python3
"""
RAG Database Rebuild Script
Recreates the vector embeddings database for the RAG-based component inspector
Use this script after adding new images to the parts_images folders
"""
import sys
from pathlib import Path
import logging

# Add ai_vision to path
sys.path.insert(0, str(Path(__file__).parent))

from embeddings.clip_embedder import CLIPEmbedder
from models.rag_inspector import RAGComponentInspector
from config import PARTS_IMAGES_DIR, LOGGING_CONFIG
from utils.logger import setup_logging

def main():
    """Rebuild the RAG database with all current images"""
    
    # Setup logging
    setup_logging(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🔨 RAG DATABASE REBUILD TOOL")
    logger.info("=" * 80)
    logger.info("")
    
    # Check parts images directory
    if not PARTS_IMAGES_DIR.exists():
        logger.error(f"❌ Parts images directory not found: {PARTS_IMAGES_DIR}")
        return False
    
    logger.info(f"📁 Parts images directory: {PARTS_IMAGES_DIR}")
    logger.info("")
    
    # Count images before rebuild
    logger.info("📊 Scanning for Arcelik-Beko washing machine component images...")
    image_counts = {}
    total_images = 0
    
    # Updated for Arcelik-Beko washing machine components
    for folder_name in ["Cabinet_Panels_Damaged", "Detergent_Dispenser_Damaged", "Front_Wall_Damaged", "Scratches_General"]:
        folder_path = PARTS_IMAGES_DIR / folder_name
        if folder_path.exists():
            images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.Jpeg"))
            image_counts[folder_name] = len(images)
            total_images += len(images)
            logger.info(f"  ✓ {folder_name}: {len(images)} images")
        else:
            logger.warning(f"  ⚠️  {folder_name}: folder not found")
            image_counts[folder_name] = 0
    
    logger.info(f"")
    logger.info(f"📈 Total images found: {total_images}")
    logger.info("")
    
    if total_images == 0:
        logger.error("❌ No images found! Cannot rebuild database.")
        return False
    
    # Confirm rebuild
    logger.info("⚠️  This will RECREATE the entire RAG database.")
    logger.info("   All previous embeddings will be replaced with new ones.")
    logger.info("")
    
    try:
        # Initialize CLIP embedder
        logger.info("🔧 Step 1/3: Initializing CLIP embedder...")
        embedder = CLIPEmbedder()
        logger.info("   ✅ CLIP embedder loaded")
        logger.info("")
        
        # Initialize RAG inspector
        logger.info("🔧 Step 2/3: Building RAG component databases...")
        logger.info("   This may take several minutes depending on image count...")
        logger.info("")
        
        rag_inspector = RAGComponentInspector(embedder)
        rag_inspector.build_component_databases(PARTS_IMAGES_DIR)
        
        logger.info("")
        logger.info("   ✅ RAG databases built successfully!")
        logger.info("")
        
        # Save databases
        logger.info("🔧 Step 3/3: Saving databases to disk...")
        save_path = Path(__file__).parent / "data" / "rag_databases.pkl"
        save_path.parent.mkdir(exist_ok=True)
        
        rag_inspector.save_databases(str(save_path))
        logger.info(f"   ✅ Saved to: {save_path}")
        logger.info("")
        
        # Summary
        logger.info("=" * 80)
        logger.info("✅ RAG DATABASE REBUILD COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📊 Database Statistics:")
        
        for comp_type, db in rag_inspector.component_databases.items():
            logger.info(f"   • {comp_type.upper()}: {len(db['embeddings'])} reference images embedded")
        
        logger.info("")
        logger.info(f"💾 Database file: {save_path}")
        logger.info(f"📦 File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info("")
        logger.info("🚀 You can now restart the BORG system to use the updated database")
        logger.info("   Run: ./start_borg_system.sh")
        logger.info("")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("⚠️  Rebuild interrupted by user")
        return False
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Error during rebuild: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
