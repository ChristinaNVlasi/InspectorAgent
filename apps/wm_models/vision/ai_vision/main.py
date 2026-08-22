"""
Main application entry point for AI Vision System
"""
import argparse
import logging
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config, LOGGING_CONFIG, PARTS_IMAGES_DIR
from embeddings.clip_embedder import CLIPEmbedder, ComponentEmbeddingDatabase
from preprocessing.image_processor import ImageProcessor
from utils.logger import setup_logging

def setup_system():
    """Initialize the AI vision system"""
    # Setup logging
    setup_logging(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Initializing AI Vision System for Alternator Components")
    
    # Load configurations
    model_config = get_config("model")
    data_config = get_config("data")
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Data config: {data_config}")
    
    return logger

def build_embeddings_database():
    """Build the embeddings database from component images"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("📋 Initializing CLIP embedder...")
        embedder = CLIPEmbedder()
        
        logger.info("🗄️ Building component database...")
        database = ComponentEmbeddingDatabase(embedder)
        
        # Build database
        save_path = Path(__file__).parent / "data" / "component_embeddings.pkl"
        save_path.parent.mkdir(exist_ok=True)
        
        database.build_database(save_path=str(save_path))
        
        logger.info("✅ Embeddings database built successfully!")
        return database
        
    except Exception as e:
        logger.error(f"❌ Error building database: {e}")
        raise

def test_system():
    """Test the system with sample images"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🧪 Testing system with sample images...")
        
        # Initialize components
        embedder = CLIPEmbedder()
        database = ComponentEmbeddingDatabase(embedder)
        
        # Load database if exists
        database_path = Path(__file__).parent / "data" / "component_embeddings.pkl"
        if database_path.exists():
            database.load_database(str(database_path))
            logger.info("📂 Loaded existing database")
        else:
            logger.info("📋 Building new database...")
            database.build_database(save_path=str(database_path))
        
        # Find a sample image to test
        from config import PARTS_IMAGES_DIR
        sample_images = list(PARTS_IMAGES_DIR.glob("**/*.Jpeg"))
        
        if sample_images:
            sample_image = sample_images[0]
            logger.info(f"🔍 Testing with sample image: {sample_image.name}")
            
            # Search for similar images
            results = database.search_similar(sample_image, top_k=3)
            
            logger.info(f"📊 Found {len(results)} similar images:")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                logger.info(f"  {i}. Similarity: {result['similarity']:.3f}")
                logger.info(f"     Component: {metadata['component_type']}")
                logger.info(f"     Condition: {metadata['condition']}")
                logger.info(f"     Path: {Path(result['image_path']).name}")
        
        logger.info("✅ System test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error testing system: {e}")
        raise



def run_web_interface():
    """Launch the web interface"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🌐 Starting web interface...")
        
        # Import and run Streamlit app
        import subprocess
        import sys
        
        streamlit_app = Path(__file__).parent / "web_ui" / "streamlit_app.py"
        
        if streamlit_app.exists():
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", str(streamlit_app)
            ])
        else:
            logger.error("❌ Streamlit app not found. Please create web_ui/streamlit_app.py")
            
    except Exception as e:
        logger.error(f"❌ Error starting web interface: {e}")
        raise

def main():
    """Main application function"""
    parser = argparse.ArgumentParser(
        description="AI Vision System for Alternator Component Analysis"
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "build", "test", "web", "all"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup system
    logger = setup_system()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == "setup":
            logger.info("🔧 System setup completed")
            
        elif args.command == "build":
            build_embeddings_database()
            
        elif args.command == "test":
            test_system()
            
        elif args.command == "web":
            run_web_interface()
            
        elif args.command == "all":
            logger.info("🚀 Running complete pipeline...")
            build_embeddings_database()
            test_system()
            logger.info("✅ Complete pipeline finished!")
            logger.info("💡 Run 'python main.py web' to start the web interface")
    
    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()