"""
Configuration settings for PCB Defect Detection System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"  # pcb_defect_detector/data/
PCB_DATA_DIR = DATA_DIR / "EBS7_TCM"

# Whole-PCB classification images (borned + corrosion full boards)
PCB_CLASSIFICATION_DIR = PROJECT_ROOT / "PCB"

# ── Vector database configurations ──────────────────────────────────────────
# Legacy single DB (kept for compatibility)
VECTOR_DB_CONFIG = {
    "persist_directory": str(BASE_DIR / "vector_db"),
    "collection_name": "pcb_defects",
    "distance_metric": "cosine"
}

# FIRST_VB: whole-PCB classification (burned vs corrosion)
# Built from: /PCB/borned/ + /PCB/corrosion/
FIRST_VDB_CONFIG = {
    "persist_directory": str(BASE_DIR / "vector_db" / "first_vb"),
    "collection_name": "first_vb",
}

# SECOND_VB: cell-level defect localization
# Built from: /NOT_OK/burned/ + /NOT_OK/corrosion/
SECOND_VDB_CONFIG = {
    "persist_directory": str(BASE_DIR / "vector_db" / "second_vb"),
    "collection_name": "second_vb",
}

# Model configuration
MODEL_CONFIG = {
    "clip_model": "openai/clip-vit-base-patch32",
    "embedding_dim": 512,
    "device": "auto"
}

# PCB Configuration
PCB_CONFIG = {
    "board_type": "EBS7_TCM",
    "image_extensions": [".jpg", ".jpeg", ".png", ".Jpeg"],
    "image_size": (224, 224)
}

# Defect detection configuration
DEFECT_CONFIG = {
    "similarity_threshold": 0.75,  # Threshold for determining if PCB is OK
    "top_k_results": 5,  # Number of similar images to retrieve
    "confidence_threshold": 0.85,  # High confidence threshold
    "min_match_similarity": 0.50,  # Minimum similarity to include defect match in results (0.0-1.0)
    "max_matches_per_cell": 50,  # Maximum database matches to search per grid cell
    "min_display_confidence": 0.91  # Only visualize/report defects above this confidence
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "enabled": True,  # ENABLED - AI-powered background removal for accurate cropping
    "min_area_ratio": 0.15,
    "padding": 10,
    "method": "ai",  # Use AI (rembg) for best results
    "min_margin_ratio": 0.005,  # More aggressive cropping (0.5% threshold)
    "use_original_on_failure": True
}

# Defect types
DEFECT_TYPES = {
    "burned": {
        "description": "Burned around soldering pads",
        "severity": "high",
        "keywords": ["burned", "fire", "charred", "scorched"]
    },
    "corrosion": {
        "description": "Corrosion on component or trace",
        "severity": "medium",
        "keywords": ["corrosion", "rust", "oxidation", "corroded"]
    },
    "missing": {
        "description": "Missing component",
        "severity": "high",
        "keywords": ["missing", "absent", "empty"]
    },
    "crack": {
        "description": "Crack in PCB or component",
        "severity": "medium",
        "keywords": ["crack", "fracture", "broken"]
    },
    "solder": {
        "description": "Soldering defect",
        "severity": "medium",
        "keywords": ["solder", "cold joint", "bridge"]
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": str(BASE_DIR / "logs" / "pcb_detector.log")
}
