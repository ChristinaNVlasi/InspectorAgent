"""
Configuration settings for AI Vision System
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent  # Go up to rEUman-ML root
PARTS_IMAGES_DIR = BASE_DIR.parent / "parts_images"  # ai_vision/../parts_images

# Data configuration
DATA_CONFIG = {
    "image_extensions": [".jpeg", ".jpg", ".png", ".Jpeg"],
    "image_size": (224, 224),
    "batch_size": 32,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
}

# Component mapping - Arcelik-Beko Washing Machine Parts
COMPONENT_MAPPING = {
    "Cabinet_Panels_Damaged": {
        "type": "cabinet_panel", 
        "condition": "damaged",
        "description": "Damaged cabinet side panel - scratches, dents, or structural damage"
    },
    "Detergent_Dispenser_Damaged": {
        "type": "detergent_dispenser", 
        "condition": "damaged",
        "description": "Damaged detergent dispenser - rust, corrosion, or mechanical damage"
    },
    "Front_Wall_Damaged": {
        "type": "front_wall", 
        "condition": "damaged",
        "description": "Damaged front wall/door assembly - dents, scratches, or panel damage"
    },
    "Scratches_General": {
        "type": "general_surface", 
        "condition": "damaged",
        "description": "General surface scratches and cosmetic damage on various parts"
    }
}

# Model configuration
MODEL_CONFIG = {
    "clip_model": "openai/clip-vit-base-patch32",
    "vit_model": "google/vit-base-patch16-224",
    "clipseg_model": "CIDAS/clipseg-rd64-refined",  # CLIPSeg for segmentation
    "embedding_dim": 768,
    "num_classes": 3,  # ok, damaged, missing
    "learning_rate": 1e-4,
    "epochs": 50,
    "patience": 10,
    "device": "auto"  # Auto-detect device (CUDA if available, else CPU)
}

# Segmentation configuration
SEGMENTATION_CONFIG = {
    "model_name": "CIDAS/clipseg-rd64-refined",
    "threshold": 0.5,  # Confidence threshold for detection (0.5 = balanced, increase to reduce false positives)
    "alpha": 0.6,  # Transparency for overlay visualization (higher = more visible)
    "output_dir": str(BASE_DIR / "data" / "segmentation_results"),
    "defect_prompts": {
        "cabinet_panel": [
            "damaged area",
            "scratch",
            "dent",
            "broken panel",
            "structural damage",
            "paint damage",
            "surface damage"
        ],
        "detergent_dispenser": [
            "rust",
            "corrosion",
            "damaged screw",
            "broken dispenser",
            "oxidation",
            "mechanical damage",
            "wear"
        ],
        "front_wall": [
            "damaged area",
            "dent",
            "scratch",
            "broken panel",
            "door damage",
            "structural defect",
            "deformation"
        ],
        "general_surface": [
            "scratch",
            "surface damage",
            "cosmetic damage",
            "paint scratch",
            "minor dent",
            "wear mark"
        ],
        "generic": [
            "damaged area",
            "scratch",
            "dent",
            "broken part",
            "corrosion",
            "rust",
            "wear and tear",
            "surface damage"
        ]
    },
    "severity_thresholds": {
        "critical": {"area_ratio": 0.3, "confidence": 0.8},
        "moderate": {"area_ratio": 0.15, "confidence": 0.6},
        "minor": {"area_ratio": 0.05, "confidence": 0.4}
    }
}

# Defect detection configuration (for compatibility with global MCP server)
DEFECT_CONFIG = {
    "similarity_threshold": 0.75,  # Threshold for determining if component is OK
    "top_k_results": 5,  # Number of similar images to retrieve
    "confidence_threshold": 0.85  # High confidence threshold
}

# Preprocessing configuration (for compatibility with global MCP server)
PREPROCESSING_CONFIG = {
    "enabled": True,
    "min_area_ratio": 0.15,
    "padding": 10,
    "method": "ai",
    "min_margin_ratio": 0.005,
    "use_original_on_failure": True
}

# Vector database configuration
VECTOR_DB_CONFIG = {
    "collection_name": "alternator_components",
    "distance_metric": "cosine",
    "top_k_similar": 5,
    "similarity_threshold": 0.7,
    "persist_directory": str(BASE_DIR / "data" / "vector_db")
}

# RAG configuration
RAG_CONFIG = {
    "max_context_length": 4000,
    "temperature": 0.1,
    "max_tokens": 1000,
    "prompt_template_path": str(BASE_DIR / "rag" / "prompt_templates.py")
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".Jpeg"}
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "rotation": "1 week",
    "retention": "1 month"
}

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Device configuration
DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration for specific section or all configs"""
    configs = {
        "data": DATA_CONFIG,
        "model": MODEL_CONFIG,
        "segmentation": SEGMENTATION_CONFIG,
        "vector_db": VECTOR_DB_CONFIG,
        "rag": RAG_CONFIG,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "component_mapping": COMPONENT_MAPPING
    }
    
    if section:
        return configs.get(section, {})
    return configs