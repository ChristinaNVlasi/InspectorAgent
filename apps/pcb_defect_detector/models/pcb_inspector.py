"""
PCB Inspector - Main defect detection system using RAG with grid visualization
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
from PIL import Image
import io
import base64

from embeddings.clip_embedder import CLIPEmbedder
from rag.vector_store import PCBVectorStore
from config import MODEL_CONFIG, VECTOR_DB_CONFIG, DEFECT_CONFIG, PREPROCESSING_CONFIG
from utils.grid_mapper import PCBGridMapper
from preprocessing.pcb_extractor import PCBExtractor

logger = logging.getLogger(__name__)

class PCBInspector:
    """RAG-based PCB defect inspector with grid visualization"""
    
    def __init__(self):
        """Initialize PCB inspector with embedder and vector store"""
        logger.info("Initializing PCB Inspector...")
        
        # Initialize CLIP embedder
        self.embedder = CLIPEmbedder(
            model_name=MODEL_CONFIG["clip_model"],
            device=MODEL_CONFIG["device"]
        )
        
        # Initialize vector store
        self.vector_store = PCBVectorStore(
            persist_directory=VECTOR_DB_CONFIG["persist_directory"],
            collection_name=VECTOR_DB_CONFIG["collection_name"]
        )
        
        # Configuration
        self.similarity_threshold = DEFECT_CONFIG["similarity_threshold"]
        self.top_k = DEFECT_CONFIG["top_k_results"]
        
        # Initialize grid mapper for defect localization
        self.grid_mapper = PCBGridMapper()
        
        # Initialize PCB extractor for automatic cropping
        self.preprocessing_enabled = PREPROCESSING_CONFIG["enabled"]
        if self.preprocessing_enabled:
            self.pcb_extractor = PCBExtractor(
                min_area_ratio=PREPROCESSING_CONFIG["min_area_ratio"],
                padding=PREPROCESSING_CONFIG["padding"]
            )
            logger.info("PCB extractor initialized for automatic cropping")
        else:
            self.pcb_extractor = None
        
        logger.info("PCB Inspector initialized successfully")
    
    def inspect_pcb(self, image_path: str) -> Dict[str, Any]:
        """
        Inspect a PCB image for defects using RAG similarity search
        
        Args:
            image_path: Path to PCB image
            
        Returns:
            Inspection results with status, defect info, and similar images
        """
        logger.info(f"Inspecting PCB image: {image_path}")
        
        # Load image
        original_image = Image.open(image_path)
        
        # Crop to PCB area if preprocessing is enabled
        processed_image = original_image
        crop_info = None
        
        if self.preprocessing_enabled and self.pcb_extractor:
            try:
                processed_image, crop_info = self.pcb_extractor.extract_pcb(
                    original_image,
                    method=PREPROCESSING_CONFIG["method"]
                )
                
                if crop_info.get('detected', False):
                    logger.info(f"PCB cropped: {crop_info.get('message', 'Success')}")
                else:
                    logger.warning("PCB detection failed, using original image")
                    processed_image = original_image
                    
            except Exception as e:
                logger.error(f"PCB extraction failed: {e}", exc_info=True)
                processed_image = original_image
                crop_info = {"detected": False, "error": str(e)}
        
        # Save processed image temporarily for embedding
        import tempfile
        temp_path = None
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
                processed_image.save(temp_path)
            
            # Generate embedding from PROCESSED/CROPPED image
            query_embedding = self.embedder.encode_image(temp_path)
        finally:
            # Clean up temp file
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()
        
        # Search for similar images in database
        # First, check against OK images
        ok_results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=self.top_k,
            where={"status": "OK"}
        )
        
        # Then check against defect images
        defect_results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=self.top_k,
            where={"status": "NOT_OK"}
        )
        
        # Analyze results with visualization (pass processed image)
        result = self._analyze_results(ok_results, defect_results, processed_image, crop_info)
        
        logger.info(f"Inspection complete: Status={result['status']}")
        
        return result
    
    def _analyze_results(self, ok_results: Dict, defect_results: Dict, processed_image: Image.Image, crop_info: Optional[Dict]) -> Dict[str, Any]:
        """
        Analyze similarity search results to determine defect status
        
        Args:
            ok_results: Results from OK images search
            defect_results: Results from defect images search
            processed_image: PIL Image (cropped/processed) for visualization
            crop_info: Information about cropping operation
            
        Returns:
            Analysis results with status and defect information
        """
        # Calculate similarity scores (ChromaDB returns distances, convert to similarity)
        # Distance = 0 means identical, higher distance = less similar
        # Similarity = 1 - (distance / 2) for cosine distance
        
        ok_distances = ok_results['distances'][0] if ok_results['distances'] else []
        defect_distances = defect_results['distances'][0] if defect_results['distances'] else []
        
        # Convert distances to similarities
        ok_similarities = [1 - (d / 2) for d in ok_distances] if ok_distances else []
        defect_similarities = [1 - (d / 2) for d in defect_distances] if defect_distances else []
        
        # Get best matches
        best_ok_similarity = max(ok_similarities) if ok_similarities else 0
        best_defect_similarity = max(defect_similarities) if defect_similarities else 0
        
        # Check if image is not a PCB (both similarities below 55%)
        MIN_PCB_CONFIDENCE = 0.55
        if best_ok_similarity < MIN_PCB_CONFIDENCE and best_defect_similarity < MIN_PCB_CONFIDENCE:
            status = "INVALID_PCB"
            confidence = max(best_ok_similarity, best_defect_similarity)
            defect_info = {
                "defect_type": "invalid_image",
                "location": "n/a",
                "description": "This image does not appear to be a PCB for analysis. Please upload a valid PCB image."
            }
            similar_images = []
        # Determine status based on similarity comparison
        elif best_ok_similarity > best_defect_similarity and best_ok_similarity >= self.similarity_threshold:
            status = "OK"
            confidence = best_ok_similarity
            defect_info = None
            similar_images = self._format_results(ok_results, ok_similarities)
        else:
            status = "NOT_OK"
            confidence = best_defect_similarity
            
            # Extract defect information from best match
            if defect_results['metadatas'] and defect_results['metadatas'][0]:
                best_defect_meta = defect_results['metadatas'][0][0]
                defect_info = {
                    "defect_type": best_defect_meta.get("defect_type", "unknown"),
                    "location": best_defect_meta.get("location", "unknown"),
                    "description": best_defect_meta.get("description", "Defect detected")
                }
            else:
                defect_info = {
                    "defect_type": "unknown",
                    "location": "unknown", 
                    "description": "Defect detected but type unknown"
                }
            
            similar_images = self._format_results(defect_results, defect_similarities)
        
        # Generate marked and zoomed images for defects
        marked_image_data = None
        zoomed_image_data = None
        view_info = None
        cropped_image_data = self._image_to_base64(processed_image)  # Always return cropped image
        
        if status == "NOT_OK" and defect_info and defect_info.get("location") not in ["unknown", "none"]:
            try:
                location = defect_info["location"]
                
                # Use processed/cropped image for marking
                # Mark defect area on processed image
                marked_img = self.grid_mapper.mark_defect_area(
                    processed_image,
                    location,
                    label=defect_info.get("defect_type", "DEFECT").upper()
                )
                marked_image_data = self._image_to_base64(marked_img)
                
                # Create zoomed view of defect area
                zoomed_img = self.grid_mapper.zoom_to_area(
                    processed_image,
                    location,
                    zoom_factor=1.5
                )
                if zoomed_img:
                    zoomed_image_data = self._image_to_base64(zoomed_img)
                
                # Get view information
                view_info = self.grid_mapper.get_view_info(location)
                
            except Exception as e:
                logger.error(f"Error generating marked images: {e}", exc_info=True)
        
        return {
            "status": status,
            "confidence": round(confidence, 4),
            "defect_info": defect_info,
            "similar_images": similar_images,
            "cropped_image": cropped_image_data,  # Cropped PCB image
            "marked_image": marked_image_data,
            "zoomed_image": zoomed_image_data,
            "view_info": view_info,
            "crop_info": crop_info,  # Info about cropping operation
            "analysis": {
                "best_ok_similarity": round(best_ok_similarity, 4),
                "best_defect_similarity": round(best_defect_similarity, 4),
                "threshold": self.similarity_threshold
            }
        }
    
    def _format_results(self, results: Dict, similarities: List[float]) -> List[Dict[str, Any]]:
        """Format search results for output"""
        formatted = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted
        
        for i, img_id in enumerate(results['ids'][0]):
            formatted.append({
                "id": img_id,
                "similarity": round(similarities[i], 4),
                "image_path": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        return formatted
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        return self.vector_store.get_stats()
    
    def _image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string for embedding in JSON/HTML
        
        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"
