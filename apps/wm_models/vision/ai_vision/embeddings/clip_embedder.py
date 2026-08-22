"""
CLIP-based image embedding system for alternator components
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
import numpy as np
from typing import List, Union, Tuple, Optional
from pathlib import Path
import json
import pickle
import logging
from tqdm import tqdm

from config import MODEL_CONFIG, PARTS_IMAGES_DIR, COMPONENT_MAPPING

logger = logging.getLogger(__name__)

class CLIPEmbedder:
    """CLIP-based image embedder for component similarity search"""
    
    def __init__(self, 
                 model_name: str = MODEL_CONFIG["clip_model"],
                 device: str = "auto"):
        """
        Initialize CLIP embedder
        
        Args:
            model_name: CLIP model name from Hugging Face
            device: Device to run model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_model = self.model.vision_model
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded CLIP model: {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def encode_image(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """
        Encode single image to embedding vector
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Normalized embedding vector
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # Extract features - handle both Tensor and ModelOutput
        # In newer transformers versions, get_image_features returns BaseModelOutputWithPooling
        if isinstance(outputs, torch.Tensor):
            # Direct tensor (older versions or different model)
            image_features = outputs
        else:
            # Model output object - use the pooled features for CLIP
            image_features = outputs[0] if not hasattr(outputs, 'pooler_output') else outputs.pooler_output
            
        # Convert to numpy and normalize
        embedding = image_features.cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        return embedding.squeeze()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector
        
        Args:
            text: Text description
            
        Returns:
            Normalized embedding vector
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Normalize embedding
        embedding = text_features.cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        return embedding.squeeze()
    
    def encode_batch_images(self, 
                          images: List[Union[Image.Image, str, Path]],
                          batch_size: int = 16) -> np.ndarray:
        """
        Encode batch of images efficiently
        
        Args:
            images: List of PIL Images or image paths
            batch_size: Processing batch size
            
        Returns:
            Array of normalized embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i:i + batch_size]
            
            # Load and prepare batch
            pil_images = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    pil_images.append(Image.open(img).convert('RGB'))
                else:
                    pil_images.append(img)
            
            # Process batch
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # Extract features - handle both Tensor and ModelOutput
            if isinstance(outputs, torch.Tensor):
                # Direct tensor (older versions or different model)
                batch_features = outputs
            else:
                # Model output object - use the pooled features for CLIP
                batch_features = outputs[0] if not hasattr(outputs, 'pooler_output') else outputs.pooler_output
            
            # Convert to numpy and normalize
            batch_embeddings = batch_features.cpu().numpy()
            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings, axis=1, keepdims=True
            )
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embedding1, embedding2: Normalized embedding vectors
            
        Returns:
            Cosine similarity score (0-1)
        """
        return np.dot(embedding1, embedding2)
    
    def find_similar_images(self, 
                           query_embedding: np.ndarray,
                           database_embeddings: np.ndarray,
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar images in database
        
        Args:
            query_embedding: Query image embedding
            database_embeddings: Database of image embeddings
            top_k: Number of similar images to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        # Compute similarities
        similarities = np.dot(database_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return indices with scores
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results

class ComponentEmbeddingDatabase:
    """Database for storing and managing component embeddings"""
    
    def __init__(self, embedder: CLIPEmbedder):
        self.embedder = embedder
        self.embeddings = []
        self.metadata = []
        self.image_paths = []
    
    def build_database(self, 
                      parts_images_dir: Path = PARTS_IMAGES_DIR,
                      save_path: Optional[str] = None) -> None:
        """
        Build embedding database from parts images
        
        Args:
            parts_images_dir: Directory containing component images
            save_path: Path to save the database
        """
        logger.info("Building component embedding database...")
        
        # Collect all image paths with metadata
        image_data = self._collect_image_metadata(parts_images_dir)
        
        # Extract image paths
        image_paths = [item['path'] for item in image_data]
        
        # Generate embeddings
        embeddings = self.embedder.encode_batch_images(image_paths)
        
        # Store data
        self.embeddings = embeddings
        self.metadata = [item['metadata'] for item in image_data]
        self.image_paths = image_paths
        
        logger.info(f"Built database with {len(self.embeddings)} images")
        
        # Save if path provided
        if save_path:
            self.save_database(save_path)
    
    def _collect_image_metadata(self, parts_dir: Path) -> List[dict]:
        """
        Collect all images with metadata from parts directory
        
        Args:
            parts_dir: Parts images directory
            
        Returns:
            List of image data with metadata
        """
        image_data = []
        
        for component_dir in parts_dir.iterdir():
            if not component_dir.is_dir():
                continue
            
            # Get component info
            component_name = component_dir.name
            component_info = COMPONENT_MAPPING.get(component_name, {
                "type": "unknown",
                "condition": "unknown",
                "description": f"Unknown component: {component_name}"
            })
            
            # Process all images in component directory
            for image_path in component_dir.glob("*.Jpeg"):
                # Parse filename for additional metadata
                filename = image_path.stem
                parts = filename.split('_')
                
                metadata = {
                    "component_type": component_info["type"],
                    "condition": component_info["condition"],
                    "description": component_info["description"],
                    "component_dir": component_name,
                    "filename": filename,
                    "image_path": str(image_path)
                }
                
                # Extract additional info from filename if possible
                if len(parts) >= 3:
                    metadata.update({
                        "sample_id": parts[0],
                        "core_id": parts[1] if len(parts) > 1 else "",
                        "line_info": parts[2] if len(parts) > 2 else ""
                    })
                
                image_data.append({
                    "path": image_path,
                    "metadata": metadata
                })
        
        return image_data
    
    def search_similar(self, 
                      query_image: Union[Image.Image, str, Path],
                      top_k: int = 5,
                      component_filter: Optional[str] = None) -> List[dict]:
        """
        Search for similar images in database
        
        Args:
            query_image: Query image or path
            top_k: Number of results to return
            component_filter: Filter by component type
            
        Returns:
            List of similar image results with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_image(query_image)
        
        # Filter embeddings if component filter specified
        if component_filter:
            filtered_indices = [
                i for i, meta in enumerate(self.metadata)
                if meta["component_type"] == component_filter
            ]
            filtered_embeddings = self.embeddings[filtered_indices]
        else:
            filtered_indices = list(range(len(self.embeddings)))
            filtered_embeddings = self.embeddings
        
        # Find similar images
        similar_results = self.embedder.find_similar_images(
            query_embedding, filtered_embeddings, top_k
        )
        
        # Prepare results with metadata
        results = []
        for filtered_idx, similarity in similar_results:
            original_idx = filtered_indices[filtered_idx]
            result = {
                "similarity": similarity,
                "image_path": self.image_paths[original_idx],
                "metadata": self.metadata[original_idx]
            }
            results.append(result)
        
        return results
    
    def save_database(self, save_path: str) -> None:
        """Save database to disk"""
        database_data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "image_paths": self.image_paths,
            "model_name": self.embedder.model_name
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(database_data, f)
        
        logger.info(f"Saved database to {save_path}")
    
    def load_database(self, load_path: str) -> None:
        """Load database from disk"""
        with open(load_path, 'rb') as f:
            database_data = pickle.load(f)
        
        self.embeddings = database_data["embeddings"]
        self.metadata = database_data["metadata"]
        self.image_paths = database_data["image_paths"]
        
        logger.info(f"Loaded database from {load_path}")

# Text descriptions for component analysis
COMPONENT_DESCRIPTIONS = {
    "pulley": [
        "healthy alternator pulley component",
        "good condition pulley wheel",
        "intact alternator pulley",
        "functioning pulley mechanism"
    ],
    "casting": [
        "broken alternator casting",
        "damaged metal housing",
        "cracked casting component",
        "fractured alternator case"
    ],
    "cover": [
        "broken alternator cover",
        "damaged protective cover",
        "cracked cover plate",
        "fractured alternator housing"
    ]
}

def create_text_embeddings(embedder: CLIPEmbedder) -> dict:
    """Create text embeddings for component descriptions"""
    text_embeddings = {}
    
    for component, descriptions in COMPONENT_DESCRIPTIONS.items():
        component_embeddings = []
        for desc in descriptions:
            embedding = embedder.encode_text(desc)
            component_embeddings.append(embedding)
        
        # Average embeddings for component
        avg_embedding = np.mean(component_embeddings, axis=0)
        text_embeddings[component] = avg_embedding
    
    return text_embeddings

# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = CLIPEmbedder()
    
    # Create database
    database = ComponentEmbeddingDatabase(embedder)
    
    # Build database (this would process all images)
    print("Building embedding database...")
    database.build_database(save_path="component_embeddings.pkl")
    
    # Example search
    query_image = PARTS_IMAGES_DIR / "A1 - Pulley" / "303944_RS-CORE-015284_Line30_A1.Jpeg"
    if query_image.exists():
        results = database.search_similar(query_image, top_k=3)
        print(f"Found {len(results)} similar images")
        for i, result in enumerate(results):
            print(f"{i+1}. Similarity: {result['similarity']:.3f}, "
                  f"Component: {result['metadata']['component_type']}")
    
    print("CLIP embedding system ready!")