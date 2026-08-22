"""
CLIP-based image embedding system for PCB defect detection
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CLIPEmbedder:
    """CLIP-based image embedder for PCB similarity search"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        """
        Initialize CLIP embedder
        
        Args:
            model_name: CLIP model name from Hugging Face
            device: Device to run model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # Load CLIP model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"CLIP model loaded successfully on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
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
            # Handle model output object - extract the actual tensor
            if hasattr(outputs, 'image_embeds'):
                image_features = outputs.image_embeds
            elif hasattr(outputs, 'pooler_output'):
                image_features = outputs.pooler_output
            else:
                # Fallback: assume outputs is the tensor itself
                image_features = outputs
            
        # Normalize embedding
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
    
    def encode_batch(self, images: List[Union[Image.Image, str, Path]]) -> np.ndarray:
        """
        Encode batch of images
        
        Args:
            images: List of PIL Images or image paths
            
        Returns:
            Array of normalized embeddings
        """
        # Load images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)
        
        # Process batch
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            batch_features = self.model.get_image_features(**inputs)
        
        # Normalize
        embeddings = batch_features.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        return float(np.dot(embedding1, embedding2))
