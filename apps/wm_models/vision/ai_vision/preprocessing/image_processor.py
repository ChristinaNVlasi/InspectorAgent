"""
Image preprocessing pipeline for alternator component analysis
"""
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some features will be disabled.")

import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional, Union
import torch
import torchvision.transforms as transforms
from pathlib import Path
import logging

from config import DATA_CONFIG

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Main image processing class for alternator component images"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                 normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        
        # Standard ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Define preprocessing transforms
        self.transforms = self._build_transforms()
        
    def _build_transforms(self) -> transforms.Compose:
        """Build preprocessing transform pipeline"""
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
            
        return transforms.Compose(transform_list)
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image from file path"""
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded image: {image_path} - Size: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def preprocess_image(self, 
                        image: Union[Image.Image, str, Path],
                        enhance_quality: bool = True) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image, or path to image file
            enhance_quality: Whether to apply quality enhancement
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Enhance image quality if requested
        if enhance_quality:
            image = self.enhance_image_quality(image)
        
        # Apply transforms
        tensor = self.transforms(image)
        
        return tensor
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better feature extraction
        
        Args:
            image: PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        if not CV2_AVAILABLE:
            # Fallback to PIL-only enhancement
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)
            
            return enhanced_image
        
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(img_enhanced)
        
        # Fine-tune contrast and sharpness
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(1.1)
        
        return enhanced_image
    
    def detect_blur(self, image: Image.Image, threshold: float = 100.0) -> bool:
        """
        Detect if image is blurry using Laplacian variance
        
        Args:
            image: PIL Image
            threshold: Blur threshold (lower = more blurry)
            
        Returns:
            True if image is sharp, False if blurry
        """
        if not CV2_AVAILABLE:
            # Simple fallback - assume images are sharp
            logger.info("OpenCV not available, assuming image is sharp")
            return True
        
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_sharp = laplacian_var > threshold
        logger.info(f"Blur detection - Laplacian variance: {laplacian_var:.2f}, Sharp: {is_sharp}")
        
        return is_sharp
    
    def extract_component_region(self, 
                                image: Image.Image,
                                padding: float = 0.1) -> Image.Image:
        """
        Extract the main component region using edge detection
        
        Args:
            image: PIL Image
            padding: Padding around detected region (0-1)
            
        Returns:
            Cropped image focusing on component
        """
        if not CV2_AVAILABLE:
            # Return original image if OpenCV not available
            logger.warning("OpenCV not available, returning original image")
            return image
        
        # Convert to numpy
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be the component)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            img_h, img_w = img_array.shape[:2]
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img_w - x, w + 2 * pad_x)
            h = min(img_h - y, h + 2 * pad_y)
            
            # Crop image
            cropped = img_array[y:y+h, x:x+w]
            return Image.fromarray(cropped)
        
        logger.warning("No component region detected, returning original image")
        return image
    
    def batch_preprocess(self, 
                        image_paths: list,
                        enhance_quality: bool = True) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            image_paths: List of image file paths
            enhance_quality: Whether to apply quality enhancement
            
        Returns:
            Batch tensor of preprocessed images
        """
        batch_tensors = []
        
        for path in image_paths:
            try:
                tensor = self.preprocess_image(path, enhance_quality)
                batch_tensors.append(tensor)
            except Exception as e:
                logger.error(f"Failed to preprocess {path}: {e}")
                continue
        
        if batch_tensors:
            return torch.stack(batch_tensors)
        else:
            raise ValueError("No images could be processed from the batch")

# Utility functions
def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate if image file is readable and has valid format
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image {image_path}: {e}")
        return False

def get_image_stats(image: Image.Image) -> dict:
    """
    Get basic statistics about an image
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with image statistics
    """
    img_array = np.array(image)
    
    stats = {
        "size": image.size,
        "mode": image.mode,
        "mean_brightness": np.mean(img_array),
        "std_brightness": np.std(img_array),
        "min_brightness": np.min(img_array),
        "max_brightness": np.max(img_array)
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = ImageProcessor()
    
    # Example image path
    sample_image_path = "/Users/christinavlasi/Documents/GitHub/rEUman-ML/apps/BORG/parts_images/A1 - Pulley/303944_RS-CORE-015284_Line30_A1.Jpeg"
    
    try:
        # Load and preprocess single image
        image = processor.load_image(sample_image_path)
        
        # Check image quality
        is_sharp = processor.detect_blur(image)
        print(f"Image is sharp: {is_sharp}")
        
        # Get image statistics
        stats = get_image_stats(image)
        print(f"Image stats: {stats}")
        
        # Preprocess for model input
        tensor = processor.preprocess_image(image)
        print(f"Preprocessed tensor shape: {tensor.shape}")
        
    except Exception as e:
        print(f"Error processing image: {e}")