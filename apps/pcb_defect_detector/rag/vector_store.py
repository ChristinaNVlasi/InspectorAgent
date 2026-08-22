"""
ChromaDB vector store for PCB defect detection RAG system
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class PCBVectorStore:
    """ChromaDB-based vector store for PCB images"""
    
    def __init__(self, persist_directory: str, collection_name: str = "pcb_defects"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized PCB vector store: {collection_name}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Use ChromaDB's built-in get_or_create method
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PCB images for defect detection via similarity search"}
            )
            logger.info(f"Collection loaded: {self.collection_name} with {collection.count()} items")
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
        
        return collection
    
    def add_images(self, 
                   embeddings: np.ndarray,
                   metadatas: List[Dict[str, Any]],
                   image_paths: List[str],
                   ids: Optional[List[str]] = None) -> None:
        """
        Add images to vector store
        
        Args:
            embeddings: Array of image embeddings
            metadatas: List of metadata dictionaries with 'status', 'defect_type', 'location', etc.
            image_paths: List of image file paths
            ids: Optional list of custom IDs
        """
        if ids is None:
            ids = [f"pcb_{i:06d}" for i in range(len(embeddings))]
        
        # Convert embeddings to list
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else [embeddings.tolist()]
        
        # Prepare documents (image paths)
        documents = [str(path) for path in image_paths]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
        
        logger.info(f"Added {len(embeddings_list)} images to vector store")
    
    def search_similar(self, 
                      query_embedding: np.ndarray,
                      n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar PCB images
        
        Args:
            query_embedding: Query image embedding
            n_results: Number of results to return
            where: Metadata filter conditions (e.g., {"status": "OK"})
            
        Returns:
            Search results with distances, metadatas, and image paths
        """
        # Ensure embedding is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze()
        
        # Convert embedding to list
        query_embedding_list = query_embedding.tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        return {
            "total_images": count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory)
        }
    
    def reset(self) -> None:
        """Reset (clear) the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info(f"Reset collection: {self.collection_name}")
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete images by IDs"""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} images from vector store")
