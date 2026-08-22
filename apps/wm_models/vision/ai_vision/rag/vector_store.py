"""
ChromaDB vector store for RAG-based image similarity search
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import logging
from PIL import Image

from config import VECTOR_DB_CONFIG, PARTS_IMAGES_DIR, COMPONENT_MAPPING
from embeddings.clip_embedder import CLIPEmbedder

logger = logging.getLogger(__name__)

class ComponentVectorStore:
    """ChromaDB-based vector store for component images"""
    
    def __init__(self, 
                 persist_directory: str = VECTOR_DB_CONFIG["persist_directory"],
                 collection_name: str = VECTOR_DB_CONFIG["collection_name"]):
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
        
        logger.info(f"Initialized ChromaDB vector store: {collection_name}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Found existing collection: {self.collection_name}")
        except ValueError:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Alternator component images for similarity search"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
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
            metadatas: List of metadata dictionaries
            image_paths: List of image file paths
            ids: Optional list of custom IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [f"img_{i:06d}" for i in range(len(embeddings))]
        
        # Convert embeddings to list
        embeddings_list = embeddings.tolist()
        
        # Prepare documents (image paths for retrieval)
        documents = [str(path) for path in image_paths]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
        
        logger.info(f"Added {len(embeddings)} images to vector store")
    
    def search_similar(self, 
                      query_embedding: np.ndarray,
                      n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar images
        
        Args:
            query_embedding: Query image embedding
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            Search results with distances, metadatas, and documents
        """
        # Convert embedding to list
        query_embedding_list = query_embedding.tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        return {
            "total_images": count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory)
        }
    
    def reset_collection(self) -> None:
        """Reset (clear) the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info(f"Reset collection: {self.collection_name}")

class RAGVectorDatabase:
    """RAG-enabled vector database for component analysis"""
    
    def __init__(self, embedder: CLIPEmbedder):
        """
        Initialize RAG vector database
        
        Args:
            embedder: CLIP embedder for generating embeddings
        """
        self.embedder = embedder
        self.vector_store = ComponentVectorStore()
        
    def build_database(self, 
                      parts_images_dir: Path = PARTS_IMAGES_DIR,
                      force_rebuild: bool = False) -> None:
        """
        Build vector database from parts images
        
        Args:
            parts_images_dir: Directory containing component images
            force_rebuild: Whether to rebuild even if collection exists
        """
        # Check if database already exists
        stats = self.vector_store.get_collection_stats()
        if stats["total_images"] > 0 and not force_rebuild:
            logger.info(f"Database already exists with {stats['total_images']} images")
            return
        
        if force_rebuild:
            self.vector_store.reset_collection()
        
        logger.info("Building RAG vector database...")
        
        # Collect all image data
        image_data = self._collect_image_data(parts_images_dir)
        
        if not image_data:
            logger.error("No images found to process")
            return
        
        # Extract image paths
        image_paths = [item['path'] for item in image_data]
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(image_paths)} images...")
        embeddings = self.embedder.encode_batch_images(image_paths, batch_size=16)
        
        # Prepare metadata
        metadatas = []
        documents = []
        ids = []
        
        for i, item in enumerate(image_data):
            metadata = item['metadata'].copy()
            
            # Ensure all metadata values are strings or numbers (ChromaDB requirement)
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    processed_metadata[key] = value
                else:
                    processed_metadata[key] = str(value)
            
            metadatas.append(processed_metadata)
            documents.append(str(item['path']))
            ids.append(f"component_{i:06d}")
        
        # Add to vector store
        self.vector_store.add_images(
            embeddings=embeddings,
            metadatas=metadatas,
            image_paths=[item['path'] for item in image_data],
            ids=ids
        )
        
        logger.info(f"Successfully built database with {len(image_data)} images")
    
    def _collect_image_data(self, parts_dir: Path) -> List[Dict[str, Any]]:
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
            image_files = list(component_dir.glob("*.Jpeg"))
            logger.info(f"Found {len(image_files)} images in {component_name}")
            
            for image_path in image_files:
                # Parse filename for additional metadata
                filename = image_path.stem
                parts = filename.split('_')
                
                metadata = {
                    "component_type": component_info["type"],
                    "condition": component_info["condition"],
                    "description": component_info["description"],
                    "component_dir": component_name,
                    "filename": filename
                }
                
                # Extract additional info from filename if possible
                if len(parts) >= 3:
                    metadata.update({
                        "sample_id": parts[0],
                        "core_id": parts[1] if len(parts) > 1 else "",
                        "line_info": parts[2] if len(parts) > 2 else ""
                    })
                
                image_data.append({
                    "path": str(image_path),
                    "metadata": metadata
                })
        
        return image_data
    
    def search_similar_components(self, 
                                query_image: Union[Image.Image, str, Path],
                                top_k: int = 5,
                                component_filter: Optional[str] = None,
                                condition_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar components with optional filtering
        
        Args:
            query_image: Query image or path
            top_k: Number of results to return
            component_filter: Filter by component type
            condition_filter: Filter by condition
            
        Returns:
            List of similar component results
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_image(query_image)
        
        # Build filter conditions
        where_conditions = {}
        if component_filter:
            where_conditions["component_type"] = component_filter
        if condition_filter:
            where_conditions["condition"] = condition_filter
        
        # Search in vector store
        results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where_conditions if where_conditions else None
        )
        
        # Process results
        processed_results = []
        
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i],
                    "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "image_path": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                }
                processed_results.append(result)
        
        return processed_results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = self.vector_store.get_collection_stats()
        
        # Add component breakdown if possible
        try:
            # Query all items to get component breakdown
            all_results = self.vector_store.collection.get()
            
            component_counts = {}
            condition_counts = {}
            
            for metadata in all_results['metadatas']:
                comp_type = metadata.get('component_type', 'unknown')
                condition = metadata.get('condition', 'unknown')
                
                component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            stats.update({
                "component_breakdown": component_counts,
                "condition_breakdown": condition_counts
            })
        
        except Exception as e:
            logger.warning(f"Could not get detailed stats: {e}")
        
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize components
    embedder = CLIPEmbedder()
    rag_db = RAGVectorDatabase(embedder)
    
    # Build database
    print("Building RAG vector database...")
    rag_db.build_database(force_rebuild=True)
    
    # Get stats
    stats = rag_db.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Test search
    sample_image = PARTS_IMAGES_DIR / "A1 - Pulley" / "303944_RS-CORE-015284_Line30_A1.Jpeg"
    if sample_image.exists():
        results = rag_db.search_similar_components(sample_image, top_k=3)
        print(f"Found {len(results)} similar images")
        for i, result in enumerate(results):
            print(f"{i+1}. Similarity: {result['similarity']:.3f}, "
                  f"Component: {result['metadata']['component_type']}")
    
    print("RAG vector database ready!")