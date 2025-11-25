"""Vector database abstraction layer for semantic search."""
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector storage and search."""

    @abstractmethod
    def __init__(self, persist_directory: str, **kwargs):
        """Initialize vector store.

        Args:
            persist_directory: Directory to persist vector data
            **kwargs: Additional configuration parameters
        """
        pass

    @abstractmethod
    def add_embeddings(self, embeddings: List[List[float]],
                      metadata: List[Dict[str, Any]],
                      ids: List[str]) -> bool:
        """Add embeddings with metadata to the store.

        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: List of unique IDs for the embeddings

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def search_similar(self, query_embedding: List[float],
                      top_k: int = 10,
                      threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of tuples: (id, similarity_score, metadata)
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs.

        Args:
            ids: List of IDs to delete

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID.

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of stored embeddings.

        Returns:
            Number of embeddings in the store
        """
        pass

    @abstractmethod
    def close(self):
        """Close the vector store connection."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""

    def __init__(self, persist_directory: str, collection_name: str = "documents", **kwargs):
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            **kwargs: Additional ChromaDB configuration
        """
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.errors import NotFoundError
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Filter out ChromaDB-specific parameters that shouldn't go to Settings
        chromadb_settings = {k: v for k, v in kwargs.items() if k not in ['dimension']}
        # Note: 'dimension' is not used by ChromaDB as it handles embedding dimensions automatically

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(**chromadb_settings)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except NotFoundError:
            # ChromaDB raises NotFoundError when collection doesn't exist
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new ChromaDB collection: {collection_name}")

    def add_embeddings(self, embeddings: List[List[float]],
                      metadata: List[Dict[str, Any]],
                      ids: List[str]) -> bool:
        """Add embeddings to ChromaDB."""
        try:
            # Normalize embeddings for cosine similarity (L2 distance on normalized vectors)
            import numpy as np
            normalized_embeddings = []
            for emb in embeddings:
                emb_array = np.array(emb)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    normalized_emb = emb_array / norm
                else:
                    normalized_emb = emb_array  # Avoid division by zero
                normalized_embeddings.append(normalized_emb.tolist())

            # Ensure metadata values are strings or numbers (ChromaDB limitation)
            processed_metadata = []
            for meta in metadata:
                processed_meta = {}
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        processed_meta[key] = value
                    else:
                        # Convert complex objects to strings
                        processed_meta[key] = str(value)
                processed_metadata.append(processed_meta)

            self.collection.add(
                embeddings=normalized_embeddings,
                metadatas=processed_metadata,
                ids=ids
            )

            logger.debug(f"Added {len(embeddings)} normalized embeddings to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding embeddings to ChromaDB: {e}")
            return False

    def search_similar(self, query_embedding: List[float],
                      top_k: int = 10,
                      threshold: float = -1.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings in ChromaDB."""
        try:
            # Normalize query embedding
            import numpy as np
            query_array = np.array(query_embedding)
            query_norm = np.linalg.norm(query_array)
            if query_norm > 0:
                normalized_query = (query_array / query_norm).tolist()
            else:
                normalized_query = query_embedding

            # Query ChromaDB (returns L2 distances for normalized vectors)
            results = self.collection.query(
                query_embeddings=[normalized_query],
                n_results=min(top_k * 3, 50),  # Get more results to rank properly
                include=['metadatas', 'distances']
            )

            logger.debug(f"ChromaDB query returned {len(results['ids'][0])} results")

            # Get all distances to find min/max for normalization
            distances = results['distances'][0]
            if distances:
                min_distance = min(distances)
                max_distance = max(distances)
                distance_range = max_distance - min_distance if max_distance > min_distance else 1.0

                search_results = []
                for i, (doc_id, distance, metadata) in enumerate(
                    zip(results['ids'][0], distances, results['metadatas'][0])):

                    # Normalize distance to 0-1 range and convert to similarity
                    if distance_range > 0:
                        normalized_distance = (distance - min_distance) / distance_range
                        similarity = 1.0 - normalized_distance  # Invert so smaller distance = higher similarity
                    else:
                        similarity = 1.0 if distance == 0 else 0.5  # Fallback for identical distances

                    logger.debug(f"Result {i}: ID={doc_id}, distance={distance:.3f}, similarity={similarity:.3f}")
                    search_results.append((doc_id, similarity, metadata))

                # Sort by similarity (highest first) and apply threshold
                search_results.sort(key=lambda x: x[1], reverse=True)
                filtered_results = [(doc_id, sim, meta) for doc_id, sim, meta in search_results if sim >= threshold]
            else:
                filtered_results = []

            logger.debug(f"ChromaDB search: {len(search_results)} total, {len(filtered_results)} after threshold {threshold}")
            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings from ChromaDB."""
        try:
            self.collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} embeddings from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
            return False

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID from ChromaDB."""
        try:
            result = self.collection.get(ids=[doc_id], include=['metadatas'])
            if result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception as e:
            logger.error(f"Error getting document by ID from ChromaDB: {e}")
            return None

    def count(self) -> int:
        """Get total number of embeddings in ChromaDB."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count from ChromaDB: {e}")
            return 0

    def close(self):
        """Close ChromaDB connection."""
        # ChromaDB handles persistence automatically
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation for high-performance search."""

    def __init__(self, persist_directory: str, dimension: int = 768, **kwargs):
        """Initialize FAISS vector store.

        Args:
            persist_directory: Directory to persist FAISS index
            dimension: Embedding dimension
            **kwargs: Additional FAISS configuration
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.index_file = self.persist_directory / "faiss_index.idx"

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.metadata_store = {}  # In-memory metadata store
        self.id_to_idx = {}  # Map document IDs to FAISS indices
        self.idx_to_id = {}  # Map FAISS indices to document IDs

        # Load existing index if available
        if self.index_file.exists():
            self._load_index()
            logger.info("Loaded existing FAISS index")
        else:
            logger.info("Created new FAISS index")

    def _load_index(self):
        """Load FAISS index from disk."""
        try:
            import faiss
            self.index = faiss.read_index(str(self.index_file))

            # Load metadata mapping (you might want to use a proper DB for this)
            metadata_file = self.persist_directory / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata_store = data.get('metadata', {})
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.idx_to_id = {int(k): v for k, v in data.get('idx_to_id', {}).items()}

        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            # Reinitialize empty index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = {}
            self.id_to_idx = {}
            self.idx_to_id = {}

    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            import faiss
            faiss.write_index(self.index, str(self.index_file))

            # Save metadata mapping
            metadata_file = self.persist_directory / "metadata.json"
            data = {
                'metadata': self.metadata_store,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }
            import json
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def add_embeddings(self, embeddings: List[List[float]],
                      metadata: List[Dict[str, Any]],
                      ids: List[str]) -> bool:
        """Add embeddings to FAISS index."""
        try:
            # Convert to numpy array and normalize for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_normalized = embeddings_array / norms

            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings_normalized)

            # Update metadata mappings
            for i, (doc_id, meta) in enumerate(zip(ids, metadata)):
                idx = start_idx + i
                self.metadata_store[doc_id] = meta
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id

            # Save to disk
            self._save_index()

            logger.debug(f"Added {len(embeddings)} embeddings to FAISS index")
            return True

        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {e}")
            return False

    def search_similar(self, query_embedding: List[float],
                      top_k: int = 10,
                      threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings in FAISS."""
        try:
            # Normalize query vector
            query_array = np.array([query_embedding], dtype=np.float32)
            query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
            query_norm[query_norm == 0] = 1
            query_normalized = query_array / query_norm

            # Search FAISS index
            scores, indices = self.index.search(query_normalized, min(top_k, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= threshold:  # FAISS returns -1 for invalid indices
                    doc_id = self.idx_to_id.get(int(idx))
                    if doc_id and doc_id in self.metadata_store:
                        results.append((doc_id, float(score), self.metadata_store[doc_id]))

            logger.debug(f"FAISS search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []

    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings from FAISS (requires index rebuild)."""
        try:
            # FAISS doesn't support deletion directly, need to rebuild index
            # Remove from metadata
            indices_to_remove = []
            for doc_id in ids:
                if doc_id in self.id_to_idx:
                    idx = self.id_to_idx[doc_id]
                    indices_to_remove.append(idx)
                    del self.metadata_store[doc_id]
                    del self.id_to_idx[doc_id]
                    if idx in self.idx_to_id:
                        del self.idx_to_id[idx]

            if indices_to_remove:
                # Rebuild index excluding removed vectors
                # This is a simplified implementation - in production you'd want to rebuild properly
                logger.warning("FAISS deletion requires index rebuild - simplified implementation")

            self._save_index()
            return True

        except Exception as e:
            logger.error(f"Error deleting from FAISS: {e}")
            return False

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        return self.metadata_store.get(doc_id)

    def count(self) -> int:
        """Get total number of embeddings."""
        return self.index.ntotal

    def close(self):
        """Save index before closing."""
        self._save_index()


def create_vector_store(store_type: str, **kwargs) -> VectorStore:
    """Factory function to create vector store instances.

    Args:
        store_type: Type of vector store ('chromadb' or 'faiss')
        **kwargs: Configuration parameters

    Returns:
        VectorStore instance
    """
    if store_type.lower() == 'chromadb':
        return ChromaVectorStore(**kwargs)
    elif store_type.lower() == 'faiss':
        return FAISSVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}. Supported: chromadb, faiss")
