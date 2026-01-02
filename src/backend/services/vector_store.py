"""Vector database abstraction layer for semantic search."""
import logging
import numpy as np
import os
import fcntl
import atexit
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Disable ChromaDB telemetry by default
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')

logger = logging.getLogger(__name__)


class ChromaDBLock:
    """File-based lock to prevent concurrent ChromaDB access from multiple processes.
    
    ChromaDB's PersistentClient doesn't handle concurrent access well, which can
    cause data corruption. This lock ensures only one process accesses the database
    at a time.
    """
    
    _instances: Dict[str, 'ChromaDBLock'] = {}
    
    def __init__(self, persist_directory: str):
        self.lock_file_path = Path(persist_directory) / ".chromadb.lock"
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = None
        self._locked = False
        
    @classmethod
    def get_lock(cls, persist_directory: str) -> 'ChromaDBLock':
        """Get or create a lock for the given directory."""
        if persist_directory not in cls._instances:
            cls._instances[persist_directory] = cls(persist_directory)
        return cls._instances[persist_directory]
    
    def _get_lock_holder_pid(self) -> Optional[int]:
        """Try to read the PID of the process holding the lock."""
        try:
            if self.lock_file_path.exists():
                with open(self.lock_file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return int(content)
        except Exception:
            pass
        return None
    
    def acquire(self, blocking: bool = False) -> bool:
        """Acquire the lock.
        
        Args:
            blocking: If True, block until lock is acquired. If False, fail immediately
                     if lock is held by another process.
        
        Returns:
            True if lock was acquired
            
        Raises:
            RuntimeError: If non-blocking and lock is held by another process
        """
        if self._locked:
            return True  # Already locked by this process
            
        self.lock_file = open(self.lock_file_path, 'w')
        try:
            lock_flags = fcntl.LOCK_EX
            if not blocking:
                lock_flags |= fcntl.LOCK_NB
                
            fcntl.flock(self.lock_file.fileno(), lock_flags)
            self._locked = True
            # Write PID to lock file for debugging
            self.lock_file.write(f"{os.getpid()}\n")
            self.lock_file.flush()
            logger.debug(f"Acquired ChromaDB lock: {self.lock_file_path}")
            return True
            
        except BlockingIOError:
            # Lock is held by another process
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            
            holder_pid = self._get_lock_holder_pid()
            pid_info = f" (PID: {holder_pid})" if holder_pid else ""
            
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: ChromaDB is locked by another process{pid_info}\n"
                f"{'='*70}\n"
                f"\n"
                f"ChromaDB cannot be accessed by multiple processes simultaneously.\n"
                f"This causes data corruption.\n"
                f"\n"
                f"SOLUTION: Stop the other process before running this one:\n"
                f"  - If the API server (app.py) is running, stop it first\n"
                f"  - If document_ingestion.py is running, wait for it to finish\n"
                f"\n"
                f"Lock file: {self.lock_file_path}\n"
                f"{'='*70}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(f"ChromaDB locked by another process{pid_info}. Stop it before continuing.")
            
        except Exception as e:
            logger.error(f"Failed to acquire ChromaDB lock: {e}")
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            raise
    
    def release(self):
        """Release the lock."""
        if self.lock_file and self._locked:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
                self.lock_file = None
                self._locked = False
                logger.debug(f"Released ChromaDB lock: {self.lock_file_path}")
            except Exception as e:
                logger.error(f"Failed to release ChromaDB lock: {e}")
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


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
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings and metadata.
        
        Returns:
            List of dictionaries containing embedding and metadata
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

    @abstractmethod
    def reset_collection(self) -> bool:
        """Reset (delete and recreate) the collection to fix corruption.
        
        Returns:
            True if reset was successful
        """
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""

    def __init__(self, persist_directory: str, collection_name: str = "documents", **kwargs):
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            **kwargs: Additional ChromaDB configuration including:
                - server_host: If set, connect to ChromaDB server instead of using PersistentClient
                - server_port: Port for ChromaDB server (default: 8000)
                - dimension: Expected embedding dimension
                - distance_metric: Distance metric for similarity search
        """
        # Disable ChromaDB telemetry by setting environment variable
        import os
        if 'ANONYMIZED_TELEMETRY' not in os.environ:
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'
            logger.info("Disabled ChromaDB telemetry")
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.errors import NotFoundError
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.base_collection_name = collection_name
        self.collection_name = collection_name
        self.expected_dimension = kwargs.get('dimension')
        
        # Check if we should use server mode (HttpClient) for concurrent access
        server_host = kwargs.get('server_host')
        server_port = kwargs.get('server_port', 8000)
        self._using_server_mode = server_host is not None
        self._lock = None

        # Extract distance metric configuration
        self.distance_metric = kwargs.get('distance_metric', 'l2')  # Default to 'l2' for backward compatibility

        if self._using_server_mode:
            # Use HttpClient for server mode - supports concurrent access
            logger.info(f"Connecting to ChromaDB server at {server_host}:{server_port}")
            self.client = chromadb.HttpClient(host=server_host, port=server_port)
        else:
            # Use PersistentClient with file locking for local mode
            # Acquire file lock to prevent concurrent access from multiple processes
            self._lock = ChromaDBLock.get_lock(str(self.persist_directory))
            self._lock.acquire(blocking=False)
            
            # Register cleanup on exit to ensure lock is released
            atexit.register(self._release_lock)

            # Filter out ChromaDB-specific parameters that shouldn't go to Settings
            chromadb_settings = {k: v for k, v in kwargs.items() 
                                if k not in ['dimension', 'distance_metric', 'server_host', 'server_port']}

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(**chromadb_settings)
            )

        # Get or create collection with distance metric (and store our expected embedding dim in metadata)
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name} (using distance metric: {self.distance_metric})")
        except NotFoundError:
            # ChromaDB raises NotFoundError when collection doesn't exist
            # Create collection with specified distance metric
            collection_metadata = {"hnsw:space": self.distance_metric}
            if self.expected_dimension:
                collection_metadata["embedding_dimension"] = int(self.expected_dimension)
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            logger.info(f"Created new ChromaDB collection: {collection_name} (distance metric: {self.distance_metric})")

        # Prefer a dimension-specific collection whenever dimension is configured.
        # This avoids hard failures (and subtle correctness issues) when the embedding model changes.
        if self.expected_dimension:
            self._switch_to_dimension_collection(warn_on_mismatch=False)

    def _dimension_collection_name(self) -> Optional[str]:
        """Return the dimension-suffixed collection name (if dimension is known)."""
        if not self.expected_dimension:
            return None
        return f"{self.base_collection_name}_dim{int(self.expected_dimension)}"

    def _switch_to_dimension_collection(self, warn_on_mismatch: bool = True) -> bool:
        """Switch to a dimension-specific collection to avoid embedding-dimension mismatches.

        This is non-destructive: we do not delete or modify the old collection; we simply
        use a different collection name for the current embedding model/dimension.
        """
        dim_name = self._dimension_collection_name()
        if not dim_name:
            return False
        if self.collection_name == dim_name:
            return True

        try:
            from chromadb.errors import NotFoundError
        except Exception:
            return False

        collection_metadata = {"hnsw:space": self.distance_metric}
        collection_metadata["embedding_dimension"] = int(self.expected_dimension)

        self.collection_name = dim_name
        log_func = logger.warning if warn_on_mismatch else logger.info
        reason_suffix = "due to embedding dimension mismatch" if warn_on_mismatch else f"for configured dimension {int(self.expected_dimension)}"
        try:
            self.collection = self.client.get_collection(name=dim_name)
            log_func(
                f"Switched to existing ChromaDB collection '{dim_name}' {reason_suffix}"
            )
        except NotFoundError:
            self.collection = self.client.create_collection(name=dim_name, metadata=collection_metadata)
            log_func(
                f"Created and switched to new ChromaDB collection '{dim_name}' {reason_suffix}"
            )
        return True

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
                    elif isinstance(value, (list, dict)):
                        # Convert lists/dicts to JSON strings for proper storage/retrieval
                        import json
                        processed_meta[key] = json.dumps(value)
                    else:
                        # Convert other complex objects to strings
                        processed_meta[key] = str(value)
                processed_metadata.append(processed_meta)

            self.collection.add(
                embeddings=normalized_embeddings,
                metadatas=processed_metadata,
                ids=ids
            )

            logger.info(f"Added {len(embeddings)} normalized embeddings to ChromaDB")
            return True

        except Exception as e:
            # If we hit an embedding dimension mismatch, automatically switch to a dimension-specific
            # collection and retry once. This prevents hard failures when the embedding model changes.
            try:
                from chromadb.errors import InvalidArgumentError
            except Exception:
                InvalidArgumentError = None  # type: ignore

            if InvalidArgumentError and isinstance(e, InvalidArgumentError) and "expecting embedding with dimension" in str(e).lower():
                if self._switch_to_dimension_collection():
                    try:
                        self.collection.add(
                            embeddings=normalized_embeddings,
                            metadatas=processed_metadata,
                            ids=ids
                        )
                        logger.info(f"Added {len(embeddings)} normalized embeddings to ChromaDB (after collection switch)")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Error adding embeddings to ChromaDB after collection switch: {retry_error}")
                        return False

            logger.error(f"Error adding embeddings to ChromaDB: {e}")
            return False

    def search_similar(self, query_embedding: List[float],
                      top_k: int = 10,
                      threshold: float = -1.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings in ChromaDB."""
        try:
            # Ensure NotFoundError name is available in this method scope
            from chromadb.errors import NotFoundError

            # Guardrail: if the query embedding dimension doesn't match our configured expectation,
            # we should not query the current collection (it will error or return nonsense).
            if self.expected_dimension and len(query_embedding) != int(self.expected_dimension):
                logger.warning(
                    f"Query embedding dimension {len(query_embedding)} does not match expected {int(self.expected_dimension)}; "
                    "switching to dimension-specific collection"
                )
                self._switch_to_dimension_collection()

            # Ensure the collection exists and is accessible
            try:
                # Try to access the collection to make sure it exists
                self.collection.count()
            except Exception as e:
                logger.warning(f"Collection not accessible, attempting to reconnect: {e}")
                try:
                    # Try to get the collection again
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"Reconnected to existing collection: {self.collection_name}")
                except NotFoundError:
                    # Create collection if it doesn't exist
                    collection_metadata = {"hnsw:space": self.distance_metric}
                    if self.expected_dimension:
                        collection_metadata["embedding_dimension"] = int(self.expected_dimension)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata=collection_metadata
                    )
                    logger.info(f"Created new collection during search: {self.collection_name}")
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect to collection: {reconnect_error}")
                    return []

            # Normalize query embedding
            import numpy as np
            query_array = np.array(query_embedding)
            query_norm = np.linalg.norm(query_array)
            if query_norm > 0:
                normalized_query = (query_array / query_norm).tolist()
            else:
                normalized_query = query_embedding

            # Query ChromaDB (distance metric depends on collection configuration)
            try:
                results = self.collection.query(
                    query_embeddings=[normalized_query],
                    n_results=top_k,  # Use the max_candidates value passed from config
                    include=['metadatas', 'distances']
                )
            except Exception as query_error:
                # Handle embedding dimension mismatch by switching to a dimension-specific collection and retrying once.
                try:
                    from chromadb.errors import InvalidArgumentError
                except Exception:
                    InvalidArgumentError = None  # type: ignore

                if InvalidArgumentError and isinstance(query_error, InvalidArgumentError) and "expecting embedding with dimension" in str(query_error).lower():
                    if self._switch_to_dimension_collection():
                        results = self.collection.query(
                            query_embeddings=[normalized_query],
                            n_results=top_k,
                            include=['metadatas', 'distances']
                        )
                    else:
                        raise
                else:
                    raise

            logger.info(f"ChromaDB query returned {len(results['ids'][0])} results")

            # Convert distances to absolute similarities
            # ChromaDB returns cosine distances (0-2 range), convert to similarity (1 to -1)
            # For positive similarities, clamp to 0-1 range
            distances = results['distances'][0]
            search_results = []
            if distances:
                for i, (doc_id, distance, metadata) in enumerate(
                    zip(results['ids'][0], distances, results['metadatas'][0])):

                    # Convert distance to similarity based on metric
                    if self.distance_metric == 'cosine':
                        # Cosine distance: 0 (identical) to 2 (opposite)
                        # Convert to similarity: 1 - (distance/2) gives 1.0 to 0.0 range
                        similarity = 1.0 - (distance / 2.0)
                    elif self.distance_metric == 'l2':
                        # L2 distance with normalized vectors: convert using 1/(1+distance)
                        similarity = 1.0 / (1.0 + distance)
                    else:
                        # For other metrics (like 'ip'), use L2-style conversion as fallback
                        similarity = 1.0 / (1.0 + distance)

                    logger.info(f"Result {i}: ID={doc_id}, distance={distance:.3f}, similarity={similarity:.3f}")
                    search_results.append((doc_id, similarity, metadata))

                # Sort by similarity (highest first) and apply threshold
                search_results.sort(key=lambda x: x[1], reverse=True)
                filtered_results = [(doc_id, sim, meta) for doc_id, sim, meta in search_results if sim >= threshold]
            else:
                filtered_results = []

            logger.info(f"ChromaDB search: {len(search_results)} total, {len(filtered_results)} after threshold {threshold}")
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
            logger.info(f"Deleted {len(ids)} embeddings from ChromaDB")
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

    def reset_collection(self) -> bool:
        """Reset (delete and recreate) the current collection to fix corruption.
        
        This is useful when the ChromaDB collection becomes corrupted (e.g., 'finding id' errors).
        After calling this, you will need to re-ingest all documents to regenerate embeddings.
        
        Returns:
            True if reset was successful
        """
        try:
            logger.warning(f"Resetting ChromaDB collection: {self.collection_name}")
            
            # Delete the existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted corrupted collection: {self.collection_name}")
            except Exception as delete_error:
                logger.warning(f"Could not delete collection (may not exist): {delete_error}")
            
            # Recreate the collection with the same settings
            collection_metadata = {"hnsw:space": self.distance_metric}
            if self.expected_dimension:
                collection_metadata["embedding_dimension"] = int(self.expected_dimension)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=collection_metadata
            )
            logger.info(f"Created fresh collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings and metadata from ChromaDB."""
        try:
            # Ensure NotFoundError name is available in this method scope
            from chromadb.errors import NotFoundError

            # Ensure the collection exists and is accessible
            try:
                # Try to access the collection to make sure it exists
                count = self.collection.count()
                logger.info(f"Collection has {count} embeddings")
                if count == 0:
                    return []
            except Exception as e:
                logger.warning(f"Collection not accessible, attempting to recreate: {e}")
                try:
                    # Try to get the collection again
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"Reconnected to existing collection: {self.collection_name}")
                    count = self.collection.count()
                    if count == 0:
                        return []
                except NotFoundError:
                    # Create collection if it doesn't exist
                    collection_metadata = {"hnsw:space": self.distance_metric}
                    if self.expected_dimension:
                        collection_metadata["embedding_dimension"] = int(self.expected_dimension)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata=collection_metadata
                    )
                    logger.info(f"Created new collection: {self.collection_name}")
                    return []
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate collection: {recreate_error}")
                    return []

            # Use the reliable method: get all data at once (ChromaDB handles this efficiently)
            try:
                result = self.collection.get(include=['embeddings', 'metadatas'])
                if not result['ids']:
                    return []

                all_data = []
                for i, (doc_id, embedding, metadata) in enumerate(zip(
                    result['ids'],
                    result['embeddings'],
                    result['metadatas']
                )):
                    try:
                        meta = metadata if metadata else {}
                        if 'id' not in meta:
                            meta['id'] = doc_id

                        all_data.append({
                            'id': doc_id,
                            'embedding': embedding,
                            'metadata': meta
                        })
                    except Exception as item_error:
                        logger.warning(f"Error processing embedding {i} with ID {doc_id}: {item_error}")
                        # Skip corrupted items but continue processing others
                        continue

                logger.info(f"Successfully retrieved {len(all_data)} embeddings")
                return all_data

            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"Error getting all embeddings from ChromaDB: {e}")
                
                # If the error is related to "finding id", this indicates corruption
                if "finding id" in error_str or "error executing plan" in error_str:
                    logger.error("=" * 60)
                    logger.error("ChromaDB DATA CORRUPTION DETECTED")
                    logger.error("=" * 60)
                    logger.error("The ChromaDB collection is corrupted and cannot retrieve embeddings.")
                    logger.error("")
                    logger.error("To fix this issue, you have two options:")
                    logger.error("")
                    logger.error("OPTION 1: Reset via API (recommended)")
                    logger.error("  Call the /api/embeddings/reset endpoint to reset the collection")
                    logger.error("  Then re-run the document ingestion to regenerate embeddings")
                    logger.error("")
                    logger.error("OPTION 2: Manual reset")
                    logger.error(f"  1. Stop the application")
                    logger.error(f"  2. Delete the vector store directory: {self.persist_directory}")
                    logger.error(f"  3. Restart the application")
                    logger.error(f"  4. Re-run the document ingestion to regenerate embeddings")
                    logger.error("=" * 60)
                    return []
                return []

        except Exception as e:
            logger.error(f"Error getting all embeddings from ChromaDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def count(self) -> int:
        """Get total number of embeddings in ChromaDB."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count from ChromaDB: {e}")
            return 0

    def _release_lock(self):
        """Release the file lock (called on close or atexit)."""
        if hasattr(self, '_lock') and self._lock:
            self._lock.release()

    def close(self):
        """Close ChromaDB connection and release lock."""
        # ChromaDB handles persistence automatically
        # Release the file lock to allow other processes to access
        self._release_lock()
        
        # Unregister atexit handler since we're closing explicitly
        try:
            atexit.unregister(self._release_lock)
        except Exception:
            pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation for high-performance search."""

    def __init__(self, persist_directory: str, dimension: int = 4096, **kwargs):
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

            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
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

            logger.info(f"FAISS search returned {len(results)} results")
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

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings and metadata from FAISS.
        
        Note: This is a partial implementation since FAISS index doesn't strictly store 
        original vectors in a retrieveable way for all index types, but for IndexFlatIP it does.
        """
        try:
            # For IndexFlatIP, we can reconstruct vectors
            # But for simplicity and safety, we'll rely on what we might have stored or return empty
            # A proper implementation would require storing original vectors alongside the index
            # if the index type is lossy (like IVFPQ).
            # Since we're using IndexFlatIP, we could technically reconstruct:
            
            if not hasattr(self.index, 'reconstruct'):
                 logger.warning("FAISS index does not support reconstruction, returning empty")
                 return []
                 
            ntotal = self.index.ntotal
            all_data = []
            
            for i in range(ntotal):
                try:
                    embedding = self.index.reconstruct(i)
                    # Convert numpy array to list
                    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                    
                    doc_id = self.idx_to_id.get(i)
                    if doc_id:
                        metadata = self.metadata_store.get(doc_id, {})
                        if 'id' not in metadata:
                            metadata['id'] = doc_id
                            
                        all_data.append({
                            'id': doc_id,
                            'embedding': embedding_list,
                            'metadata': metadata
                        })
                except Exception:
                    continue
                    
            return all_data
            
        except Exception as e:
            logger.error(f"Error getting all embeddings from FAISS: {e}")
            return []

    def count(self) -> int:
        """Get total number of embeddings."""
        return self.index.ntotal

    def close(self):
        """Save index before closing."""
        self._save_index()

    def reset_collection(self) -> bool:
        """Reset (delete and recreate) the FAISS index to fix corruption.
        
        Returns:
            True if reset was successful
        """
        try:
            import faiss
            logger.warning("Resetting FAISS index")
            
            # Delete index files
            if self.index_file.exists():
                self.index_file.unlink()
                logger.info(f"Deleted FAISS index file: {self.index_file}")
            
            metadata_file = self.persist_directory / "metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"Deleted FAISS metadata file: {metadata_file}")
            
            # Reinitialize empty index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = {}
            self.id_to_idx = {}
            self.idx_to_id = {}
            
            logger.info("Created fresh FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset FAISS index: {e}")
            return False


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
