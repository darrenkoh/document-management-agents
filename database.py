"""Database operations for storing document classifications and embeddings."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from tinydb import TinyDB, Query
import numpy as np

from vector_store import create_vector_store, VectorStore

logger = logging.getLogger(__name__)


class DocumentDatabase:
    """Manages document classification storage and semantic search."""

    def __init__(self, db_path: str = "documents.json", vector_store: Optional[VectorStore] = None, config=None):
        """Initialize the document database.

        Args:
            db_path: Path to the TinyDB JSON database file
            vector_store: Optional vector store instance for embeddings
            config: Configuration object for semantic search settings
        """
        self.db_path = Path(db_path)
        self.db = TinyDB(str(self.db_path))
        self.documents = self.db.table('documents')
        self.query = Query()
        self.vector_store = vector_store
        self.config = config
    
    def store_classification(self, file_path: str, content: str, categories: str,
                           metadata: Optional[Dict[str, Any]] = None, file_hash: Optional[str] = None) -> int:
        """Store a document classification in the database.
        
        Args:
            file_path: Path to the original file
            content: Extracted text content
            categories: Classification categories (comma-separated or hyphenated)
            metadata: Optional additional metadata
        
        Returns:
            Document ID in the database
        """
        doc = {
            'file_path': str(file_path),
            'filename': Path(file_path).name,
            'content': content,
            'content_preview': content[:500] if len(content) > 500 else content,
            'categories': categories,
            'classification_date': datetime.now().isoformat(),
            'metadata': metadata or {},
            'file_hash': file_hash,
            'embedding_stored': False  # Will be set to True when stored in vector DB
        }
        
        # Check if document already exists
        existing = self.documents.search(self.query.file_path == str(file_path))
        if existing:
            # Update existing document
            doc_id = existing[0].doc_id
            self.documents.update(doc, doc_ids=[doc_id])
            logger.info(f"Updated classification for {Path(file_path).name} in database")
            return doc_id
        else:
            # Insert new document
            doc_id = self.documents.insert(doc)
            logger.info(f"Stored classification for {Path(file_path).name} in database (ID: {doc_id})")
            return doc_id
    
    def store_embedding(self, file_path: str, embedding: List[float]) -> bool:
        """Store embedding vector for a document directly in vector store.
        
        Args:
            file_path: Path to the document
            embedding: Embedding vector as list of floats
        
        Returns:
            True if successful
        """
        if not self.vector_store:
            logger.error("Vector store not available - cannot store embeddings")
            return False

        try:
            existing = self.documents.search(self.query.file_path == str(file_path))
            if existing:
                doc_id = existing[0].doc_id
                doc = existing[0]

                doc_id_str = str(doc_id)
                metadata = {
                    'file_path': doc.get('file_path'),
                    'filename': doc.get('filename'),
                    'categories': doc.get('categories'),
                    'content_preview': doc.get('content_preview'),
                    'file_hash': doc.get('file_hash')
                }

                success = self.vector_store.add_embeddings(
                    embeddings=[embedding],
                    metadata=[metadata],
                    ids=[doc_id_str]
                )

                if success:
                    # Mark as stored in vector DB (no embedding stored in JSON)
                    self.documents.update({'embedding_stored': True}, doc_ids=[doc_id])
                    logger.debug(f"Stored embedding in vector store for {Path(file_path).name}")
                    return True
                else:
                    logger.error(f"Failed to store embedding in vector store for {file_path}")
                    return False
            else:
                logger.warning(f"Document not found for embedding: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
    
    def get_document(self, file_path: str) -> Optional[Dict]:
        """Get a document by file path.

        Args:
            file_path: Path to the document

        Returns:
            Document dictionary or None if not found
        """
        results = self.documents.search(self.query.file_path == str(file_path))
        return results[0] if results else None

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get a document by file hash.

        Args:
            file_hash: SHA-256 hash of the file content

        Returns:
            Document dictionary or None if not found
        """
        results = self.documents.search(self.query.file_hash == file_hash)
        return results[0] if results else None
    
    def search_semantic(self, query_embedding: List[float], top_k: int = 10,
                      threshold: float = None) -> List[Dict]:
        """Perform semantic search using vector database.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (uses config if None)

        Returns:
            List of documents sorted by similarity (highest first)
        """
        if not self.vector_store:
            logger.error("Vector store not available for semantic search")
            return []

        # Use config values if available
        if self.config:
            if threshold is None:
                threshold = self.config.semantic_search_min_threshold
            max_candidates = self.config.semantic_search_max_candidates
            debug_enabled = self.config.semantic_search_debug or logger.isEnabledFor(logging.DEBUG)
        else:
            if threshold is None:
                threshold = -1.0
            max_candidates = 50
            debug_enabled = logger.isEnabledFor(logging.DEBUG)

        logger.debug(f"Using vector store for semantic search (top_k={top_k}, threshold={threshold}, max_candidates={max_candidates}, debug={debug_enabled})")

        # Request more results from vector store to ensure we get good matches
        vector_results = self.vector_store.search_similar(query_embedding, top_k=max_candidates, threshold=threshold)

        results = []
        for doc_id_str, similarity, metadata in vector_results:
            try:
                doc_id = int(doc_id_str)
                # Get full document from TinyDB
                doc = self.documents.get(doc_id=doc_id)
                if doc:
                    doc_copy = dict(doc)
                    doc_copy['doc_id'] = doc_id
                    doc_copy['similarity'] = similarity
                    results.append(doc_copy)
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing vector search result for ID {doc_id_str}: {e}")
                continue

        logger.debug(f"Vector store search returned {len(results)} results")
        return results
    
    def search_by_category(self, category: str) -> List[Dict]:
        """Search documents by category.

        Args:
            category: Category to search for

        Returns:
            List of matching documents
        """
        # Search for category in categories field (handles hyphenated categories)
        tinydb_results = self.documents.search(
            self.query.categories.test(lambda x: category.lower() in x.lower() if x else False)
        )

        # Convert TinyDB documents to dicts for consistency with semantic search
        results = []
        for doc in tinydb_results:
            doc_dict = dict(doc)
            doc_dict['doc_id'] = doc.doc_id
            results.append(doc_dict)

        return results
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents from the database.

        Returns:
            List of all documents
        """
        docs = []
        for doc in self.documents.all():
            doc_dict = dict(doc)
            doc_dict['doc_id'] = doc.doc_id
            docs.append(doc_dict)
        return docs
    
    def export_to_json(self, output_path: str) -> bool:
        """Export all documents to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        
        Returns:
            True if successful
        """
        try:
            documents = self.get_all_documents()
            # Remove doc_id from export (it's TinyDB internal)
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_documents': len(documents),
                'documents': [
                    {k: v for k, v in doc.items() if k != 'embedding' or v is None} 
                    for doc in documents
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(documents)} documents to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        self.db.close()
        if self.vector_store:
            self.vector_store.close()

