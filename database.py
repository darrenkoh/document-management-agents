"""Database operations for storing document classifications and embeddings."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from tinydb import TinyDB, Query
import numpy as np

logger = logging.getLogger(__name__)


class DocumentDatabase:
    """Manages document classification storage and semantic search."""
    
    def __init__(self, db_path: str = "documents.json"):
        """Initialize the document database.
        
        Args:
            db_path: Path to the TinyDB JSON database file
        """
        self.db_path = Path(db_path)
        self.db = TinyDB(str(self.db_path))
        self.documents = self.db.table('documents')
        self.query = Query()
    
    def store_classification(self, file_path: str, content: str, categories: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> int:
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
            'embedding': None  # Will be set when embedding is generated
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
        """Store embedding vector for a document.
        
        Args:
            file_path: Path to the document
            embedding: Embedding vector as list of floats
        
        Returns:
            True if successful
        """
        try:
            existing = self.documents.search(self.query.file_path == str(file_path))
            if existing:
                doc_id = existing[0].doc_id
                self.documents.update({'embedding': embedding}, doc_ids=[doc_id])
                logger.debug(f"Stored embedding for {Path(file_path).name}")
                return True
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
    
    def search_semantic(self, query_embedding: List[float], top_k: int = 10,
                      threshold: float = 0.0) -> List[Dict]:
        """Perform semantic search using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of documents sorted by similarity (highest first)
        """
        query_vec = np.array(query_embedding)
        results = []

        # Get all documents with embeddings
        all_docs = self.documents.all()
        logger.debug(f"Total documents in database: {len(all_docs)}")

        docs_with_embeddings = sum(1 for doc in all_docs if doc.get('embedding'))
        logger.debug(f"Documents with embeddings: {docs_with_embeddings}")

        for doc in all_docs:
            embedding = doc.get('embedding')
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                try:
                    doc_embedding = np.array(embedding)

                    # Check if dimensions match
                    if len(query_vec) != len(doc_embedding):
                        logger.warning(f"Embedding dimension mismatch: query={len(query_vec)}, doc={len(doc_embedding)} for {doc.get('filename', 'N/A')}")
                        continue

                    # Calculate cosine similarity
                    similarity = np.dot(query_vec, doc_embedding) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(doc_embedding)
                    )

                    similarity_float = float(similarity)
                    logger.debug(f"Document '{doc.get('filename', 'N/A')}' similarity: {similarity_float}")

                    if similarity >= threshold:
                        doc_copy = dict(doc)
                        # Preserve the TinyDB doc_id
                        doc_copy['doc_id'] = doc.doc_id
                        doc_copy['similarity'] = similarity_float
                        results.append(doc_copy)
                        logger.debug(f"Document '{doc.get('filename', 'N/A')}' passed threshold ({similarity_float} >= {threshold})")
                    else:
                        logger.debug(f"Document '{doc.get('filename', 'N/A')}' below threshold ({similarity_float} < {threshold})")
                except Exception as e:
                    logger.error(f"Error processing embedding for {doc.get('filename', 'N/A')}: {e}")
                    continue

        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        logger.debug(f"Final results count: {len(results)}")

        return results[:top_k]
    
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

