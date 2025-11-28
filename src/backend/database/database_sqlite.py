"""SQLite-based Document Database - High-performance alternative to TinyDB."""
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.services.vector_store import create_vector_store, VectorStore

logger = logging.getLogger(__name__)


class SQLiteDocumentDatabase:
    """SQLite-based document database with JSON support for high performance."""

    def __init__(self, db_path: str = "documents.db", vector_store: Optional[VectorStore] = None, config=None):
        """Initialize the SQLite document database.

        Args:
            db_path: Path to the SQLite database file
            vector_store: Optional vector store instance for embeddings
            config: Configuration object
        """
        self.db_path = Path(db_path)
        self.vector_store = vector_store
        self.config = config
        self._local = threading.local()  # Thread-local storage for connections

        # Create database and tables
        self._create_tables()

    @property
    def connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            self._local.conn.execute("PRAGMA synchronous=NORMAL")  # Balance speed/safety
            self._local.conn.execute("PRAGMA cache_size=10000")  # 10MB cache
            self._local.conn.row_factory = sqlite3.Row  # Dict-like rows
        return self._local.conn

    def _create_tables(self):
        """Create database tables and indexes."""
        with self._get_cursor() as cursor:
            # Documents table with JSON metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_preview TEXT,
                    categories TEXT,
                    classification_date TEXT,
                    metadata TEXT,  -- JSON field
                    file_hash TEXT,
                    embedding_stored BOOLEAN DEFAULT FALSE,
                    created_at REAL,
                    updated_at REAL
                )
            ''')

            # Indexes for fast lookups (critical for performance)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON documents(file_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_categories ON documents(categories)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embedding_stored ON documents(embedding_stored)')

    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursors with automatic commit/rollback."""
        conn = self.connection
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def store_classification(self, file_path: str, content: str, categories: str,
                           metadata: Optional[Dict[str, Any]] = None, file_hash: Optional[str] = None) -> int:
        """Store a document classification in the database.

        Args:
            file_path: Path to the original file
            content: Extracted text content
            categories: Classification categories
            metadata: Optional additional metadata
            file_hash: Optional file hash for duplicate detection

        Returns:
            Document ID in the database
        """
        now = datetime.now().timestamp()

        with self._get_cursor() as cursor:
            # Check if document exists
            cursor.execute('SELECT id FROM documents WHERE file_path = ?', (str(file_path),))
            existing = cursor.fetchone()

            doc_data = {
                'file_path': str(file_path),
                'filename': Path(file_path).name,
                'content': content,
                'content_preview': content[:500] if len(content) > 500 else content,
                'categories': categories,
                'classification_date': datetime.now().isoformat(),
                'metadata': json.dumps(metadata or {}),
                'file_hash': file_hash,
                'embedding_stored': False,
                'updated_at': now
            }

            if existing:
                # Update existing
                doc_id = existing[0]
                update_fields = ', '.join(f'{k} = ?' for k in doc_data.keys())
                values = list(doc_data.values()) + [doc_id]
                cursor.execute(f'UPDATE documents SET {update_fields} WHERE id = ?', values)
                logger.info(f"Updated classification for {Path(file_path).name} in database")
                return doc_id
            else:
                # Insert new
                doc_data['created_at'] = now
                columns = ', '.join(doc_data.keys())
                placeholders = ', '.join('?' * len(doc_data))
                values = list(doc_data.values())
                cursor.execute(f'INSERT INTO documents ({columns}) VALUES ({placeholders})', values)
                doc_id = cursor.lastrowid
                logger.info(f"Stored classification for {Path(file_path).name} in database (ID: {doc_id})")
                return doc_id

    def get_document(self, file_path: str) -> Optional[Dict]:
        """Get a document by file path.

        Args:
            file_path: Path to the document

        Returns:
            Document dictionary or None if not found
        """
        with self._get_cursor() as cursor:
            cursor.execute('SELECT * FROM documents WHERE file_path = ?', (str(file_path),))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get a document by file hash (FAST - uses index).

        Args:
            file_hash: SHA-256 hash of the file content

        Returns:
            Document dictionary or None if not found
        """
        with self._get_cursor() as cursor:
            cursor.execute('SELECT * FROM documents WHERE file_hash = ?', (file_hash,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row) -> Dict:
        """Convert SQLite Row to dictionary with proper JSON parsing."""
        doc_dict = dict(row)
        # Parse JSON metadata
        if doc_dict.get('metadata'):
            try:
                doc_dict['metadata'] = json.loads(doc_dict['metadata'])
            except json.JSONDecodeError:
                doc_dict['metadata'] = {}
        return doc_dict

    def store_embedding(self, file_path: str, embedding: List[float]) -> bool:
        """Store embedding vector for a document.

        Args:
            file_path: Path to the document
            embedding: Embedding vector

        Returns:
            True if successful
        """
        if not self.vector_store:
            logger.error("Vector store not available - cannot store embeddings")
            return False

        try:
            doc = self.get_document(file_path)
            if not doc:
                logger.warning(f"Document not found for embedding: {file_path}")
                return False

            doc_id_str = str(doc['id'])
            metadata = {
                'file_path': doc['file_path'],
                'filename': doc['filename'],
                'categories': doc['categories'],
                'content_preview': doc['content_preview'],
                'file_hash': doc['file_hash']
            }

            success = self.vector_store.add_embeddings(
                embeddings=[embedding],
                metadata=[metadata],
                ids=[doc_id_str]
            )

            if success:
                with self._get_cursor() as cursor:
                    cursor.execute(
                        'UPDATE documents SET embedding_stored = TRUE, updated_at = ? WHERE id = ?',
                        (datetime.now().timestamp(), doc['id'])
                    )
                logger.debug(f"Stored embedding in vector store for {Path(file_path).name}")
                return True
            else:
                logger.error(f"Failed to store embedding in vector store for {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False

    def search_semantic(self, query_embedding: List[float], top_k: int = 10,
                      threshold: float = None, max_candidates: int = None) -> List[Dict]:
        """Perform semantic search using vector database."""
        if not self.vector_store:
            logger.error("Vector store not available for semantic search")
            return []

        # Use config values if available
        if self.config:
            if threshold is None:
                threshold = self.config.semantic_search_min_threshold
            if max_candidates is None:
                max_candidates = self.config.semantic_search_max_candidates
            debug_enabled = self.config.semantic_search_debug or logger.isEnabledFor(logging.DEBUG)
        else:
            if threshold is None:
                threshold = -1.0
            if max_candidates is None:
                max_candidates = 50
            debug_enabled = logger.isEnabledFor(logging.DEBUG)

        logger.debug(f"Using vector store for semantic search (top_k={top_k}, threshold={threshold}, max_candidates={max_candidates}, debug={debug_enabled})")

        vector_results = self.vector_store.search_similar(query_embedding, top_k=max_candidates, threshold=threshold)

        results = []
        with self._get_cursor() as cursor:
            for doc_id_str, similarity, metadata in vector_results:
                try:
                    doc_id = int(doc_id_str)
                    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
                    row = cursor.fetchone()
                    if row:
                        doc_copy = self._row_to_dict(row)
                        doc_copy['similarity'] = similarity
                        results.append(doc_copy)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing vector search result for ID {doc_id_str}: {e}")
                    continue

        logger.debug(f"Vector store search returned {len(results)} results")
        return results

    def search_by_category(self, category: str) -> List[Dict]:
        """Search documents by category using fast indexed search."""
        with self._get_cursor() as cursor:
            # Use LIKE for substring matching (case-insensitive)
            cursor.execute(
                'SELECT * FROM documents WHERE LOWER(categories) LIKE LOWER(?)',
                (f'%{category}%',)
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            doc_dict = self._row_to_dict(row)
            results.append(doc_dict)

        return results

    def get_all_documents(self) -> List[Dict]:
        """Get all documents from the database."""
        with self._get_cursor() as cursor:
            cursor.execute('SELECT * FROM documents ORDER BY updated_at DESC')
            rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def export_to_json(self, output_path: str) -> bool:
        """Export all documents to a JSON file."""
        try:
            documents = self.get_all_documents()
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_documents': len(documents),
                'documents': documents
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(documents)} documents to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False

    def refresh(self):
        """Refresh database connections (no-op for SQLite - connections are persistent)."""
        logger.info("SQLite database connections refreshed")

    def close(self):
        """Close all database connections."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
        if self.vector_store:
            self.vector_store.close()

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Get a single document by its ID.

        Args:
            doc_id: The document ID

        Returns:
            Document dictionary or None if not found
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting document by ID {doc_id}: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self._get_cursor() as cursor:
            cursor.execute('SELECT COUNT(*) as total_docs FROM documents')
            total_docs = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) as embedded_docs FROM documents WHERE embedding_stored = TRUE')
            embedded_docs = cursor.fetchone()[0]

        return {
            'total_documents': total_docs,
            'embedded_documents': embedded_docs,
            'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }
