"""BM25 index manager for keyword-based document search."""
import logging
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
import re

logger = logging.getLogger(__name__)


class BM25Index:
    """Manages BM25 index for keyword-based document search."""

    def __init__(self):
        """Initialize BM25 index."""
        self.bm25 = None
        self.doc_ids = []  # Maps index position to document ID
        self.doc_contents = []  # Tokenized document contents
        self.tokenized_corpus = []  # Tokenized corpus for BM25
        self._is_built = False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase word tokens
        """
        if not text:
            return []
        # Convert to lowercase and split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def build_index(self, documents: List[Dict]) -> bool:
        """Build BM25 index from documents.
        
        Args:
            documents: List of document dictionaries with 'id' and 'content' fields
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.doc_ids = []
            self.doc_contents = []
            self.tokenized_corpus = []

            for doc in documents:
                doc_id = doc.get('id') or doc.get('doc_id')
                if doc_id is None:
                    logger.warning("Document missing ID, skipping")
                    continue

                # Get content (prefer full content over preview)
                content = doc.get('content', '') or doc.get('content_preview', '')
                if not content:
                    logger.warning(f"Document {doc_id} has no content, skipping")
                    continue

                # Tokenize content
                tokens = self._tokenize(content)
                if not tokens:
                    logger.warning(f"Document {doc_id} has no tokens after tokenization, skipping")
                    continue

                self.doc_ids.append(doc_id)
                self.doc_contents.append(content)
                self.tokenized_corpus.append(tokens)

            if not self.tokenized_corpus:
                logger.warning("No valid documents to index")
                self._is_built = False
                return False

            # Build BM25 index
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._is_built = True
            logger.info(f"Built BM25 index with {len(self.doc_ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self._is_built = False
            return False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search BM25 index.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, bm25_score) tuples, sorted by score descending
        """
        if not self._is_built or not self.bm25:
            logger.warning("BM25 index not built, returning empty results")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided to BM25 search")
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.warning("Query has no tokens after tokenization")
                return []

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Create list of (doc_id, score) tuples
            results = []
            for i, score in enumerate(scores):
                if i < len(self.doc_ids):
                    results.append((self.doc_ids[i], float(score)))

            # Sort by score descending and take top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error searching BM25 index: {e}")
            return []

    def update_document(self, doc_id: int, content: str) -> bool:
        """Update a document in the index.
        
        Args:
            doc_id: Document ID
            content: New document content
            
        Returns:
            True if successful, False otherwise
        """
        if not self._is_built:
            logger.warning("BM25 index not built, cannot update document")
            return False

        try:
            # Find document in index
            if doc_id in self.doc_ids:
                idx = self.doc_ids.index(doc_id)
                # Update content and retokenize
                self.doc_contents[idx] = content
                self.tokenized_corpus[idx] = self._tokenize(content)
                # Rebuild BM25 index
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Updated document {doc_id} in BM25 index")
                return True
            else:
                # Document not in index, add it
                return self.add_document(doc_id, content)

        except Exception as e:
            logger.error(f"Error updating document {doc_id} in BM25 index: {e}")
            return False

    def add_document(self, doc_id: int, content: str) -> bool:
        """Add a document to the index.
        
        Args:
            doc_id: Document ID
            content: Document content
            
        Returns:
            True if successful, False otherwise
        """
        if not content:
            logger.warning(f"Document {doc_id} has no content, skipping")
            return False

        try:
            tokens = self._tokenize(content)
            if not tokens:
                logger.warning(f"Document {doc_id} has no tokens after tokenization, skipping")
                return False

            self.doc_ids.append(doc_id)
            self.doc_contents.append(content)
            self.tokenized_corpus.append(tokens)

            # Rebuild BM25 index
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                self._is_built = True
                logger.info(f"Added document {doc_id} to BM25 index")
                return True
            return False

        except Exception as e:
            logger.error(f"Error adding document {doc_id} to BM25 index: {e}")
            return False

    def remove_document(self, doc_id: int) -> bool:
        """Remove a document from the index.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not self._is_built:
            logger.warning("BM25 index not built, cannot remove document")
            return False

        try:
            if doc_id in self.doc_ids:
                idx = self.doc_ids.index(doc_id)
                # Remove from all lists
                del self.doc_ids[idx]
                del self.doc_contents[idx]
                del self.tokenized_corpus[idx]

                # Rebuild BM25 index if there are still documents
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                else:
                    self.bm25 = None
                    self._is_built = False

                logger.info(f"Removed document {doc_id} from BM25 index")
                return True
            else:
                logger.warning(f"Document {doc_id} not found in BM25 index")
                return False

        except Exception as e:
            logger.error(f"Error removing document {doc_id} from BM25 index: {e}")
            return False

    def is_built(self) -> bool:
        """Check if index is built.
        
        Returns:
            True if index is built, False otherwise
        """
        return self._is_built

    def get_document_count(self) -> int:
        """Get number of documents in index.
        
        Returns:
            Number of documents indexed
        """
        return len(self.doc_ids) if self._is_built else 0

    def get_index_details(self, limit: int = 100) -> Dict:
        """Get detailed information about the BM25 index.
        
        Args:
            limit: Maximum number of documents to include in details
            
        Returns:
            Dictionary containing index details including:
            - document_ids: List of document IDs in the index
            - document_details: List of dicts with doc_id, token_count, sample_tokens
            - total_documents: Total number of documents
            - total_tokens: Total number of unique tokens across all documents
            - vocabulary_size: Number of unique tokens in the corpus
        """
        if not self._is_built or not self.bm25:
            return {
                'document_ids': [],
                'document_details': [],
                'total_documents': 0,
                'total_tokens': 0,
                'vocabulary_size': 0
            }
        
        # Get all unique tokens across the corpus
        all_tokens = set()
        for tokens in self.tokenized_corpus:
            all_tokens.update(tokens)
        
        # Build document details
        document_details = []
        for i, doc_id in enumerate(self.doc_ids[:limit]):
            tokens = self.tokenized_corpus[i] if i < len(self.tokenized_corpus) else []
            # Get sample tokens (first 20 unique tokens)
            sample_tokens = list(set(tokens))[:20]
            
            document_details.append({
                'doc_id': doc_id,
                'token_count': len(tokens),
                'unique_token_count': len(set(tokens)),
                'sample_tokens': sample_tokens
            })
        
        return {
            'document_ids': self.doc_ids[:limit],
            'document_details': document_details,
            'total_documents': len(self.doc_ids),
            'total_tokens': sum(len(tokens) for tokens in self.tokenized_corpus),
            'vocabulary_size': len(all_tokens),
            'showing_first_n': min(limit, len(self.doc_ids))
        }
