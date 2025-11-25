"""Embedding generation for semantic search."""
import logging
import ollama
from typing import List, Optional

# Try to import httpx exceptions in case ollama uses httpx
try:
    import httpx
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError,
                             httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
except ImportError:
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for document content using Ollama."""
    
    def __init__(self, endpoint: str, model: str = "qwen3-embedding:8b", timeout: int = 30):
        """Initialize embedding generator.

        Args:
            endpoint: Ollama API endpoint
            model: Embedding model name (default: qwen3-embedding:8b)
            timeout: API timeout in seconds
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.client = ollama.Client(host=endpoint, timeout=timeout)
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector as list of floats, or None if failed
        """
        try:
            # Truncate text if too long (embedding models have limits)
            # Most embedding models handle up to ~8192 tokens, so ~6000 chars is safe
            truncated_text = text[:6000] if len(text) > 6000 else text

            logger.debug(f"Generating embedding for text (length: {len(truncated_text)} chars)")

            response = self.client.embeddings(
                model=self.model,
                prompt=truncated_text
            )

            embedding = response.get('embedding', [])
            if embedding:
                logger.debug(f"Generated embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.warning("No embedding returned from model")
                return None

        except CONNECTION_EXCEPTIONS as e:
            # Connection/timeout errors should be logged as failures
            logger.error(f"LLM connection error during embedding generation: {e}")
            return None
        except Exception as e:
            # Other errors
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a search query.
        
        Args:
            query: Search query text
        
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        return self.generate_embedding(query)

