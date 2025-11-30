"""Embedding generation for semantic search with semantic chunking."""
import logging
import ollama
import re
from typing import List, Optional, Dict, Tuple

# Try to import httpx exceptions in case ollama uses httpx
try:
    import httpx
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError,
                             httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
except ImportError:
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for document content using Ollama with semantic chunking."""

    def __init__(self, endpoint: str, model: str = "qwen3-embedding:8b",
                 summarizer_model: str = "deepseek-r1:8b", timeout: int = 30):
        """Initialize embedding generator.

        Args:
            endpoint: Ollama API endpoint
            model: Embedding model name (default: qwen3-embedding:8b)
            summarizer_model: Model for document summarization (default: deepseek-r1:8b)
            timeout: API timeout in seconds
        """
        self.endpoint = endpoint
        self.model = model
        self.summarizer_model = summarizer_model
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

    def semantic_chunk_text(self, text: str, chunk_size: int = 4000,
                           overlap: int = 200) -> List[str]:
        """Split text into semantic chunks with overlap.

        Args:
            text: Full document text
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []

        # First try to split on natural boundaries (paragraphs, then sentences)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += (" " + paragraph) if current_chunk else paragraph

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If we still have very long chunks, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split long chunks at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = ""
                for sentence in sentences:
                    if len(sub_chunk) + len(sentence) > chunk_size and sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                        overlap_text = sub_chunk[-overlap:] if len(sub_chunk) > overlap else sub_chunk
                        sub_chunk = overlap_text + " " + sentence
                    else:
                        sub_chunk += (" " + sentence) if sub_chunk else sentence
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks

    def generate_document_summary(self, text: str, max_length: int = 1000) -> Optional[str]:
        """Generate a concise summary of the document.

        Args:
            text: Full document text
            max_length: Maximum length of summary in characters

        Returns:
            Document summary or None if generation failed
        """
        # Truncate very long documents for summarization
        if len(text) > 10000:
            text = text[:10000] + "..."

        prompt = f"""Please provide a concise summary of the following document in {max_length // 10} sentences or less, capturing the main topics, key information, and purpose:

{text}

Summary:"""

        try:
            response = self.client.generate(
                model=self.summarizer_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': min(500, max_length // 2),
                }
            )

            summary = response.get('response', '').strip()
            if summary:
                # Clean up and truncate if needed
                summary = summary.replace('Summary:', '').strip()
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
                return summary
            else:
                logger.warning("No summary generated from LLM")
                return None

        except CONNECTION_EXCEPTIONS as e:
            logger.error(f"LLM connection error during summarization: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return None

    def generate_document_embeddings(self, text: str, chunk_size: int = 4000,
                                   overlap: int = 200, generate_summary: bool = True) -> Dict[str, List]:
        """Generate embeddings for a document using semantic chunking and optional summary.

        Args:
            text: Full document text
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            generate_summary: Whether to generate a summary embedding

        Returns:
            Dictionary with 'chunks' (list of chunk embeddings) and 'summary' (summary embedding or None)
        """
        result = {'chunks': [], 'summary': None}

        # Generate chunk embeddings
        chunks = self.semantic_chunk_text(text, chunk_size, overlap)
        logger.debug(f"Split document into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            logger.debug(f"Generating embedding for chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
            embedding = self.generate_embedding(chunk)
            if embedding:
                result['chunks'].append(embedding)
            else:
                logger.warning(f"Failed to generate embedding for chunk {i+1}")

        # Generate summary embedding if requested
        if generate_summary and len(text) > chunk_size:
            logger.debug("Generating document summary...")
            summary_text = self.generate_document_summary(text)
            if summary_text:
                logger.debug(f"Generated summary (length: {len(summary_text)} chars)")
                summary_embedding = self.generate_embedding(summary_text)
                if summary_embedding:
                    result['summary'] = summary_embedding
                    logger.debug("Generated summary embedding")
                else:
                    logger.warning("Failed to generate summary embedding")
            else:
                logger.warning("Failed to generate document summary")

        return result

