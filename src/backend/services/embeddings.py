"""Embedding generation for semantic search with semantic chunking."""
import logging
from openai import OpenAI
import re
import time
from typing import List, Optional, Dict, Tuple

# Try to import httpx exceptions in case OpenAI uses httpx
try:
    import httpx
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError,
                             httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
except ImportError:
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

from src.backend.utils.retry import retry_on_llm_failure, RetryError

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for document content using OpenAI-compatible LLM APIs with semantic chunking."""

    def __init__(self, endpoint: str, embedding_endpoint: str = None, model: str = "text-embedding-3-small",
                 summarizer_model: str = "gpt-3.5-turbo", timeout: int = 30,
                 max_retries: int = 3, retry_base_delay: float = 1.0,
                 summary_initial_tokens: int = 4000,
                 summary_token_increment: int = 1000):
        """Initialize embedding generator.

        Args:
            endpoint: OpenAI-compatible API endpoint for summarization (e.g., http://localhost:11434/v1 for Ollama with OpenAI compatibility)
            embedding_endpoint: OpenAI-compatible API endpoint for embeddings (defaults to endpoint if not provided)
            model: Embedding model name (default: text-embedding-3-small)
            summarizer_model: Model for document summarization (default: gpt-3.5-turbo)
            timeout: API timeout in seconds
            max_retries: Maximum number of retry attempts for failed API calls
            retry_base_delay: Base delay in seconds between retry attempts
            summary_initial_tokens: Initial token budget for summary generation
            summary_token_increment: Token budget increment on each retry
        """
        self.endpoint = endpoint
        self.embedding_endpoint = embedding_endpoint or endpoint
        self.model = model
        self.summarizer_model = summarizer_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.summary_initial_tokens = summary_initial_tokens
        self.summary_token_increment = summary_token_increment
        # Use a dummy API key since local servers often don't require authentication
        self.client = OpenAI(base_url=endpoint, api_key="dummy", timeout=timeout)
        self.embedding_client = OpenAI(base_url=self.embedding_endpoint, api_key="dummy", timeout=timeout)

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing encoding tokens and unwanted markup.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response string
        """
        # Remove <think>...</think> blocks (reasoning models like DeepSeek-R1, Claude)
        response = re.sub(r'<think>[\s\S]*?</think>', '', response, flags=re.IGNORECASE)

        # Remove DeepSeek-style encoding tokens like <|ref|>content<|/ref|><|det|>[[...]]<|/det|>
        response = re.sub(r'<\|[^>]+\|>.*?<\|/[^>]+\|>', '', response, flags=re.DOTALL)

        # Remove any remaining standalone tokens like <|ref|>, <|det|>, etc.
        response = re.sub(r'<\|[^>]+\|>', '', response)

        # Remove any remaining encoding artifacts that might be left
        response = re.sub(r'\[\[.*?\]\]', '', response)  # Remove coordinate-like arrays

        return response.strip()

    def _call_llm_embeddings(self, model: str, prompt: str) -> List[float]:
        """Make LLM embeddings call with retry logic."""
        @retry_on_llm_failure(max_retries=self.max_retries,
                             base_delay=self.retry_base_delay,
                             exceptions=CONNECTION_EXCEPTIONS)
        def _embeddings():
            response = self.embedding_client.embeddings.create(model=model, input=prompt)
            return response.data[0].embedding

        return _embeddings()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector as list of floats, or None if failed
        """
        try:
            # Truncate text if too long (qwen3-embedding:8b supports 32K tokens, ~128K chars)
            # Using 30K chars (~7.5K tokens) to stay well within limits and ensure quality
            truncated_text = text[:30000] if len(text) > 30000 else text

            logger.info(f"Generating embedding for text (length: {len(truncated_text)} chars)")

            embedding = self._call_llm_embeddings(
                model=self.model,
                prompt=truncated_text
            )

            if embedding:
                logger.info(f"Generated embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.warning("No embedding returned from model")
                return None

        except RetryError as e:
            # All retry attempts failed
            logger.error(f"Embedding generation failed after retries: {e.last_exception}")
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

    def _call_llm_generate(self, model: str, prompt: str, options: Dict) -> Dict:
        """Make LLM generate call with retry logic."""
        @retry_on_llm_failure(max_retries=self.max_retries,
                             base_delay=self.retry_base_delay,
                             exceptions=CONNECTION_EXCEPTIONS)
        def _generate():
            max_tokens = options.get('num_predict', 1000)
            temperature = options.get('temperature', 0.7)

            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Convert OpenAI response format to Ollama-like format for compatibility
            return {
                'response': response.choices[0].message.content,
                'done_reason': 'stop',  # Assume completion was successful
                'eval_count': None,     # Not available in OpenAI API
                'error': None
            }

        return _generate()

    def generate_document_summary(
        self,
        text: str,
        initial_tokens: Optional[int] = None,
        token_increment: Optional[int] = None,
    ) -> Optional[str]:
        """Generate a comprehensive summary of the document.

        Args:
            text: Full document text
            initial_tokens: Starting token budget for the summarizer call
            token_increment: Additional tokens to add on each retry

        Returns:
            Document summary or None if generation failed
        """
        # Truncate very long documents for summarization
        if len(text) > 10000:
            text = text[:10000] + "..."

        initial_tokens = initial_tokens if initial_tokens is not None else self.summary_initial_tokens
        token_increment = token_increment if token_increment is not None else self.summary_token_increment

        prompt = f"""Please provide a comprehensive summary of the following document, capturing all key information, main topics, important details, and purpose:

{text}

Summary:"""

        # Retry on empty summary response
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                # Increase token limit with each retry to handle length issues
                token_limit = initial_tokens + (attempt * token_increment)
                response = self._call_llm_generate(
                    model=self.summarizer_model,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,
                        'num_predict': token_limit,
                    }
                )

                # Check for LLM response errors
                done_reason = response.get('done_reason', '')
                summary = response.get('response', '').strip()

                # Check if the response indicates an error condition
                error_indicators = ['error', 'fail', 'timeout', 'cancel']
                # Also treat 'length' as an error since it means the model hit token limits
                has_error = any(indicator in done_reason.lower() for indicator in error_indicators) or response.get('error') or done_reason.lower() == 'length'

                if has_error:
                    logger.warning(f"LLM response indicates error (done_reason: {done_reason}, error: {response.get('error', 'N/A')})")
                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2.0 ** attempt)  # Exponential backoff
                        logger.warning(f"LLM error detected (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"LLM error after {self.max_retries + 1} attempts (done_reason: {done_reason})")
                        return None

                if summary:
                    # Clean up LLM encoding tokens and other artifacts
                    summary = self._clean_llm_response(summary)
                    summary = summary.replace('Summary:', '').strip()
                    return summary
                else:
                    # Empty summary response - log detailed response info for debugging
                    logger.warning(f"No summary content in LLM response (attempt {attempt + 1}/{self.max_retries + 1})")
                    logger.warning(f"Response details - done_reason: {done_reason}, eval_count: {response.get('eval_count', 'N/A')}")
                    logger.warning(f"Response type: {type(response)}, has 'response' field: {hasattr(response, 'get') and 'response' in str(response)}")

                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2.0 ** attempt)  # Exponential backoff
                        logger.warning(f"No summary generated from LLM (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"No summary generated from LLM after {self.max_retries + 1} attempts")
                        logger.error(f"Final response details - done_reason: {done_reason}, eval_count: {response.get('eval_count', 'N/A')}, response_type: {type(response)}")
                        return None

            except RetryError as e:
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2.0 ** attempt)  # Exponential backoff
                    logger.warning(f"Document summary generation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e.last_exception}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Document summary generation failed after {self.max_retries + 1} attempts: {e.last_exception}")
                    return None
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2.0 ** attempt)  # Exponential backoff
                    logger.warning(f"Error generating document summary (attempt {attempt + 1}/{self.max_retries + 1}): {e}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Error generating document summary after {self.max_retries + 1} attempts: {e}")
                    return None

        # Should not reach here, but just in case
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
            Dictionary with 'chunks' (list of chunk embeddings), 'summary' (summary embedding or None),
            and 'summary_text' (summary text or None)
        """
        result = {'chunks': [], 'summary': None, 'summary_text': None}

        # Generate chunk embeddings
        chunks = self.semantic_chunk_text(text, chunk_size, overlap)
        logger.info(f"Split document into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            logger.info(f"Generating embedding for chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
            embedding = self.generate_embedding(chunk)
            if embedding:
                result['chunks'].append(embedding)
            else:
                logger.warning(f"Failed to generate embedding for chunk {i+1}")

        # Generate summary embedding if requested
        if generate_summary and len(text) > chunk_size:
            logger.info("Generating document summary...")
            summary_text = self.generate_document_summary(text)
            if summary_text:
                logger.info(f"Generated summary (length: {len(summary_text)} chars)")
                result['summary_text'] = summary_text
                summary_embedding = self.generate_embedding(summary_text)
                if summary_embedding:
                    result['summary'] = summary_embedding
                    logger.info("Generated summary embedding")
                else:
                    logger.warning("Failed to generate summary embedding")
            else:
                logger.warning("Failed to generate document summary")

        return result

