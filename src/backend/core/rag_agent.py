"""Agentic RAG implementation for document analysis using LLM."""
import logging
import ollama
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
import time
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.classifier import Classifier
from src.backend.utils.retry import retry_on_llm_failure, RetryError

logger = logging.getLogger(__name__)


class RAGAgent:
    """Agent that analyzes retrieved documents using LLM to determine relevance."""

    def __init__(self, endpoint: str, model: str, timeout: int = 300, num_predict: int = 6000,
                 max_retries: int = 3, retry_base_delay: float = 1.0, answer_prompt_template: Optional[str] = None):
        """Initialize RAG agent.

        Args:
            endpoint: Ollama API endpoint
            model: Model name to use (e.g., 'deepseek-r1:8b')
            timeout: API timeout in seconds
            num_predict: Maximum number of tokens to predict
            max_retries: Maximum number of retry attempts for failed API calls
            retry_base_delay: Base delay in seconds between retry attempts
            answer_prompt_template: Optional custom prompt template for answer generation
                                   Use {query} and {documents} as placeholders
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.num_predict = num_predict
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.answer_prompt_template = answer_prompt_template
        self.client = ollama.Client(host=endpoint, timeout=timeout)

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing encoding tokens and unwanted markup.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response string
        """
        # Remove DeepSeek-style encoding tokens like <|ref|>content<|/ref|><|det|>[[...]]<|/det|>
        response = re.sub(r'<\|[^>]+\|>.*?<\|/[^>]+\|>', '', response, flags=re.DOTALL)

        # Remove any remaining standalone tokens like <|ref|>, <|det|>, etc.
        response = re.sub(r'<\|[^>]+\|>', '', response)

        # Remove any remaining encoding artifacts that might be left
        response = re.sub(r'\[\[.*?\]\]', '', response)  # Remove coordinate-like arrays

        return response.strip()

    def _call_llm_generate(self, prompt: str, options: Dict, stream: bool = False) -> Any:
        """Make LLM generate call with retry logic."""
        @retry_on_llm_failure(max_retries=self.max_retries,
                             base_delay=self.retry_base_delay,
                             exceptions=(Exception,))  # RAG can fail for various reasons
        def _generate():
            generate_kwargs = {
                'model': self.model,
                'prompt': prompt,
                'options': options
            }
            if stream:
                generate_kwargs['stream'] = stream
            return self.client.generate(**generate_kwargs)

        return _generate()

    def analyze_relevance(self, query: str, documents: List[Dict], verbose: bool = False) -> List[Dict]:
        """Analyze retrieved documents and determine their relevance to the query.

        Args:
            query: The search query
            documents: List of retrieved documents with similarity scores
            verbose: If True, log detailed LLM interactions

        Returns:
            List of documents with added relevance analysis
        """
        if not documents:
            return []

        analyzed_docs = []

        for i, doc in enumerate(documents):
            try:
                # Analyze this document's relevance
                relevance_analysis = self._analyze_single_document(query, doc, verbose=verbose)

                # Add analysis to document
                doc_copy = doc.copy()
                doc_copy.update(relevance_analysis)
                analyzed_docs.append(doc_copy)

                if verbose:
                    logger.info(f"Document {i+1}/{len(documents)}: {doc.get('filename', 'unknown')} - "
                              f"Relevance: {doc_copy.get('relevance_score', 'unknown')}")

            except Exception as e:
                logger.error(f"Error analyzing document {doc.get('filename', 'unknown')}: {e}")
                # Keep original document if analysis fails
                analyzed_docs.append(doc)

        # Re-rank documents based on relevance analysis
        return self._rerank_documents(analyzed_docs)

    def _analyze_single_document(self, query: str, document: Dict, verbose: bool = False) -> Dict:
        """Analyze a single document's relevance to the query.

        Args:
            query: The search query
            document: Document dictionary
            verbose: If True, log detailed LLM interactions

        Returns:
            Dictionary with relevance analysis results
        """
        # Build analysis prompt
        prompt = self._build_relevance_prompt(query, document)

        if verbose:
            logger.info("=" * 80)
            logger.info(f"LLM Relevance Analysis for: {document.get('filename', 'unknown')}")
            logger.info("=" * 80)
            logger.info("PROMPT:")
            logger.info("-" * 80)
            logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            logger.info("-" * 80)

        try:
            # Call LLM with retry logic
            start_time = time.time()
            response = self._call_llm_generate(
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for consistent analysis
                    'num_predict': self.num_predict,
                }
            )
            analysis_duration = time.time() - start_time

            # Extract response
            raw_response = response.get('response', '')
            thinking = response.get('thinking', '')

            if verbose:
                logger.info("LLM RESPONSE:")
                logger.info("-" * 80)
                logger.info(f"Response: '{raw_response}'")
                if thinking:
                    logger.info(f"Thinking (length: {len(thinking)}): {thinking[:200]}...")
                logger.info("-" * 80)
                logger.info("=" * 80)

            # Parse the response
            analysis = self._parse_relevance_response(raw_response, thinking)

            # Add timing information
            analysis['analysis_duration'] = analysis_duration

            return analysis

        except RetryError as e:
            logger.error(f"LLM analysis failed after retries: {e.last_exception}")
            return {
                'relevance_score': 0.5,  # Neutral score on error
                'relevance_reasoning': f'Analysis failed after retries: {str(e.last_exception)}',
                'is_relevant': True,  # Default to relevant
                'analysis_duration': 0.0
            }
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {
                'relevance_score': 0.5,  # Neutral score on error
                'relevance_reasoning': f'Analysis failed: {str(e)}',
                'is_relevant': True,  # Default to relevant
                'analysis_duration': 0.0
            }

    def _build_relevance_prompt(self, query: str, document: Dict) -> str:
        """Build the relevance analysis prompt.

        Args:
            query: Search query
            document: Document dictionary

        Returns:
            Formatted prompt for LLM
        """
        filename = document.get('filename', 'unknown')
        content_preview = document.get('content_preview', '')
        categories = document.get('categories', '')
        similarity = document.get('similarity', 0.0)

        # Truncate content if too long (keep first 2000 chars for analysis)
        if len(content_preview) > 2000:
            content_preview = content_preview[:2000] + "..."

        prompt = f"""You are an expert document search analyst. Your job is to carefully evaluate whether a document is relevant to the user’s search query and assign an accurate relevance score.

SEARCH QUERY: "{query}"

DOCUMENT INFORMATION:
- Filename: {filename}
- Categories: {categories}
- Embedding Similarity Score: {similarity:.3f} (for context only; do not copy this as the final score)
- Content Preview (most relevant excerpt):
{content_preview}

TASK:
1. Read the query and the document information carefully.
2. Decide if the document actually helps answer or is directly related to the search query.
3. Assign a relevance score from 0.0 to 1.0:
   • 1.0 = Perfect match, exactly what the user is looking for
   • 0.9 = Extremely relevant, contains the core information needed
   • 0.7–0.8 = Clearly relevant, useful but not perfect
   • 0.4–0.6 = Partially relevant or tangentially related
   • 0.1–0.3 = Only loosely or vaguely related
   • 0.0 = Completely irrelevant or no connection
4. Answer with YES only if the score is 0.7 or higher; otherwise answer NO.

REQUIRED OUTPUT FORMAT (exactly, no extra text):
Score: <0.0–1.0 with one decimal place>
Relevant: <YES or NO>
Reasoning: <2–3 short sentences max, explain the match or lack thereof>

Examples:

Score: 1.0
Relevant: YES
Reasoning: Document is the exact United Airlines flight itinerary for the trip to Japan mentioned in the query.

Score: 0.8
Relevant: YES
Reasoning: Contains the 2024 tax return with Schedule C, directly relevant to the query about business income and deductions.

Score: 0.5
Relevant: NO
Reasoning: Document is a utility bill from 2022; it mentions an address but does not relate to current rental agreements or moving plans.

Score: 0.0
Relevant: NO
Reasoning: This is a grocery receipt; no connection to the query about passport renewal or travel documents."""

        return prompt

    def _parse_relevance_response(self, response: str, thinking: str = "") -> Dict:
        """Parse the LLM response for relevance analysis.

        Args:
            response: Raw LLM response
            thinking: Thinking content from reasoning models

        Returns:
            Dictionary with parsed analysis
        """
        # Clean LLM responses before parsing
        response = self._clean_llm_response(response)
        thinking = self._clean_llm_response(thinking)

        # Use response if available, otherwise try to extract from thinking
        text_to_parse = response.strip() if response.strip() else thinking.strip()

        if not text_to_parse:
            return {
                'relevance_score': 0.5,
                'relevance_reasoning': 'No response from LLM',
                'is_relevant': True
            }

        # Parse the structured response
        score = 0.5  # Default
        is_relevant = True  # Default
        reasoning = text_to_parse  # Default to full text

        # Look for score pattern
        score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', text_to_parse, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to 0-1 range
            except ValueError:
                pass

        # Look for relevant pattern
        relevant_match = re.search(r'Relevant:\s*(YES|NO|TRUE|FALSE)', text_to_parse, re.IGNORECASE)
        if relevant_match:
            relevant_text = relevant_match.group(1).upper()
            is_relevant = relevant_text in ['YES', 'TRUE']

        # Look for reasoning pattern
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|\nScore:|\nRelevant:|$)', text_to_parse,
                                  re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return {
            'relevance_score': score,
            'relevance_reasoning': reasoning,
            'is_relevant': is_relevant
        }

    def _rerank_documents(self, documents: List[Dict]) -> List[Dict]:
        """Re-rank documents based on relevance analysis.

        Args:
            documents: List of documents with relevance analysis

        Returns:
            Re-ranked list of documents
        """
        # Sort by relevance score (highest first), then by original similarity as tiebreaker
        def sort_key(doc):
            relevance_score = doc.get('relevance_score', 0.5)
            similarity = doc.get('similarity', 0.0)
            return (relevance_score, similarity)

        return sorted(documents, key=sort_key, reverse=True)

    def generate_answer(self, query: str, documents: List[Dict], verbose: bool = False) -> Generator[Tuple[str, Optional[List[Dict]]], None, None]:
        """Generate an answer to a question using retrieved documents with streaming support.

        Args:
            query: The user's question
            documents: List of retrieved documents with content
            verbose: If True, log detailed LLM interactions

        Yields:
            Tuples of (chunk, None) for answer chunks, and (full_answer, citations) at the end
        """
        # Validate and log query
        if not query or not query.strip():
            logger.error("Empty query provided to generate_answer")
            yield ("I received an empty question. Please provide a valid question.", None)
            return
        
        logger.info(f"generate_answer called with query: '{query}' and {len(documents) if documents else 0} documents")
        
        if not documents:
            yield ("I couldn't find any relevant documents to answer your question.", None)
            return

        # Build answer generation prompt
        prompt = self._build_answer_prompt(query, documents)
        
        if verbose:
            logger.debug(f"Generated prompt (first 500 chars): {prompt[:500]}")

        if verbose:
            logger.info("=" * 80)
            logger.info(f"Generating answer for query: {query}")
            logger.info(f"Using {len(documents)} documents as context")
            logger.info("=" * 80)

        try:
            # Use streaming generation with retry logic
            stream = self._call_llm_generate(
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Slightly higher for more natural answers
                    'num_predict': self.num_predict,
                },
                stream=True
            )

            full_answer = ""
            for chunk in stream:
                if 'response' in chunk:
                    chunk_text = chunk['response']
                    full_answer += chunk_text
                    yield (chunk_text, None)

            # Clean the answer
            full_answer = self._clean_llm_response(full_answer)

            # Extract citations from the answer
            citations = self._extract_citations(full_answer, documents)

            if verbose:
                logger.info(f"Generated answer (length: {len(full_answer)})")
                logger.info(f"Extracted {len(citations)} citations")

            # Yield final answer with citations
            yield (full_answer, citations)

        except RetryError as e:
            logger.error(f"Answer generation failed after retries: {e.last_exception}")
            error_msg = f"I encountered an error while generating an answer after retries: {str(e.last_exception)}"
            yield (error_msg, None)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            error_msg = f"I encountered an error while generating an answer: {str(e)}"
            yield (error_msg, None)

    def _build_answer_prompt(self, query: str, documents: List[Dict]) -> str:
        """Build the answer generation prompt with document context.

        Args:
            query: User's question
            documents: List of retrieved documents

        Returns:
            Formatted prompt for LLM
        """
        # Validate query
        if not query or not query.strip():
            logger.warning("Empty query provided to _build_answer_prompt")
            query = "No question provided"
        
        # Validate documents
        if not documents:
            logger.warning("No documents provided to _build_answer_prompt")
            return f"""You are an expert assistant. No documents were provided to answer the question.

QUESTION: {query}

Please inform the user that no documents are available to answer their question."""

        # Build document context section
        doc_contexts = []
        content_stats = {'has_content': 0, 'has_preview_only': 0, 'no_content': 0}
        
        for i, doc in enumerate(documents[:10], 1):  # Limit to top 10 documents
            filename = doc.get('filename', 'unknown')
            doc_id = doc.get('id') or doc.get('doc_id', 'unknown')
            # Use full content instead of content_preview for better answer quality
            content = doc.get('content', '')
            if not content:
                content = doc.get('content_preview', '')
                if content:
                    logger.warning(f"Document {doc_id} ({filename}) has no 'content' field, using 'content_preview' ({len(content)} chars)")
                    content_stats['has_preview_only'] += 1
                else:
                    logger.error(f"Document {doc_id} ({filename}) has neither 'content' nor 'content_preview' field!")
                    content_stats['no_content'] += 1
            else:
                content_stats['has_content'] += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Document {doc_id} ({filename}) has 'content' field ({len(content)} chars)")
            
            categories = doc.get('categories', '')
            
            # Truncate content to reasonable length (keep first 12000 chars per doc)
            # This allows access to much more of the document than the 500-char preview
            original_content_len = len(content)
            if len(content) > 12000:
                content = content[:12000] + "..."
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Truncated document {doc_id} content from {original_content_len} to 12000 chars")
            
            doc_context = f"""
[Document {i}]
- ID: {doc_id}
- Filename: {filename}
- Categories: {categories}
- Content:
{content}
"""
            doc_contexts.append(doc_context)

        documents_text = "\n".join(doc_contexts)

        # Ensure query is not empty
        query_text = query.strip() if query else "No question provided"
        if not query_text or query_text == "No question provided":
            logger.error(f"Invalid query in _build_answer_prompt: original='{query}'")
            query_text = "No question was provided - please ask a question"
        
        # Use custom template if provided, otherwise use default
        if self.answer_prompt_template:
            prompt = self.answer_prompt_template.format(
                query=query_text,
                documents=documents_text
            )
        else:
            # Default prompt template
            prompt = f"""You are an expert assistant that answers questions based on provided documents. You must answer ONLY the specific question asked, using ONLY information from the documents provided.

CRITICAL: Answer the EXACT question asked. Do not provide information about related but different topics. If the question asks about property tax, do NOT provide mortgage interest information. If the question asks about a specific year, only provide information for that year.

USER'S QUESTION: {query_text}

DOCUMENTS PROVIDED:
{documents_text}

INSTRUCTIONS:
1. Read the USER'S QUESTION above very carefully. Identify the specific topic, location, and year (if mentioned).
2. Search through the DOCUMENTS PROVIDED above to find information that DIRECTLY answers the USER'S QUESTION.
3. IGNORE any information in the documents that is not directly related to the USER'S QUESTION, even if it seems similar.
4. If the question asks about "property tax", look for property tax amounts, NOT mortgage interest, NOT other taxes.
5. If the question asks about a specific year (e.g., "2014"), only provide information for that exact year.
6. If the question asks about a specific location (e.g., "Dublin"), only provide information for that location.
7. When you find the relevant information, cite the source using [Document N] format (e.g., [Document 1], [Document 2]).
8. If the specific information requested is not available in the documents, clearly state: "Based on the provided documents, I cannot find [specific information requested]."
9. Provide a clear, direct answer that addresses the USER'S QUESTION exactly.
10. At the end of your answer, provide a JSON list of all documents you referenced, in this exact format:
   CITATIONS: [{{"document_number": 1, "reason": "brief reason for citation"}}, {{"document_number": 2, "reason": "brief reason for citation"}}]

Now answer the USER'S QUESTION using ONLY relevant information from the DOCUMENTS PROVIDED above:

ANSWER:"""

        # Log prompt statistics
        total_content_length = sum(len(doc.get('content', '') or doc.get('content_preview', '')) for doc in documents[:10])
        prompt_length = len(prompt)
        logger.info(f"Built answer prompt: query='{query.strip()}', {len(documents[:10])} documents, "
                   f"content stats: {content_stats}, total content={total_content_length} chars, prompt={prompt_length} chars")
        
        if content_stats['has_preview_only'] > 0:
            logger.warning(f"{content_stats['has_preview_only']} documents only have content_preview (500 chars), not full content!")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Prompt preview (first 1000 chars):\n{prompt[:1000]}")

        return prompt

    def _extract_citations(self, answer: str, documents: List[Dict]) -> List[Dict]:
        """Extract cited documents from the answer text.

        Args:
            answer: Generated answer text
            documents: List of source documents

        Returns:
            List of cited document dictionaries with metadata
        """
        citations = []
        cited_indices = set()

        # First, try to extract from structured CITATIONS format at the end
        try:
            # Look for CITATIONS: [...] at the end of the answer
            citations_pattern = r'CITATIONS:\s*\[.*\](?:\s*$)'
            citations_match = re.search(citations_pattern, answer, re.IGNORECASE | re.DOTALL)

            if citations_match:
                import json
                citations_text = citations_match.group(0).replace('CITATIONS:', '').strip()
                citations_data = json.loads(citations_text)

                for citation_item in citations_data:
                    if isinstance(citation_item, dict) and 'document_number' in citation_item:
                        doc_index = citation_item['document_number'] - 1  # Convert to 0-based index
                        if 0 <= doc_index < len(documents):
                            cited_indices.add(doc_index)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse structured citations: {e}")

        # Fallback: Find all [DocumentN] / [Document N] / [DocN] references in the answer.
        # Some models omit the space, so tolerate it.
        citation_pattern = r'\[(?:Document|Doc)\s*(\d+)\]'
        matches = re.finditer(citation_pattern, answer, re.IGNORECASE)
        for match in matches:
            doc_index = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= doc_index < len(documents):
                cited_indices.add(doc_index)

        # Build citation list with document metadata
        for idx in sorted(cited_indices):
            doc = documents[idx]
            citation = {
                'id': doc.get('id') or doc.get('doc_id'),
                'filename': doc.get('filename', 'unknown'),
                'categories': doc.get('categories', ''),
                'similarity': doc.get('similarity', 0.0),
                'content_preview': doc.get('content_preview', '')[:200] + '...' if len(doc.get('content_preview', '')) > 200 else doc.get('content_preview', '')
            }
            citations.append(citation)

        return citations
