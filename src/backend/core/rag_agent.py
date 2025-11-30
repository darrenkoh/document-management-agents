"""Agentic RAG implementation for document analysis using LLM."""
import logging
import ollama
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.classifier import Classifier

logger = logging.getLogger(__name__)


class RAGAgent:
    """Agent that analyzes retrieved documents using LLM to determine relevance."""

    def __init__(self, endpoint: str, model: str, timeout: int = 300, num_predict: int = 6000):
        """Initialize RAG agent.

        Args:
            endpoint: Ollama API endpoint
            model: Model name to use (e.g., 'deepseek-r1:8b')
            timeout: API timeout in seconds
            num_predict: Maximum number of tokens to predict
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.num_predict = num_predict
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
            # Call LLM
            start_time = time.time()
            response = self.client.generate(
                model=self.model,
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

        prompt = f"""You are an expert document analyst. Analyze whether this document is relevant to the search query.

SEARCH QUERY: "{query}"

DOCUMENT INFORMATION:
- Filename: {filename}
- Categories: {categories}
- Similarity Score: {similarity:.3f}
- Content Preview:
{content_preview}

TASK:
1. Determine if this document is relevant to the search query
2. Provide a relevance score from 0.0 to 1.0 (where 1.0 is highly relevant, 0.0 is not relevant)
3. Give brief reasoning for your assessment

RESPONSE FORMAT:
Score: [0.0-1.0]
Relevant: [YES/NO]
Reasoning: [brief explanation]

Example:
Score: 0.8
Relevant: YES
Reasoning: This document contains flight confirmation details matching the query for travel documents."""

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
