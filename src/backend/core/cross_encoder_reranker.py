"""Cross-Encoder Reranking and Reciprocal Rank Fusion (RRF) for hybrid document retrieval."""
import logging
import time
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RerankingScore:
    """Container for re-ranking scores from different rankers."""
    doc_id: Any
    filename: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rag_relevance: float = 0.0
    cross_encoder_score: float = 0.0
    rrf_score: float = 0.0  # Final RRF-combined score


class CrossEncoderReranker:
    """Cross-encoder for joint query-document scoring and Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, endpoint: str, model: str, temperature: float = 0.1,
                 timeout: int = 300, max_retries: int = 3, retry_base_delay: float = 1.0,
                 enable_rrf: bool = True, rrf_k: int = 5,
                 rrf_weights: Optional[Dict[str, float]] = None):
        """Initialize cross-encoder reranker.
        
        Args:
            endpoint: OpenAI-compatible API endpoint
            model: Model name for cross-encoder scoring
            temperature: Sampling temperature
            timeout: API timeout in seconds
            max_retries: Maximum retry attempts
            retry_base_delay: Base delay for exponential backoff
            enable_rrf: Enable Reciprocal Rank Fusion
            rrf_k: RRF k parameter (typically 5-10)
            rrf_weights: Weights for each ranker in RRF (semantic, bm25, rag, cross_encoder)
        """
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.enable_rrf = enable_rrf
        self.rrf_k = rrf_k
        self.rrf_weights = rrf_weights or {
            'semantic': 0.4,
            'bm25': 0.2,
            'rag': 0.3,
            'cross_encoder': 0.1
        }
        
        # Validate weights
        total_weight = sum(self.rrf_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Renormalize if needed
            self.rrf_weights = {k: v / total_weight for k, v in self.rrf_weights.items()}
        
        # Initialize OpenAI client (compatible with local Ollama/vLLM servers)
        self.client = OpenAI(base_url=endpoint, api_key="dummy", timeout=timeout)
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if the endpoint is available."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5
            )
            return True
        except Exception as e:
            logger.warning(f"Cross-encoder endpoint unavailable: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if cross-encoder is available."""
        return self._available
    
    def score_query_document(self, query: str, document: Dict, verbose: bool = False) -> float:
        """Score a query-document pair using cross-encoder.
        
        Args:
            query: User query
            document: Document dictionary with content
            verbose: If True, log detailed interactions
            
        Returns:
            Cross-encoder similarity score (0.0 to 1.0)
        """
        if not self._available:
            logger.debug("Cross-encoder unavailable, returning fallback score")
            return self._fallback_score(query, document)
        
        # Extract document content
        content = document.get('content', '') or document.get('content_preview', '')
        filename = document.get('filename', 'unknown')
        categories = document.get('categories', '')
        
        # Truncate content
        max_length = 8000
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        # Build prompt
        prompt = self._build_cross_encoder_prompt(query, filename, categories, content)
        
        if verbose:
            logger.debug(f"Cross-encoder prompt for {filename}: {prompt[:200]}...")
        
        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=self.temperature,
            )
            raw_response = response.choices[0].message.content
            score = self._parse_score(raw_response)
            
            if verbose:
                logger.debug(f"Cross-encoder score for {filename}: {score}")
            
            return score
            
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed for {filename}: {e}")
            return self._fallback_score(query, document)
    
    def _build_cross_encoder_prompt(self, query: str, filename: str, categories: str, content: str) -> str:
        """Build cross-encoder scoring prompt."""
        return f"""Score how relevant the document is to the query on a scale from 0.0 to 1.0.

QUERY: {query}

DOCUMENT:
- Filename: {filename}
- Categories: {categories}
- Content: {content[:2000]}

OUTPUT: <score> (e.g., 0.85)"""
    
    def _parse_score(self, response: str) -> float:
        """Parse numeric score from LLM response."""
        # Look for number pattern
        match = re.search(r'([0-9]*\.?[0-9]+)', response)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        return 0.5  # Default on parse failure
    
    def _fallback_score(self, query: str, document: Dict) -> float:
        """Compute fallback score based on keyword overlap."""
        query_lower = query.lower()
        content_lower = (document.get('content', '') or document.get('content_preview', '')).lower()
        
        # Count overlapping words
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        if not query_words or not content_words:
            return 0.5
        
        overlap = len(query_words & content_words)
        total = len(query_words | content_words)
        
        return overlap / total if total > 0 else 0.5
    
    def score_documents(self, query: str, documents: List[Dict],
                        verbose: bool = False) -> List[Tuple[Dict, float]]:
        """Score multiple documents against a query.
        
        Args:
            query: User query
            documents: List of documents
            verbose: If True, log detailed interactions
            
        Returns:
            List of (document, score) tuples
        """
        results = []
        for doc in documents:
            try:
                score = self.score_query_document(query, doc, verbose=verbose)
                results.append((doc, score))
            except Exception as e:
                logger.warning(f"Error scoring document {doc.get('filename', 'unknown')}: {e}")
                results.append((doc, 0.5))
        return results
    
    def reciprocal_rank_fusion(self, ranked_lists: Dict[str, List[Tuple[Any, float]]],
                               k: int = 5) -> List[Tuple[Any, float]]:
        """Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Formula: RRF(d) = sum(1.0 / (rank_i(d) + k))
        
        Args:
            ranked_lists: Dictionary of ranker_name -> list of (doc_id, score) tuples sorted by rank
            k: RRF k parameter (default 5)
            
        Returns:
            List of (doc_id, rrf_score) tuples sorted by RRF score (highest first)
        """
        if not ranked_lists:
            return []
        
        # Collect all unique document IDs
        all_doc_ids = set()
        for ranker_name, ranked_list in ranked_lists.items():
            for doc_id, _ in ranked_list:
                all_doc_ids.add(doc_id)
        
        # Compute RRF score for each document
        doc_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            for ranker_name, ranked_list in ranked_lists.items():
                weight = self.rrf_weights.get(ranker_name, 0.0)
                rank = self._get_rank(ranked_list, doc_id)
                rrf_score += weight * (1.0 / (rank + k))
            doc_scores[doc_id] = rrf_score
        
        # Sort by RRF score descending
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def _get_rank(self, ranked_list: List[Tuple[Any, float]], doc_id: Any) -> int:
        """Get the rank of a document in a ranked list (1-indexed)."""
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            if item_id == doc_id:
                return rank
        return len(ranked_list) + 10  # Not found gets low rank
    
    def apply_rrf(self, query: str, documents: List[Dict], scores: Dict[str, List[Tuple[int, float]]],
                 verbose: bool = False) -> List[Dict]:
        """Apply RRF to combine multiple ranking signals.
        
        Args:
            query: User query (for logging)
            documents: List of original documents
            scores: Dictionary of ranker_name -> list of (doc_id, score) tuples
            
        Returns:
            Re-ranked list of documents
        """
        if not self.enable_rrf or not scores:
            if verbose:
                logger.debug("RRF disabled or no scores provided, returning original order")
            return documents
        
        # Build ranked lists from scores
        ranked_lists = {}
        for ranker_name, score_list in scores.items():
            # Sort by score descending
            sorted_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            ranked_lists[ranker_name] = sorted_list
        
        if verbose:
            logger.debug(f"Applying RRF with {len(ranked_lists)} rankers")
            for ranker_name, ranked_list in ranked_lists.items():
                weight = self.rrf_weights.get(ranker_name, 0.0)
                logger.debug(f"  {ranker_name} (weight: {weight}): {len(ranked_list)} items")
        
        # Apply RRF
        rrf_ranking = self.reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        
        # Reorder documents according to RRF ranking
        doc_id_to_doc = {}
        for doc in documents:
            doc_id = doc.get('doc_id') or doc.get('id')
            if doc_id is not None:
                doc_id_to_doc[doc_id] = doc
        
        ranked_docs = []
        for doc_id, rrf_score in rrf_ranking:
            if doc_id in doc_id_to_doc:
                doc = doc_id_to_doc[doc_id]
                doc['rrf_score'] = rrf_score
                ranked_docs.append(doc)
        
        # Add docs that weren't in the ranking
        ranked_doc_ids = set(d.get('doc_id') or d.get('id') for d in ranked_docs)
        for doc in documents:
            doc_id = doc.get('doc_id') or doc.get('id')
            if doc_id not in ranked_doc_ids:
                doc['rrf_score'] = 0.0
                ranked_docs.append(doc)
        
        if verbose:
            logger.debug(f"RRF complete, reordered {len(ranked_docs)} documents")
            for i, doc in enumerate(ranked_docs[:3]):
                doc_id = doc.get('doc_id') or doc.get('id')
                logger.debug(f"  Rank {i+1}: doc_id={doc_id}, rrf_score={doc.get('rrf_score', 'N/A')}")
        
        return ranked_docs


def reciprocal_rank_fusion(rankings: Dict[str, List[Tuple[Any, float]]],
                           k: int = 5) -> List[Tuple[Any, float]]:
    """Standard RRF function without class instance.
    
    Args:
        rankings: Dictionary of ranker_name -> list of (doc_id, score) tuples
        k: RRF k parameter (default 5)
        
    Returns:
        List of (doc_id, rrf_score) tuples sorted by RRF score (highest first)
    """
    if not rankings:
        return []
    
    # Collect all unique document IDs
    all_doc_ids = set()
    for ranked_list in rankings.values():
        for doc_id, _ in ranked_list:
            all_doc_ids.add(doc_id)
    
    # Equal weights
    num_rankers = len(rankings)
    weight = 1.0 / num_rankers
    
    # Compute RRF score for each document
    doc_scores = {}
    for doc_id in all_doc_ids:
        rrf_score = 0.0
        for ranker_name, ranked_list in rankings.items():
            # Find rank
            rank = None
            for r, (item_id, _) in enumerate(ranked_list, start=1):
                if item_id == doc_id:
                    rank = r
                    break
            if rank is None:
                rank = len(ranked_list) + 10
            rrf_score += weight * (1.0 / (rank + k))
        doc_scores[doc_id] = rrf_score
    
    # Sort by RRF score descending
    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
