"""LLM-based file classification using Ollama."""
import logging
import ollama
from typing import Optional, Dict
import re

logger = logging.getLogger(__name__)


class Classifier:
    """Classifies file content using Ollama LLM."""
    
    def __init__(self, endpoint: str, model: str, timeout: int = 30, num_predict: int = 200, prompt_template: Optional[str] = None):
        """Initialize classifier.
        
        Args:
            endpoint: Ollama API endpoint
            model: Model name to use
            timeout: API timeout in seconds
            num_predict: Maximum number of tokens to predict
            prompt_template: Optional custom prompt template with {filename} and {content} placeholders
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.num_predict = num_predict
        self.prompt_template = prompt_template
        self.client = ollama.Client(host=endpoint, timeout=timeout)
        self._cache: Dict[str, str] = {}
    
    def classify(self, content: str, filename: Optional[str] = None, verbose: bool = False) -> Optional[str]:
        """Classify file content into a category.
        
        Args:
            content: Text content of the file
            filename: Optional filename for context
            verbose: If True, log detailed LLM request and response
        
        Returns:
            Category name or None if classification fails
        """
        # Check cache
        cache_key = f"{filename}:{hash(content[:1000])}" if filename else str(hash(content[:1000]))
        if cache_key in self._cache:
            logger.debug(f"Using cached classification for {filename}")
            return self._cache[cache_key]
        
        try:
            # Build prompt
            prompt = self._build_prompt(content, filename)
            
            # Log prompt if verbose
            if verbose:
                logger.info("=" * 80)
                logger.info(f"LLM Classification Request for: {filename or 'unknown'}")
                logger.info("=" * 80)
                logger.info("PROMPT:")
                logger.info("-" * 80)
                logger.info(prompt)
                logger.info("-" * 80)
            
            # Call Ollama
            # Note: For reasoning models (like deepseek-r1), we need higher num_predict
            # to allow for thinking tokens and actual response
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent classification
                    'num_predict': self.num_predict,  # Configurable limit
                }
            )
            
            # Get raw response - check both 'response' and 'thinking' fields
            # Reasoning models (like deepseek-r1) may put output in 'thinking' field
            raw_response = response.get('response', '')
            thinking = response.get('thinking', '')
            done_reason = response.get('done_reason', '')
            
            # Log full response if verbose
            if verbose:
                logger.info("LLM RESPONSE:")
                logger.info("-" * 80)
                logger.info(f"Response field: '{raw_response}'")
                if thinking:
                    logger.info(f"Thinking field (length: {len(thinking)}):")
                    logger.info(f"  First 500 chars: {thinking[:500]}")
                    if len(thinking) > 500:
                        logger.info(f"  Last 200 chars: {thinking[-200:]}")
                logger.info(f"Done reason: {done_reason}")
                logger.info(f"Eval count: {response.get('eval_count', 'N/A')}")
                logger.info("-" * 80)
                logger.info("=" * 80)
            
            # If response is empty but thinking exists, try to extract from thinking
            # This happens with reasoning models that use thinking tokens
            if not raw_response and thinking:
                if verbose:
                    logger.info(f"Response field empty, attempting to extract from thinking field")
                # For reasoning models, try to find the category in the thinking text
                # Look for common category words
                thinking_lower = thinking.strip().lower()
                category_keywords = ['invoice', 'contract', 'receipt', 'letter', 'report', 
                                     'resume', 'certificate', 'form', 'statement', 'manual',
                                     'article', 'email', 'memo', 'note', 'presentation', 
                                     'spreadsheet', 'confirmation', 'booking', 'ticket', 'itinerary',
                                     'flight', 'travel', 'other']
                
                # Try to extract categories from the last part of thinking (where the answer usually is)
                # Look for comma-separated categories or multiple category mentions
                thinking_lines = thinking.strip().split('\n')
                last_lines = '\n'.join(thinking_lines[-10:])  # Check last 10 lines
                last_lines_lower = last_lines.lower()
                
                # Look for comma-separated categories pattern
                comma_pattern = r'\b(?:' + '|'.join(category_keywords) + r')(?:\s*,\s*(?:' + '|'.join(category_keywords) + r')){0,2}\b'
                comma_match = re.search(comma_pattern, last_lines_lower)
                
                if comma_match:
                    raw_response = comma_match.group(0)
                    if verbose:
                        logger.info(f"Extracted comma-separated categories '{raw_response}' from thinking field")
                else:
                    # Check if any keyword appears in thinking (collect up to 3)
                    found_keywords = []
                    for keyword in category_keywords:
                        idx = thinking_lower.rfind(keyword)
                        if idx != -1:
                            found_keywords.append((idx, keyword))
                    
                    if found_keywords:
                        # Sort by position (latest first) and take up to 3 unique keywords
                        found_keywords.sort(reverse=True)
                        unique_keywords = []
                        seen = set()
                        for idx, keyword in found_keywords:
                            if keyword not in seen:
                                unique_keywords.append(keyword)
                                seen.add(keyword)
                                if len(unique_keywords) >= 3:
                                    break
                        raw_response = ','.join(unique_keywords)
                        if verbose:
                            logger.info(f"Extracted categories '{raw_response}' from thinking field")
                    else:
                        # Try to extract from last lines as single category
                        for line in reversed(thinking_lines[-10:]):  # Check last 10 lines
                            line = line.strip().lower()
                            if line and len(line) < 100:  # Allow longer for multiple categories
                                # Check if it looks like categories
                                for keyword in category_keywords:
                                    if keyword in line:
                                        raw_response = keyword
                                        if verbose:
                                            logger.info(f"Extracted '{keyword}' from thinking line: '{line}'")
                                        break
                                if raw_response:
                                    break
            
            # Extract category from response
            category = self._extract_category(raw_response)
            
            if verbose:
                logger.info(f"Extracted category: '{category}' (from raw response: '{raw_response}')")
            
            if category:
                # Cache the result
                self._cache[cache_key] = category
                logger.info(f"Classified as: {category}")
                return category
            else:
                logger.warning(f"Could not extract category from LLM response. Raw response: '{raw_response}'")
                return "uncategorized"
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            if verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return "uncategorized"
    
    def _build_prompt(self, content: str, filename: Optional[str] = None) -> str:
        """Build classification prompt for LLM.
        
        Args:
            content: File content
            filename: Optional filename
        
        Returns:
            Formatted prompt string
        """
        # Truncate content if too long (keep first 3000 chars for context)
        truncated_content = content[:3000] if len(content) > 3000 else content
        
        # Use custom template if provided, otherwise use default
        if self.prompt_template:
            prompt = self.prompt_template.format(
                filename=filename if filename else 'unknown',
                content=truncated_content
            )
        else:
            # Default prompt
            prompt = f"""Analyze the following document content and classify it into up to 3 specific categories.

Common categories include but should be be limited to: invoice, contract, receipt, letter, report, resume, certificate, form, statement, manual, article, email, memo, note, presentation, spreadsheet, confirmation, booking, ticket, itinerary, image, other.

Filename: {filename if filename else 'unknown'}

Content:
{truncated_content}

Based on the content above, classify this document into up to 3 categories. Respond with ONLY the category names separated by commas, nothing else. Use lowercase and single words or short phrases (e.g., "invoice", "contract", "receipt", "confirmation", "booking"). If uncertain, use "other". Examples: "invoice", "confirmation,booking", "contract,legal,agreement".

Categories:"""
        
        return prompt
    
    def _extract_category(self, response: str) -> str:
        """Extract category names from LLM response (up to 3 categories).
        
        Args:
            response: Raw LLM response
        
        Returns:
            Extracted category names joined with "-" and sorted ascendingly
        """
        # Clean the response
        response = response.strip().lower()
        
        # Remove common prefixes/suffixes
        response = re.sub(r'^(category|categories|classification|type|class):\s*', '', response)
        
        # Extract categories - they might be separated by commas, semicolons, or newlines
        # First, try to split by common separators
        categories = []
        
        # Split by comma, semicolon, or newline
        parts = re.split(r'[,;\n]', response)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Remove special chars except hyphens and spaces
            part = re.sub(r'[^\w\s-]', '', part)
            
            # Extract words/phrases
            # Match words separated by spaces or hyphens
            words = re.findall(r'\b[a-z]+(?:\s+[a-z]+)?\b', part)
            if words:
                # Join words with underscore if multiple words
                category = '_'.join(word.strip() for word in words if word.strip())
                if category and category not in categories:
                    categories.append(category)
        
        # If no categories found with separators, try to extract from the whole response
        if not categories:
            # Remove special chars except hyphens and spaces
            cleaned = re.sub(r'[^\w\s-]', '', response)
            # Extract first few words as potential categories
            words = cleaned.split()
            if words:
                # Try to group words into categories (max 3)
                for word in words[:3]:
                    if word and word not in categories:
                        categories.append(word)
        
        # Limit to 3 categories
        categories = categories[:3]
        
        # If no categories found, return uncategorized
        if not categories:
            return "uncategorized"
        
        # Sort categories ascendingly and join with "-"
        categories.sort()
        return "-".join(categories)
    
    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
        logger.debug("Classification cache cleared")

