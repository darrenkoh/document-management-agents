"""LLM-based file classification using Ollama."""
import logging
import ollama
from typing import Optional, Dict, List
import re
import time

# Try to import httpx exceptions in case ollama uses httpx
try:
    import httpx
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError,
                             httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
except ImportError:
    CONNECTION_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

from src.backend.utils.retry import retry_on_llm_failure, RetryError

logger = logging.getLogger(__name__)


class Classifier:
    """Classifies file content using Ollama LLM."""

    def __init__(self, endpoint: str, model: str, timeout: int = 30, num_predict: int = 200, prompt_template: Optional[str] = None, existing_categories_getter=None, existing_sub_categories_getter=None, summarizer=None, max_retries: int = 3, retry_base_delay: float = 1.0):
        """Initialize classifier.

        Args:
            endpoint: Ollama API endpoint
            model: Model name to use
            timeout: API timeout in seconds
            num_predict: Maximum number of tokens to predict
            prompt_template: Optional custom prompt template with {filename} and {content} placeholders
            existing_categories_getter: Optional callable that returns list of existing categories from database
            existing_sub_categories_getter: Optional callable that returns list of existing sub-categories from database
            summarizer: Optional callable that takes text and returns a summary string
            max_retries: Maximum number of retry attempts for failed API calls
            retry_base_delay: Base delay in seconds between retry attempts
        """
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.num_predict = num_predict
        self.prompt_template = prompt_template
        self.existing_categories_getter = existing_categories_getter
        self.existing_sub_categories_getter = existing_sub_categories_getter
        self.summarizer = summarizer
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = ollama.Client(host=endpoint, timeout=timeout)
        self._cache: Dict[str, str] = {}
    
    def _call_llm_generate(self, prompt: str, options: Dict) -> Dict:
        """Make LLM generate call with retry logic."""
        @retry_on_llm_failure(max_retries=self.max_retries,
                             base_delay=self.retry_base_delay,
                             exceptions=CONNECTION_EXCEPTIONS)
        def _generate():
            return self.client.generate(model=self.model, prompt=prompt, options=options)

        return _generate()

    def classify(self, content: str, filename: Optional[str] = None, verbose: bool = False) -> Optional[tuple[str, float, Optional[List[str]]]]:
        """Classify file content into a category and optional sub-categories.

        Args:
            content: Text content of the file
            filename: Optional filename for context
            verbose: If True, log detailed LLM request and response

        Returns:
            Tuple of (category name, duration in seconds, sub_categories list) or None if classification fails
        """
        # Check cache
        cache_key = f"{filename}:{hash(content[:1000])}" if filename else str(hash(content[:1000]))
        if cache_key in self._cache:
            logger.info(f"Using cached classification for {filename}")
            cached_result = self._cache[cache_key]
            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                # Backward compatibility for old cache entries
                category, duration = cached_result
                return (category, duration, [])
            else:
                # New format: (category, duration, sub_categories)
                return cached_result
        
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

            # Call Ollama with timing and retry logic
            # Note: For reasoning models (like deepseek-r1), we need higher num_predict
            # to allow for thinking tokens and actual response
            start_time = time.time()
            response = self._call_llm_generate(
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent classification
                    'num_predict': self.num_predict,  # Configurable limit
                }
            )
            classification_duration = time.time() - start_time
            
            # Get raw response - check both 'response' and 'thinking' fields
            # Reasoning models (like deepseek-r1) may put output in 'thinking' field
            raw_response = response.get('response', '')
            thinking = response.get('thinking', '')
            done_reason = response.get('done_reason', '')
            
            # Log full response if verbose
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
            
            # Extract category and sub-categories from response
            category, sub_categories = self._extract_category_and_subcategories(raw_response)

            if verbose:
                logger.info(f"Extracted category: '{category}', sub-categories: {sub_categories} (from raw response: '{raw_response}')")

            if category:
                # Cache the result
                self._cache[cache_key] = (category, classification_duration, sub_categories)
                logger.info(f"Classified as: {category}" + (f" with sub-categories: {sub_categories}" if sub_categories else ""))
                return (category, classification_duration, sub_categories)
            else:
                logger.warning(f"Could not extract category from LLM response. Raw response: '{raw_response}'")
                return ("uncategorized", classification_duration, [])
        
        except CONNECTION_EXCEPTIONS as e:
            # Connection/timeout errors should be treated as failures
            logger.error(f"LLM connection error during classification: {e}")
            if verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None  # Return None to indicate failure
        except Exception as e:
            # Other errors (e.g., parsing issues) can fall back to uncategorized
            logger.error(f"Classification error: {e}")
            if verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return ("uncategorized", 0.0, [])  # Return tuple with 0 duration for errors
    
    def _build_prompt(self, content: str, filename: Optional[str] = None) -> str:
        """Build classification prompt for LLM.

        Args:
            content: File content
            filename: Optional filename

        Returns:
            Formatted prompt string
        """
        # Prepare content for classification - use summary for long documents if available
        if len(content) > 3000 and self.summarizer:
            logger.info(f"Document too long ({len(content)} chars), generating summary for classification")
            try:
                summarized_content = self.summarizer(content, max_length=1500)  # Longer summary for classification
                if summarized_content:
                    content_for_classification = summarized_content
                    logger.info(f"Using document summary for classification ({len(summarized_content)} chars)")
                else:
                    # Fallback to truncation if summarization fails
                    content_for_classification = content[:3000]
                    logger.warning("Summarization failed, falling back to truncated content")
            except Exception as e:
                logger.warning(f"Summarization error: {e}, falling back to truncated content")
                content_for_classification = content[:3000]
        else:
            # Use truncated content for short documents or when no summarizer available
            content_for_classification = content[:3000] if len(content) > 3000 else content

        # Get existing categories and sub-categories from database if available
        existing_categories_str = ""
        existing_sub_categories_str = ""
        if self.existing_categories_getter:
            try:
                existing_categories = self.existing_categories_getter()
                if existing_categories:
                    # Extract unique categories from hyphen-separated strings
                    unique_categories = set()
                    for category_str in existing_categories:
                        if category_str and category_str != 'uncategorized':
                            categories = category_str.split('-')
                            unique_categories.update(cat.strip() for cat in categories if cat.strip())
                    if unique_categories:
                        existing_categories_str = f"\n\nExisting categories in your database (prefer these when possible): {', '.join(sorted(unique_categories))}"
            except Exception as e:
                logger.warning(f"Failed to get existing categories: {e}")

        if self.existing_sub_categories_getter:
            try:
                existing_sub_categories = self.existing_sub_categories_getter()
                if existing_sub_categories:
                    # Flatten sub-categories from all documents
                    unique_sub_categories = set()
                    for sub_cat_list in existing_sub_categories:
                        if sub_cat_list:
                            if isinstance(sub_cat_list, list):
                                unique_sub_categories.update(sub_cat.strip() for sub_cat in sub_cat_list if sub_cat and sub_cat.strip())
                            else:
                                # Handle old format if needed
                                pass
                    if unique_sub_categories:
                        existing_sub_categories_str = f"\n\nExisting sub-categories in your database (prefer these when possible): {', '.join(sorted(unique_sub_categories))}"
            except Exception as e:
                logger.warning(f"Failed to get existing sub-categories: {e}")

        # Use custom template if provided, otherwise use default
        if self.prompt_template:
            prompt = self.prompt_template.format(
                filename=filename if filename else 'unknown',
                content=content_for_classification
            )
            # Add existing categories to custom template if available
            if existing_categories_str:
                prompt += existing_categories_str
        else:
            # Default prompt
            prompt = f"""Analyze the following document content and classify it into ONE main category and optionally up to 3 sub-categories.

Main categories include: Finance, Shopping, Travel, Home, School, Other{existing_categories_str}.

Filename: {filename if filename else 'unknown'}

Content:
{content_for_classification}

First, classify this document into ONE main category from the list above. Then, if the document would benefit from additional context beyond the main category (especially if the main category is "Other"), suggest up to 3 specific sub-categories that describe the document's content more precisely.

{existing_sub_categories_str}

Respond in this exact format:
MAIN: [main category]
SUB: [sub-category1, sub-category2, sub-category3] (or leave empty if no sub-categories needed)

Examples:
MAIN: Finance
SUB: invoice, tax, quarterly

MAIN: Other
SUB: medical, prescription, pharmacy

MAIN: Travel
SUB: (leave empty)

Use lowercase for sub-categories and separate with commas."""

        return prompt
    
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

    def _extract_category_and_subcategories(self, response: str) -> tuple[str, List[str]]:
        """Extract main category and sub-categories from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (main_category, sub_categories_list)
        """
        # Clean the response
        response = self._clean_llm_response(response).strip()

        # Initialize defaults
        main_category = "uncategorized"
        sub_categories = []

        # Look for MAIN: and SUB: patterns
        main_match = re.search(r'MAIN:\s*([^\n]+)', response, re.IGNORECASE)
        sub_match = re.search(r'SUB:\s*([^\n]+)', response, re.IGNORECASE)

        # Extract main category
        if main_match:
            main_cat = main_match.group(1).strip()
            if main_cat:
                # Clean and normalize main category
                main_category = self._normalize_category(main_cat)

        # Extract sub-categories
        if sub_match:
            sub_text = sub_match.group(1).strip()
            if sub_text and sub_text.lower() not in ['(leave empty)', 'leave empty', 'none', '']:
                # Split by commas and clean
                sub_cats = [cat.strip() for cat in sub_text.split(',') if cat.strip()]
                # Normalize and limit to 3
                sub_categories = [self._normalize_sub_category(cat) for cat in sub_cats[:3] if cat]

        # If no main category found, try fallback parsing (backward compatibility)
        if main_category == "uncategorized":
            # Try to extract from the whole response as before
            fallback_category = self._extract_category_fallback(response)
            if fallback_category and fallback_category != "uncategorized":
                main_category = fallback_category

        return main_category, sub_categories

    def _normalize_category(self, category: str) -> str:
        """Normalize main category name."""
        category = category.strip().lower()
        # Map to our standard categories
        category_map = {
            'finance': 'Finance',
            'shopping': 'Shopping',
            'travel': 'Travel',
            'home': 'Home',
            'school': 'School',
            'other': 'Other',
            'uncategorized': 'Other'
        }
        return category_map.get(category, category.title())

    def _normalize_sub_category(self, sub_category: str) -> str:
        """Normalize sub-category name."""
        sub_category = sub_category.strip().lower()
        # Remove special chars and normalize
        sub_category = re.sub(r'[^\w\s-]', '', sub_category)
        return sub_category.replace(' ', '_')

    def _extract_category_fallback(self, response: str) -> str:
        """Extract category names from LLM response (fallback for old format).

        Args:
            response: Raw LLM response

        Returns:
            Extracted category name
        """
        # Clean the response
        response = self._clean_llm_response(response).strip().lower()

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

        # Limit to 3 categories and map to main category
        categories = categories[:3]

        # If no categories found, return uncategorized
        if not categories:
            return "uncategorized"

        # For backward compatibility, try to map to main categories
        main_category_map = {
            'invoice': 'Finance',
            'contract': 'Finance',
            'receipt': 'Shopping',
            'letter': 'Other',
            'report': 'Other',
            'resume': 'Other',
            'certificate': 'Other',
            'form': 'Other',
            'statement': 'Finance',
            'manual': 'Other',
            'article': 'Other',
            'email': 'Other',
            'memo': 'Other',
            'note': 'Other',
            'presentation': 'Other',
            'spreadsheet': 'Other',
            'confirmation': 'Other',
            'booking': 'Travel',
            'ticket': 'Travel',
            'itinerary': 'Travel',
            'image': 'Other',
            'other': 'Other'
        }

        # Try to find a main category from the extracted categories
        for cat in categories:
            if cat in main_category_map:
                return main_category_map[cat]

        # Default to Other if no mapping found
        return "Other"

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
        logger.info("Classification cache cleared")

