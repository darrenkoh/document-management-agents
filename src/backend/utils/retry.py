"""Retry utilities for handling failed LLM API calls."""
import time
import logging
from typing import Callable, Any, Optional, Type
from functools import wraps

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception):
        super().__init__(message)
        self.last_exception = last_exception


def retry_on_llm_failure(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0,
                        exceptions: tuple = (Exception,)) -> Callable:
    """Decorator that retries LLM API calls on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which delay increases each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function that will retry on failures
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"LLM API call failed after {max_retries + 1} attempts: {e}")
                        raise RetryError(
                            f"LLM API call failed after {max_retries + 1} attempts",
                            last_exception
                        )

        return wrapper
    return decorator


def create_retry_wrapper(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
    """Create a retry wrapper function for LLM API calls.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds

    Returns:
        Function that can be used to wrap LLM API calls with retry logic
    """
    def retry_wrapper(func: Callable, *args, **kwargs) -> Any:
        """Wrapper that retries the function call on failure."""
        last_exception = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    delay = base_delay * (2.0 ** attempt)  # Exponential backoff
                    logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"LLM API call failed after {max_retries + 1} attempts: {e}")
                    raise RetryError(
                        f"LLM API call failed after {max_retries + 1} attempts",
                        last_exception
                    )

    return retry_wrapper
