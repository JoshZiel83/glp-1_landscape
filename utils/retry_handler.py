"""
Retry logic and rate limiting utilities for API calls.

This module provides decorators and utilities for handling transient failures
and enforcing rate limits when calling external APIs.
"""

import time
import functools
import logging

from typing import Callable, Type, Tuple, Optional
from utils.exceptions import NCBIRateLimitError, NCBITimeoutError, NCBIConnectionError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter to enforce maximum requests per second.

    Usage:
        >>> limiter = RateLimiter(max_per_second=3)
        >>> limiter.wait()  # Sleeps if necessary to maintain rate limit
    """

    def __init__(self, max_per_second: float = 3.0):
        """
        Initialize rate limiter.

        Args:
            max_per_second: Maximum number of requests allowed per second
        """
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self.last_call = 0.0
        logger.debug(f"Initialized rate limiter", extra={"max_per_second": max_per_second})

    def wait(self):
        """Wait if necessary to maintain the rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_call

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        self.last_call = time.time()


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (including first try)
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry (exponential backoff)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called before each retry

    Returns:
        Decorated function that implements retry logic

    Example:
        >>> @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        ... def fetch_data():
        ...     # May fail transiently
        ...     return api_call()
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(f"Attempting {func.__name__} (attempt {attempt}/{max_attempts})")
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"Succeeded on attempt {attempt}/{max_attempts}",
                                   extra={"function": func.__name__})
                    return result

                except exceptions as e:
                    is_last_attempt = (attempt == max_attempts)

                    if is_last_attempt:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}",
                                   exc_info=True,
                                   extra={"function": func.__name__, "attempts": max_attempts})
                        raise

                    logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}",
                                 extra={"function": func.__name__, "attempt": attempt,
                                       "retry_delay": delay})

                    if on_retry:
                        on_retry(attempt, delay, e)

                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper
    return decorator


def retry_on_network_error(max_attempts: int = 3):
    """
    Convenience decorator for retrying on common network errors.

    Retries on: ConnectionError, TimeoutError, NCBIConnectionError, NCBITimeoutError

    Args:
        max_attempts: Maximum number of retry attempts

    Example:
        >>> @retry_on_network_error(max_attempts=3)
        ... def fetch_from_api():
        ...     return requests.get(url)
    """
    return retry_with_backoff(
        max_attempts=max_attempts,
        initial_delay=2.0,
        backoff_factor=2.0,
        exceptions=(ConnectionError, TimeoutError, NCBIConnectionError, NCBITimeoutError)
    )


def rate_limited(limiter: RateLimiter):
    """
    Decorator to enforce rate limiting on a function.

    Args:
        limiter: RateLimiter instance to use

    Example:
        >>> ncbi_limiter = RateLimiter(max_per_second=3)
        >>> @rate_limited(ncbi_limiter)
        ... def query_ncbi():
        ...     return Entrez.esearch(...)
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_ncbi_rate_limiter(has_api_key: bool = False) -> RateLimiter:
    """
    Create a rate limiter configured for NCBI API usage guidelines.

    NCBI Guidelines:
    - Without API key: 3 requests/second
    - With API key: 10 requests/second

    Args:
        has_api_key: Whether user has registered an NCBI API key

    Returns:
        Configured RateLimiter instance
    """
    max_per_second = 10.0 if has_api_key else 3.0
    logger.info(f"Creating NCBI rate limiter",
               extra={"max_per_second": max_per_second, "has_api_key": has_api_key})
    return RateLimiter(max_per_second=max_per_second)



def create_clinicaltrials_rate_limiter(max_per_second: float = 0.8) -> RateLimiter:
    """
    Create a rate limiter configured for ClinicalTrials.gov API usage.

    ClinicalTrials.gov API has a rate limit of ~50 requests per minute per IP.
    Using a conservative default of 0.8 requests per second (48 req/min)
    to stay safely under the limit.

    Args:
        max_per_second: Maximum requests per second (default: 0.8)

    Returns:
        Configured RateLimiter instance
    """
    logger.info(f"Creating ClinicalTrials.gov rate limiter",
               extra={"max_per_second": max_per_second})
    return RateLimiter(max_per_second=max_per_second)



