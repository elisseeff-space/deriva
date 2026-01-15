"""
Retry utilities with exponential backoff.

Uses the backoff library for robust retry handling with jitter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import backoff
from backoff._typing import Details

logger = logging.getLogger(__name__)

# Type variable for decorated function return type
T = TypeVar("T")

# Exceptions that should trigger retry
RETRIABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Includes network errors
)


def on_backoff(details: Details) -> None:
    """Log backoff events."""
    wait = details.get("wait", 0)
    tries = details.get("tries", 0)
    target = details.get("target")
    target_name = getattr(target, "__name__", "unknown") if target else "unknown"
    exception = details.get("exception")

    logger.warning(
        "Retry %d for %s, backing off %.2fs. Error: %s",
        tries,
        target_name,
        wait,
        exception,
    )


def on_giveup(details: Details) -> None:
    """Log when retries are exhausted."""
    tries = details.get("tries", 0)
    target = details.get("target")
    target_name = getattr(target, "__name__", "unknown") if target else "unknown"
    exception = details.get("exception")

    logger.error(
        "Giving up on %s after %d attempts. Final error: %s",
        target_name,
        tries,
        exception,
    )


def create_retry_decorator(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = RETRIABLE_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Create a retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay factor for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorator function

    Example:
        @create_retry_decorator(max_retries=5)
        def flaky_api_call():
            ...
    """
    return backoff.on_exception(
        backoff.expo,
        exception=exceptions,
        max_tries=max_retries + 1,  # backoff counts total tries, not retries
        factor=base_delay,
        max_value=max_delay,
        jitter=backoff.full_jitter,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
    )


def retry_on_rate_limit(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying on rate limit errors (HTTP 429).

    Uses exponential backoff with jitter to handle rate limits gracefully.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Decorator function

    Example:
        @retry_on_rate_limit(max_retries=5)
        def api_call():
            response = requests.get(url)
            if response.status_code == 429:
                raise RateLimitError("Rate limited")
            return response
    """
    # Import here to avoid issues if these aren't installed
    try:
        from httpx import HTTPStatusError
        from pydantic_ai import exceptions as pai_exceptions

        rate_limit_exceptions = (
            ConnectionError,
            TimeoutError,
            HTTPStatusError,
        )

        # Add PydanticAI rate limit exception if available
        if hasattr(pai_exceptions, "RateLimitError"):
            rate_limit_exceptions = (
                *rate_limit_exceptions,
                pai_exceptions.RateLimitError,
            )
    except ImportError:
        rate_limit_exceptions = (ConnectionError, TimeoutError)

    return backoff.on_exception(
        backoff.expo,
        exception=rate_limit_exceptions,
        max_tries=max_retries + 1,
        factor=base_delay,
        max_value=max_delay,
        jitter=backoff.full_jitter,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
    )


__all__ = [
    "create_retry_decorator",
    "retry_on_rate_limit",
    "RETRIABLE_EXCEPTIONS",
]
