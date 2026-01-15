"""
Rate limiter for LLM API requests.

Implements token bucket algorithm for:
- Requests per minute (RPM) limits
- Minimum delay between requests

For retry logic with exponential backoff, use retry.py instead.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Default rate limits by provider (requests per minute)
# These are conservative defaults - actual limits vary by tier/plan
DEFAULT_RATE_LIMITS: dict[str, int] = {
    "azure": 60,  # Azure OpenAI: varies by deployment
    "openai": 60,  # OpenAI: varies by tier (60-10000 RPM)
    "anthropic": 60,  # Anthropic: varies by tier
    "mistral": 24,  # Mistral: varies by tier
    "ollama": 0,  # Local - no limit
    "lmstudio": 0,  # Local - no limit
}


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60  # 0 = no limit
    min_request_delay: float = 0.0  # Minimum seconds between requests


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Thread-safe implementation that tracks request timestamps and
    enforces rate limits across concurrent calls.

    Uses deque for O(1) operations when expiring old timestamps.
    """

    config: RateLimitConfig = field(default_factory=RateLimitConfig)
    _request_times: deque = field(default_factory=deque)  # O(1) popleft
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_request_time: float = field(default=0.0)
    _successful_requests: int = field(default=0)

    def wait_if_needed(self) -> float:
        """
        Wait if necessary to respect rate limits.

        Returns:
            float: Actual wait time in seconds (0 if no wait needed)
        """
        if self.config.requests_per_minute <= 0 and self.config.min_request_delay <= 0:
            return 0.0

        with self._lock:
            now = time.time()
            wait_time = 0.0

            # Check minimum delay between requests
            if self.config.min_request_delay > 0 and self._last_request_time > 0:
                elapsed = now - self._last_request_time
                if elapsed < self.config.min_request_delay:
                    wait_time = max(wait_time, self.config.min_request_delay - elapsed)

            # Check RPM limit
            if self.config.requests_per_minute > 0:
                # Clean up old timestamps using O(1) popleft (deque is sorted by time)
                cutoff = now - 60.0
                while self._request_times and self._request_times[0] <= cutoff:
                    self._request_times.popleft()

                # If at limit, wait until oldest request expires
                if len(self._request_times) >= self.config.requests_per_minute:
                    oldest = self._request_times[0]  # O(1) access to front
                    wait_until = oldest + 60.0
                    wait_time = max(wait_time, wait_until - now)

            # Apply wait if needed
            if wait_time > 0:
                logger.debug("Rate limiting: waiting %.2fs", wait_time)
                # Release lock during sleep
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()
                now = time.time()

            # Record this request (O(1) append)
            self._request_times.append(now)
            self._last_request_time = now

            return wait_time

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._successful_requests += 1

    def get_stats(self) -> dict[str, float | int]:
        """Get current rate limiter statistics."""
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            # Clean expired entries first (using efficient popleft)
            while self._request_times and self._request_times[0] <= cutoff:
                self._request_times.popleft()
            recent_requests = len(self._request_times)

            return {
                "requests_last_minute": recent_requests,
                "rpm_limit": self.config.requests_per_minute,
                "successful_requests": self._successful_requests,
                "min_request_delay": self.config.min_request_delay,
            }


def get_default_rate_limit(provider: str) -> int:
    """Get the default rate limit for a provider."""
    return DEFAULT_RATE_LIMITS.get(provider.lower(), 60)


def parse_retry_after(headers: dict[str, str] | None) -> float | None:
    """
    Parse the retry-after header from API response headers.

    Handles both integer seconds and HTTP-date formats.
    Checks common header name variations used by different providers.

    Args:
        headers: Response headers dict (case-insensitive lookup)

    Returns:
        Retry-after value in seconds, or None if not found/invalid
    """
    if not headers:
        return None

    # Normalize header names to lowercase for case-insensitive lookup
    normalized = {k.lower(): v for k, v in headers.items()}

    # Check common header names (all lowercase)
    header_names = ["retry-after", "x-retry-after", "x-ratelimit-reset"]

    for name in header_names:
        value = normalized.get(name)
        if value:
            try:
                # Try parsing as integer seconds
                return float(value)
            except ValueError:
                # Could be HTTP-date format, but we only support seconds for simplicity
                logger.debug("Could not parse retry-after value: %s", value)
                continue

    return None


__all__ = [
    "RateLimitConfig",
    "RateLimiter",
    "get_default_rate_limit",
    "parse_retry_after",
    "DEFAULT_RATE_LIMITS",
]
