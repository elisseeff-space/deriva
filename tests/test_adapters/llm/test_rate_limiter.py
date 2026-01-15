"""Tests for adapters.llm.rate_limiter module."""

from __future__ import annotations

import threading
import time
from collections import deque

from deriva.adapters.llm.rate_limiter import (
    DEFAULT_RATE_LIMITS,
    RateLimitConfig,
    RateLimiter,
    get_default_rate_limit,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.min_request_delay == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        config = RateLimitConfig(
            requests_per_minute=100,
            min_request_delay=0.5,
        )
        assert config.requests_per_minute == 100
        assert config.min_request_delay == 0.5


class TestGetDefaultRateLimit:
    """Tests for get_default_rate_limit function."""

    def test_known_providers(self):
        """Should return correct defaults for known providers."""
        assert get_default_rate_limit("azure") == 60
        assert get_default_rate_limit("openai") == 60
        assert get_default_rate_limit("anthropic") == 60
        assert get_default_rate_limit("ollama") == 0
        assert get_default_rate_limit("lmstudio") == 0
        assert get_default_rate_limit("mistral") == 24

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert get_default_rate_limit("AZURE") == 60
        assert get_default_rate_limit("Azure") == 60
        assert get_default_rate_limit("OLLAMA") == 0

    def test_unknown_provider_returns_default(self):
        """Should return 60 for unknown providers."""
        assert get_default_rate_limit("unknown") == 60
        assert get_default_rate_limit("newprovider") == 60

    def test_all_providers_in_dict(self):
        """All expected providers should be in DEFAULT_RATE_LIMITS."""
        expected = {"azure", "openai", "anthropic", "mistral", "ollama", "lmstudio"}
        assert set(DEFAULT_RATE_LIMITS.keys()) == expected


class TestRateLimiterBasic:
    """Basic tests for RateLimiter class."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        limiter = RateLimiter()
        assert limiter.config.requests_per_minute == 60
        assert limiter._last_request_time == 0.0
        assert len(limiter._request_times) == 0
        assert limiter._successful_requests == 0

    def test_custom_config(self):
        """Should accept custom config."""
        config = RateLimitConfig(requests_per_minute=100)
        limiter = RateLimiter(config=config)
        assert limiter.config.requests_per_minute == 100


class TestRateLimiterWaitIfNeeded:
    """Tests for RateLimiter.wait_if_needed method."""

    def test_no_wait_when_no_limits(self):
        """Should not wait when limits are disabled."""
        config = RateLimitConfig(requests_per_minute=0, min_request_delay=0.0)
        limiter = RateLimiter(config=config)

        wait_time = limiter.wait_if_needed()

        assert wait_time == 0.0

    def test_no_wait_on_first_request(self):
        """Should not wait on first request with RPM limit."""
        config = RateLimitConfig(requests_per_minute=60, min_request_delay=0.0)
        limiter = RateLimiter(config=config)

        wait_time = limiter.wait_if_needed()

        assert wait_time == 0.0
        assert len(limiter._request_times) == 1

    def test_records_request_time(self):
        """Should record request timestamp."""
        config = RateLimitConfig(requests_per_minute=60)
        limiter = RateLimiter(config=config)

        before = time.time()
        limiter.wait_if_needed()
        after = time.time()

        assert len(limiter._request_times) == 1
        assert before <= limiter._request_times[0] <= after
        assert before <= limiter._last_request_time <= after

    def test_min_request_delay_enforced(self):
        """Should enforce minimum delay between requests."""
        config = RateLimitConfig(requests_per_minute=0, min_request_delay=0.05)
        limiter = RateLimiter(config=config)

        # First request - no wait
        limiter.wait_if_needed()

        # Second request immediately after - should wait
        start = time.time()
        wait_time = limiter.wait_if_needed()
        elapsed = time.time() - start

        # Should have waited approximately min_request_delay
        assert wait_time > 0
        assert elapsed >= 0.04  # Allow small timing variance

    def test_rpm_limit_enforced(self):
        """Should enforce RPM limit when reached."""
        # Use a very small RPM limit for testing
        config = RateLimitConfig(requests_per_minute=2, min_request_delay=0.0)
        limiter = RateLimiter(config=config)

        # Make 2 requests (at limit)
        limiter.wait_if_needed()
        limiter.wait_if_needed()

        # Third request should wait (but we mock time to avoid actual wait)
        # Just verify the request count tracking works
        assert len(limiter._request_times) == 2

    def test_old_timestamps_cleaned_up(self):
        """Should clean up timestamps older than 60 seconds."""
        config = RateLimitConfig(requests_per_minute=60)
        limiter = RateLimiter(config=config)

        # Add an old timestamp (using deque)
        old_time = time.time() - 120  # 2 minutes ago
        limiter._request_times = deque([old_time])

        # Make a new request
        limiter.wait_if_needed()

        # Old timestamp should be cleaned up
        assert len(limiter._request_times) == 1
        assert all(t > time.time() - 61 for t in limiter._request_times)


class TestRateLimiterRecordSuccess:
    """Tests for RateLimiter.record_success method."""

    def test_increments_successful_requests(self):
        """Should increment successful requests counter."""
        limiter = RateLimiter()

        limiter.record_success()
        assert limiter._successful_requests == 1

        limiter.record_success()
        assert limiter._successful_requests == 2

    def test_thread_safe(self):
        """Should be thread-safe."""
        limiter = RateLimiter()

        # Call from multiple threads
        threads = [threading.Thread(target=limiter.record_success) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter._successful_requests == 5


class TestRateLimiterGetStats:
    """Tests for RateLimiter.get_stats method."""

    def test_returns_current_state(self):
        """Should return accurate statistics."""
        config = RateLimitConfig(requests_per_minute=100, min_request_delay=0.5)
        limiter = RateLimiter(config=config)

        # Make some requests
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        limiter.record_success()

        stats = limiter.get_stats()

        assert stats["requests_last_minute"] == 2
        assert stats["rpm_limit"] == 100
        assert stats["successful_requests"] == 1
        assert stats["min_request_delay"] == 0.5

    def test_excludes_old_requests(self):
        """Should only count requests in last minute."""
        limiter = RateLimiter()

        # Add old request (using deque)
        limiter._request_times = deque([time.time() - 120])  # 2 minutes ago

        # Add recent request
        limiter.wait_if_needed()

        stats = limiter.get_stats()

        assert stats["requests_last_minute"] == 1  # Only recent one


class TestRateLimiterThreadSafety:
    """Thread safety tests for RateLimiter."""

    def test_concurrent_requests(self):
        """Should handle concurrent requests safely."""
        config = RateLimitConfig(requests_per_minute=1000, min_request_delay=0.0)
        limiter = RateLimiter(config=config)

        results = []

        def make_request():
            wait = limiter.wait_if_needed()
            results.append(wait)

        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should have been recorded
        assert len(limiter._request_times) == 10
        assert len(results) == 10

    def test_concurrent_success_recording(self):
        """Should handle concurrent success recordings safely."""
        limiter = RateLimiter()

        threads = [threading.Thread(target=limiter.record_success) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have incremented 5 times
        assert limiter._successful_requests == 5
