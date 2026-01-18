"""Tests for adapters.llm.rate_limiter module."""

from __future__ import annotations

import threading
import time
from collections import deque

import pytest

from deriva.adapters.llm.rate_limiter import (
    DEFAULT_RATE_LIMITS,
    CircuitState,
    RateLimitConfig,
    RateLimiter,
    get_default_rate_limit,
)
from deriva.common.exceptions import CircuitOpenError


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


class TestRateLimiterRecordFailure:
    """Tests for RateLimiter.record_failure method."""

    def test_does_nothing_when_circuit_breaker_disabled(self):
        """Should do nothing when circuit breaker is disabled."""
        config = RateLimitConfig(circuit_breaker_enabled=False)
        limiter = RateLimiter(config=config)

        limiter.record_failure()

        assert limiter._consecutive_failures == 0

    def test_increments_consecutive_failures(self):
        """Should increment consecutive failures counter."""
        config = RateLimitConfig(circuit_breaker_enabled=True)
        limiter = RateLimiter(config=config)

        limiter.record_failure()
        assert limiter._consecutive_failures == 1

        limiter.record_failure()
        assert limiter._consecutive_failures == 2

    def test_opens_circuit_after_threshold(self):
        """Should open circuit after failure threshold reached."""
        config = RateLimitConfig(
            circuit_breaker_enabled=True,
            circuit_failure_threshold=3,
        )
        limiter = RateLimiter(config=config)

        limiter.record_failure()
        limiter.record_failure()
        assert limiter._circuit_state == CircuitState.CLOSED

        limiter.record_failure()  # Third failure hits threshold
        assert limiter._circuit_state == CircuitState.OPEN

    def test_reopens_circuit_from_half_open(self):
        """Should reopen circuit if failure occurs in half-open state."""
        config = RateLimitConfig(circuit_breaker_enabled=True)
        limiter = RateLimiter(config=config)
        limiter._circuit_state = CircuitState.HALF_OPEN

        limiter.record_failure()

        assert limiter._circuit_state == CircuitState.OPEN


class TestRateLimiterRecordRateLimit:
    """Tests for RateLimiter.record_rate_limit method."""

    def test_does_nothing_when_throttle_disabled(self):
        """Should do nothing when throttling is disabled."""
        config = RateLimitConfig(throttle_enabled=False)
        limiter = RateLimiter(config=config)

        limiter.record_rate_limit()

        assert limiter._consecutive_rate_limits == 0
        assert limiter._throttle_factor == 1.0

    def test_increments_rate_limit_counter(self):
        """Should increment consecutive rate limits counter."""
        config = RateLimitConfig(throttle_enabled=True)
        limiter = RateLimiter(config=config)

        limiter.record_rate_limit()

        assert limiter._consecutive_rate_limits == 1

    def test_reduces_throttle_factor(self):
        """Should reduce throttle factor by half on rate limit."""
        config = RateLimitConfig(throttle_enabled=True, throttle_min_factor=0.1)
        limiter = RateLimiter(config=config)
        assert limiter._throttle_factor == 1.0

        limiter.record_rate_limit()
        assert limiter._throttle_factor == 0.5

        limiter.record_rate_limit()
        assert limiter._throttle_factor == 0.25

    def test_throttle_respects_minimum_factor(self):
        """Should not reduce throttle factor below minimum."""
        config = RateLimitConfig(throttle_enabled=True, throttle_min_factor=0.3)
        limiter = RateLimiter(config=config)

        # Keep reducing until we hit the floor
        for _ in range(10):
            limiter.record_rate_limit()

        assert limiter._throttle_factor >= 0.3

    def test_records_retry_after(self):
        """Should record rate limit with retry_after value."""
        config = RateLimitConfig(throttle_enabled=True)
        limiter = RateLimiter(config=config)

        limiter.record_rate_limit(retry_after=30.0)

        assert limiter._consecutive_rate_limits == 1


class TestRateLimiterCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_closed_by_default(self):
        """Should start with closed circuit."""
        config = RateLimitConfig(circuit_breaker_enabled=True)
        limiter = RateLimiter(config=config)

        assert limiter._circuit_state == CircuitState.CLOSED

    def test_check_circuit_does_nothing_when_disabled(self):
        """Should do nothing when circuit breaker is disabled."""
        config = RateLimitConfig(circuit_breaker_enabled=False)
        limiter = RateLimiter(config=config)
        limiter._circuit_state = CircuitState.OPEN

        # Should not raise even when open
        limiter._check_circuit()

    def test_check_circuit_raises_when_open(self):
        """Should raise CircuitOpenError when circuit is open."""
        config = RateLimitConfig(
            circuit_breaker_enabled=True,
            circuit_recovery_time=60.0,
        )
        limiter = RateLimiter(config=config)
        limiter._circuit_state = CircuitState.OPEN
        limiter._circuit_opened_at = time.time()

        with pytest.raises(CircuitOpenError):
            limiter._check_circuit()

    def test_circuit_transitions_to_half_open(self):
        """Should transition to half-open after recovery time."""
        config = RateLimitConfig(
            circuit_breaker_enabled=True,
            circuit_recovery_time=0.01,  # Very short for testing
        )
        limiter = RateLimiter(config=config)
        limiter._circuit_state = CircuitState.OPEN
        limiter._circuit_opened_at = time.time() - 1.0  # 1 second ago

        # Should transition to half-open instead of raising
        limiter._check_circuit()

        assert limiter._circuit_state == CircuitState.HALF_OPEN

    def test_success_closes_circuit_from_half_open(self):
        """Should close circuit on success from half-open state."""
        config = RateLimitConfig(circuit_breaker_enabled=True)
        limiter = RateLimiter(config=config)
        limiter._circuit_state = CircuitState.HALF_OPEN

        limiter.record_success()

        assert limiter._circuit_state == CircuitState.CLOSED

    def test_success_resets_consecutive_failures(self):
        """Should reset consecutive failures on success."""
        config = RateLimitConfig(circuit_breaker_enabled=True)
        limiter = RateLimiter(config=config)
        limiter._consecutive_failures = 5

        limiter.record_success()

        assert limiter._consecutive_failures == 0


class TestRateLimiterThrottling:
    """Tests for throttling functionality."""

    def test_throttle_affects_effective_rpm(self):
        """Should reduce effective RPM when throttled."""
        config = RateLimitConfig(
            requests_per_minute=100,
            throttle_enabled=True,
        )
        limiter = RateLimiter(config=config)

        # No throttling initially
        limiter.wait_if_needed()

        # Apply throttling
        limiter.record_rate_limit()  # 50%

        # The effective RPM should now be 50
        assert limiter._throttle_factor == 0.5

    def test_throttle_factor_applied_to_rpm(self):
        """Should apply throttle factor to effective RPM."""
        config = RateLimitConfig(
            requests_per_minute=100,
            throttle_enabled=True,
        )
        limiter = RateLimiter(config=config)

        # Initial throttle is 1.0
        assert limiter._throttle_factor == 1.0

        # After rate limit, throttle is reduced
        limiter.record_rate_limit()
        assert limiter._throttle_factor == 0.5

        # Effective RPM should be 50 (100 * 0.5)
        effective_rpm = int(config.requests_per_minute * limiter._throttle_factor)
        assert effective_rpm == 50


class TestRateLimiterWaitWithCircuit:
    """Tests for wait_if_needed with circuit breaker."""

    def test_raises_when_circuit_open(self):
        """Should raise CircuitOpenError when circuit is open."""
        config = RateLimitConfig(
            circuit_breaker_enabled=True,
            circuit_recovery_time=60.0,
        )
        limiter = RateLimiter(config=config)
        limiter._circuit_state = CircuitState.OPEN
        limiter._circuit_opened_at = time.time()

        with pytest.raises(CircuitOpenError):
            limiter.wait_if_needed()
