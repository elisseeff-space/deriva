"""Tests for adapters.llm.retry module."""

from __future__ import annotations

import pytest

from deriva.adapters.llm.retry import (
    RETRIABLE_EXCEPTIONS,
    create_retry_decorator,
    on_backoff,
    on_giveup,
    retry_on_rate_limit,
)


class TestRetriableExceptions:
    """Tests for RETRIABLE_EXCEPTIONS constant."""

    def test_contains_connection_error(self):
        """Should include ConnectionError."""
        assert ConnectionError in RETRIABLE_EXCEPTIONS

    def test_contains_timeout_error(self):
        """Should include TimeoutError."""
        assert TimeoutError in RETRIABLE_EXCEPTIONS

    def test_contains_os_error(self):
        """Should include OSError for network errors."""
        assert OSError in RETRIABLE_EXCEPTIONS


class TestOnBackoff:
    """Tests for on_backoff callback function."""

    def test_logs_backoff_event(self, caplog):
        """Should log warning with backoff details."""
        import logging

        caplog.set_level(logging.WARNING)

        def dummy_func():
            pass

        details = {
            "wait": 2.5,
            "tries": 2,
            "target": dummy_func,
            "exception": ConnectionError("Connection refused"),
        }

        on_backoff(details)

        assert "Retry 2" in caplog.text
        assert "dummy_func" in caplog.text
        assert "2.50s" in caplog.text

    def test_handles_missing_details(self, caplog):
        """Should handle missing details gracefully."""
        import logging

        caplog.set_level(logging.WARNING)

        on_backoff({})

        # Should still log without crashing
        assert "Retry 0" in caplog.text


class TestOnGiveup:
    """Tests for on_giveup callback function."""

    def test_logs_giveup_event(self, caplog):
        """Should log error when retries exhausted."""
        import logging

        caplog.set_level(logging.ERROR)

        def dummy_func():
            pass

        details = {
            "tries": 5,
            "target": dummy_func,
            "exception": TimeoutError("Request timed out"),
        }

        on_giveup(details)

        assert "Giving up" in caplog.text
        assert "dummy_func" in caplog.text
        assert "5 attempts" in caplog.text

    def test_handles_missing_details(self, caplog):
        """Should handle missing details gracefully."""
        import logging

        caplog.set_level(logging.ERROR)

        on_giveup({})

        # Should still log without crashing
        assert "Giving up" in caplog.text


class TestCreateRetryDecorator:
    """Tests for create_retry_decorator function."""

    def test_returns_decorator(self):
        """Should return a callable decorator."""
        decorator = create_retry_decorator()
        assert callable(decorator)

    def test_decorated_function_works_on_success(self):
        """Should allow successful function calls."""
        decorator = create_retry_decorator(max_retries=2)

        @decorator
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retries_on_retriable_exception(self):
        """Should retry on retriable exceptions."""
        call_count = 0

        decorator = create_retry_decorator(max_retries=3, base_delay=0.01)

        @decorator
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_gives_up_after_max_retries(self):
        """Should give up after max retries exceeded."""
        decorator = create_retry_decorator(max_retries=2, base_delay=0.01)

        @decorator
        def always_fails():
            raise TimeoutError("Always times out")

        with pytest.raises(TimeoutError):
            always_fails()

    def test_does_not_retry_non_retriable_exceptions(self):
        """Should not retry on non-retriable exceptions."""
        call_count = 0
        decorator = create_retry_decorator(max_retries=3)

        @decorator
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retriable")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # No retries

    def test_custom_exceptions(self):
        """Should retry on custom exception types."""

        class CustomError(Exception):
            pass

        call_count = 0
        decorator = create_retry_decorator(max_retries=2, base_delay=0.01, exceptions=(CustomError,))

        @decorator
        def raises_custom():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("Custom error")
            return "success"

        result = raises_custom()
        assert result == "success"
        assert call_count == 2


class TestRetryOnRateLimit:
    """Tests for retry_on_rate_limit decorator."""

    def test_returns_decorator(self):
        """Should return a callable decorator."""
        decorator = retry_on_rate_limit()
        assert callable(decorator)

    def test_decorated_function_works_on_success(self):
        """Should allow successful function calls."""
        decorator = retry_on_rate_limit(max_retries=2)

        @decorator
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retries_on_connection_error(self):
        """Should retry on ConnectionError."""
        call_count = 0
        decorator = retry_on_rate_limit(max_retries=3, base_delay=0.01)

        @decorator
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Rate limited")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2

    def test_retries_on_timeout_error(self):
        """Should retry on TimeoutError."""
        call_count = 0
        decorator = retry_on_rate_limit(max_retries=3, base_delay=0.01)

        @decorator
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timed out")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2

    def test_handles_httpx_import_error(self):
        """Should work even if httpx not installed."""
        # Just verify it creates a decorator without crashing
        decorator = retry_on_rate_limit(max_retries=2)
        assert callable(decorator)

    def test_gives_up_after_max_retries(self):
        """Should give up after max retries."""
        decorator = retry_on_rate_limit(max_retries=2, base_delay=0.01)

        @decorator
        def always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()
