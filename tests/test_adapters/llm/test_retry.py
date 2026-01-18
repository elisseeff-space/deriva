"""Tests for adapters.llm.retry module."""

from __future__ import annotations

from typing import Any, cast

import pytest
from backoff._typing import Details

from deriva.adapters.llm.retry import (
    RETRIABLE_EXCEPTIONS,
    _extract_retry_after_from_exception,
    classify_exception,
    create_retry_decorator,
    create_retry_with_classification,
    is_transient,
    on_backoff,
    on_giveup,
    retry_on_rate_limit,
    should_giveup,
    wait_with_retry_after,
)
from deriva.common.exceptions import (
    CircuitOpenError,
    RateLimitError,
    TransientError,
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

        # Cast to Any because backoff adds 'exception' at runtime but it's not in the TypedDict
        details = cast(
            Details,
            {
                "target": dummy_func,
                "args": (),
                "kwargs": {},
                "tries": 2,
                "elapsed": 0.0,
                "wait": 2.5,
                "exception": ConnectionError("Connection refused"),
            },
        )

        on_backoff(details)

        assert "Retry 2" in caplog.text
        assert "dummy_func" in caplog.text
        assert "2.50s" in caplog.text

    def test_handles_missing_details(self, caplog):
        """Should handle missing details gracefully."""
        import logging

        caplog.set_level(logging.WARNING)

        on_backoff(cast(Any, {}))

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

        # Cast to Details because backoff adds 'exception' at runtime but it's not in the TypedDict
        details = cast(
            Details,
            {
                "target": dummy_func,
                "args": (),
                "kwargs": {},
                "tries": 5,
                "elapsed": 0.0,
                "exception": TimeoutError("Request timed out"),
            },
        )

        on_giveup(details)

        assert "Giving up" in caplog.text
        assert "dummy_func" in caplog.text
        assert "5 attempts" in caplog.text

    def test_handles_missing_details(self, caplog):
        """Should handle missing details gracefully."""
        import logging

        caplog.set_level(logging.ERROR)

        on_giveup(cast(Any, {}))

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


class TestClassifyException:
    """Tests for classify_exception function."""

    def test_classifies_rate_limit_error(self):
        """Should classify RateLimitError as rate_limited."""
        exc = RateLimitError("Rate limited", retry_after=30.0)
        category, retry_after = classify_exception(exc)
        assert category == "rate_limited"
        assert retry_after == 30.0

    def test_classifies_transient_error(self):
        """Should classify TransientError as transient."""
        exc = TransientError("Temporary failure")
        category, retry_after = classify_exception(exc)
        assert category == "transient"
        assert retry_after is None

    def test_classifies_circuit_open_error(self):
        """Should classify CircuitOpenError as permanent."""
        exc = CircuitOpenError("Circuit is open")
        category, retry_after = classify_exception(exc)
        assert category == "permanent"
        assert retry_after is None

    def test_classifies_connection_error(self):
        """Should classify ConnectionError as transient."""
        exc = ConnectionError("Connection refused")
        category, retry_after = classify_exception(exc)
        assert category == "transient"
        assert retry_after is None

    def test_classifies_timeout_error(self):
        """Should classify TimeoutError as transient."""
        exc = TimeoutError("Request timed out")
        category, retry_after = classify_exception(exc)
        assert category == "transient"
        assert retry_after is None

    def test_classifies_os_error(self):
        """Should classify OSError as transient."""
        exc = OSError("Network unreachable")
        category, retry_after = classify_exception(exc)
        assert category == "transient"
        assert retry_after is None

    def test_classifies_unknown_exception_as_transient(self):
        """Should classify unknown exceptions as transient by default."""
        exc = RuntimeError("Unknown error")
        category, retry_after = classify_exception(exc)
        assert category == "transient"
        assert retry_after is None


class TestExtractRetryAfterFromException:
    """Tests for _extract_retry_after_from_exception function."""

    def test_returns_none_for_no_body(self):
        """Should return None when exception has no body."""
        exc = Exception("No body")
        result = _extract_retry_after_from_exception(exc)
        assert result is None

    def test_extracts_retry_after_from_dict_body(self):
        """Should extract retry_after from dict body."""
        exc = Exception("Has body")
        exc.body = {"retry_after": 30.0}  # type: ignore
        result = _extract_retry_after_from_exception(exc)
        assert result == 30.0

    def test_extracts_retry_after_with_hyphen(self):
        """Should extract retry-after with hyphen."""
        exc = Exception("Has body")
        exc.body = {"retry-after": 45.0}  # type: ignore
        result = _extract_retry_after_from_exception(exc)
        assert result == 45.0

    def test_extracts_retry_after_camel_case(self):
        """Should extract retryAfter in camelCase."""
        exc = Exception("Has body")
        exc.body = {"retryAfter": 60.0}  # type: ignore
        result = _extract_retry_after_from_exception(exc)
        assert result == 60.0

    def test_extracts_from_nested_error_object(self):
        """Should extract retry_after from nested error object."""
        exc = Exception("Has nested body")
        exc.body = {"error": {"retry_after": 15.0}}  # type: ignore
        result = _extract_retry_after_from_exception(exc)
        assert result == 15.0

    def test_returns_none_for_invalid_value(self):
        """Should return None for non-numeric values."""
        exc = Exception("Has body")
        exc.body = {"retry_after": "not-a-number"}  # type: ignore
        result = _extract_retry_after_from_exception(exc)
        assert result is None

    def test_returns_none_for_non_dict_body(self):
        """Should return None for non-dict body."""
        exc = Exception("Has body")
        exc.body = "string body"  # type: ignore
        result = _extract_retry_after_from_exception(exc)
        assert result is None


class TestIsTransient:
    """Tests for is_transient function."""

    def test_returns_true_for_transient_error(self):
        """Should return True for transient errors."""
        exc = TransientError("Temporary")
        assert is_transient(exc) is True

    def test_returns_true_for_rate_limit_error(self):
        """Should return True for rate limit errors."""
        exc = RateLimitError("Rate limited")
        assert is_transient(exc) is True

    def test_returns_false_for_permanent_error(self):
        """Should return False for permanent errors."""
        exc = CircuitOpenError("Circuit open")
        assert is_transient(exc) is False


class TestShouldGiveup:
    """Tests for should_giveup function."""

    def test_returns_true_for_permanent_error(self):
        """Should return True for permanent errors."""
        exc = CircuitOpenError("Circuit open")
        assert should_giveup(exc) is True

    def test_returns_false_for_transient_error(self):
        """Should return False for transient errors."""
        exc = TransientError("Temporary")
        assert should_giveup(exc) is False

    def test_returns_false_for_rate_limit_error(self):
        """Should return False for rate limit errors."""
        exc = RateLimitError("Rate limited")
        assert should_giveup(exc) is False


class TestWaitWithRetryAfter:
    """Tests for wait_with_retry_after generator."""

    def test_initial_yield_is_zero(self):
        """Should yield 0.0 initially."""
        gen = wait_with_retry_after()
        first_value = next(gen)
        assert first_value == 0.0

    def test_uses_retry_after_when_available(self):
        """Should use retry_after from exception when available."""
        gen = wait_with_retry_after(base=2.0, max_value=60.0)
        next(gen)  # Initial yield

        exc = RateLimitError("Rate limited", retry_after=10.0)
        wait_time = gen.send(exc)
        assert wait_time == 10.0

    def test_caps_retry_after_at_max_value(self):
        """Should cap retry_after at max_value."""
        gen = wait_with_retry_after(base=2.0, max_value=5.0)
        next(gen)

        exc = RateLimitError("Rate limited", retry_after=100.0)
        wait_time = gen.send(exc)
        assert wait_time == 5.0

    def test_uses_exponential_backoff_without_retry_after(self):
        """Should use exponential backoff when no retry_after."""
        gen = wait_with_retry_after(base=1.0, max_value=60.0)
        next(gen)

        exc = ConnectionError("No retry after")
        wait_time = gen.send(exc)
        # With jitter, wait_time should be between 0 and base * 2^0 = 1.0
        assert 0 <= wait_time <= 1.0


class TestCreateRetryWithClassification:
    """Tests for create_retry_with_classification function."""

    def test_returns_decorator(self):
        """Should return a callable decorator."""
        decorator = create_retry_with_classification()
        assert callable(decorator)

    def test_decorated_function_works_on_success(self):
        """Should allow successful function calls."""
        decorator = create_retry_with_classification(max_retries=2)

        @decorator
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retries_on_transient_error(self):
        """Should retry on transient errors."""
        call_count = 0
        decorator = create_retry_with_classification(max_retries=3, base_delay=0.01)

        @decorator
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2

    def test_gives_up_on_permanent_error(self):
        """Should give up immediately on permanent errors."""
        call_count = 0
        decorator = create_retry_with_classification(max_retries=3, base_delay=0.01)

        @decorator
        def raises_permanent():
            nonlocal call_count
            call_count += 1
            raise CircuitOpenError("Circuit is open")

        with pytest.raises(CircuitOpenError):
            raises_permanent()

        # Should only be called once (no retries for permanent errors)
        assert call_count == 1
