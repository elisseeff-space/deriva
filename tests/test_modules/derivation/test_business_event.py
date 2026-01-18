"""Tests for modules.derivation.business_event module."""

from __future__ import annotations

from deriva.modules.derivation.base import Candidate
from deriva.modules.derivation.business_event import (
    EVENT_DECORATOR_PATTERNS,
    BusinessEventDerivation,
)


class TestBusinessEventDerivationInit:
    """Tests for BusinessEventDerivation initialization."""

    def test_element_type_is_business_event(self):
        """Should have BusinessEvent as element type."""
        derivation = BusinessEventDerivation()
        assert derivation.ELEMENT_TYPE == "BusinessEvent"

    def test_has_relationship_rules(self):
        """Should have outbound and inbound relationship rules defined."""
        derivation = BusinessEventDerivation()
        assert hasattr(derivation, "OUTBOUND_RULES")
        assert hasattr(derivation, "INBOUND_RULES")
        assert len(derivation.OUTBOUND_RULES) > 0
        assert len(derivation.INBOUND_RULES) > 0


class TestHasEventDecorator:
    """Tests for _has_event_decorator method."""

    def test_returns_true_for_webhook_decorator(self):
        """Should return True for webhook decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="handle_webhook",
            labels=["Method"],
            properties={"decorators": ["webhook"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_event_handler_decorator(self):
        """Should return True for event_handler decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="on_message",
            labels=["Method"],
            properties={"decorators": ["event_handler"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_celery_task_decorator(self):
        """Should return True for celery.task decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="process_task",
            labels=["Method"],
            properties={"decorators": ["celery.task"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_signal_decorator(self):
        """Should return True for signal decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="on_save",
            labels=["Method"],
            properties={"decorators": ["signal.connect"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_listener_decorator(self):
        """Should return True for listener decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="on_event",
            labels=["Method"],
            properties={"decorators": ["listener"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_subscriber_decorator(self):
        """Should return True for subscriber decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="handle_message",
            labels=["Method"],
            properties={"decorators": ["subscriber"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_consumer_decorator(self):
        """Should return True for consumer decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="consume_message",
            labels=["Method"],
            properties={"decorators": ["consumer"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_true_for_callback_decorator(self):
        """Should return True for callback decorator."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="on_complete",
            labels=["Method"],
            properties={"decorators": ["callback"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_returns_false_for_no_decorators(self):
        """Should return False when no decorators present."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="process_data",
            labels=["Method"],
            properties={},
        )
        assert derivation._has_event_decorator(candidate) is False

    def test_returns_false_for_empty_decorators(self):
        """Should return False when decorators list is empty."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="process_data",
            labels=["Method"],
            properties={"decorators": []},
        )
        assert derivation._has_event_decorator(candidate) is False

    def test_returns_false_for_non_event_decorators(self):
        """Should return False for non-event decorators."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="get_data",
            labels=["Method"],
            properties={"decorators": ["cached", "route", "property"]},
        )
        assert derivation._has_event_decorator(candidate) is False

    def test_handles_non_string_decorators(self):
        """Should handle non-string items in decorators list."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="mixed_handler",
            labels=["Method"],
            properties={"decorators": [None, 123, "event_handler"]},
        )
        assert derivation._has_event_decorator(candidate) is True

    def test_case_insensitive_matching(self):
        """Should match decorators case-insensitively."""
        derivation = BusinessEventDerivation()
        candidate = Candidate(
            node_id="1",
            name="handler",
            labels=["Method"],
            properties={"decorators": ["WEBHOOK_HANDLER"]},
        )
        assert derivation._has_event_decorator(candidate) is True


class TestFilterCandidates:
    """Tests for filter_candidates method."""

    def test_prioritizes_decorator_handlers(self):
        """Should prioritize candidates with event decorators."""
        derivation = BusinessEventDerivation()
        candidates = [
            Candidate(
                node_id="1",
                name="regular_method",
                labels=["Method"],
                properties={},
            ),
            Candidate(
                node_id="2",
                name="event_handler",
                labels=["Method"],
                properties={"decorators": ["webhook"]},
            ),
        ]
        enrichments = {}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=10,
            include_patterns=set(),
            exclude_patterns=set(),
        )

        # Decorator handlers should be prioritized
        assert len(result) > 0
        decorator_found = any(c.properties.get("has_event_decorator") for c in result)
        assert decorator_found

    def test_filters_empty_names(self):
        """Should filter out candidates with empty names."""
        derivation = BusinessEventDerivation()
        candidates = [
            Candidate(
                node_id="1",
                name="",
                labels=["Method"],
                properties={},
            ),
            Candidate(
                node_id="2",
                name="valid_handler",
                labels=["Method"],
                properties={},
            ),
        ]
        enrichments = {}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=10,
            include_patterns=set(),
            exclude_patterns=set(),
        )

        # Only valid name should pass
        assert all(c.name for c in result)

    def test_respects_max_candidates(self):
        """Should respect max_candidates limit."""
        derivation = BusinessEventDerivation()
        candidates = [
            Candidate(
                node_id=str(i),
                name=f"handler_{i}",
                labels=["Method"],
                properties={},
            )
            for i in range(20)
        ]
        enrichments = {}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=5,
            include_patterns=set(),
            exclude_patterns=set(),
        )

        assert len(result) <= 5

    def test_applies_pattern_matching(self):
        """Should apply include/exclude pattern matching."""
        derivation = BusinessEventDerivation()
        candidates = [
            Candidate(
                node_id="1",
                name="on_order_created",
                labels=["Method"],
                properties={},
            ),
            Candidate(
                node_id="2",
                name="process_data",
                labels=["Method"],
                properties={},
            ),
        ]
        enrichments = {}

        result = derivation.filter_candidates(
            candidates=candidates,
            enrichments=enrichments,
            max_candidates=10,
            include_patterns={"on_", "event"},
            exclude_patterns=set(),
        )

        assert len(result) > 0


class TestConstants:
    """Tests for module constants."""

    def test_event_decorator_patterns_has_common_patterns(self):
        """Should include common event decorator patterns."""
        assert "webhook" in EVENT_DECORATOR_PATTERNS
        assert "event" in EVENT_DECORATOR_PATTERNS
        assert "signal" in EVENT_DECORATOR_PATTERNS
        assert "listener" in EVENT_DECORATOR_PATTERNS
        assert "celery" in EVENT_DECORATOR_PATTERNS
        assert "callback" in EVENT_DECORATOR_PATTERNS
