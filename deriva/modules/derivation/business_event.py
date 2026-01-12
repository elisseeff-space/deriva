"""
BusinessEvent Derivation.

A BusinessEvent represents an organizational state change. Events may originate
from the environment (e.g., customers), or be triggered by services, processes,
or other events.

Graph signals:
- Event handler methods (on*, handle*, emit*, trigger*)
- Message/event classes and types
- Callback functions
- Signal definitions

Filtering strategy:
1. Query Method and TypeDefinition nodes
2. Filter for event/handler patterns
3. Exclude utility event handlers
4. Focus on business-relevant events

LLM role:
- Identify which handlers represent business events
- Generate meaningful event names
- Write documentation describing the event trigger and impact

ArchiMate Layer: Business Layer
ArchiMate Type: BusinessEvent

Typical Sources:
    - Method nodes with event handler patterns (on_*, handle_*)
    - Event class definitions and signal handlers
"""

from __future__ import annotations

import logging
from typing import Any

from deriva.modules.derivation.base import (
    Candidate,
    RelationshipRule,
    enrich_candidate,
    filter_by_pagerank,
)
from deriva.modules.derivation.element_base import PatternBasedDerivation

logger = logging.getLogger(__name__)


class BusinessEventDerivation(PatternBasedDerivation):
    """
    BusinessEvent element derivation.

    Uses pattern-based filtering to identify business events
    from event handler methods and event type definitions.
    """

    ELEMENT_TYPE = "BusinessEvent"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="BusinessProcess",
            rel_type="Triggering",
            description="Business events trigger business processes",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="BusinessProcess",
            rel_type="Triggering",
            description="Business processes trigger business events",
        ),
    ]

    def filter_candidates(
        self,
        candidates: list[Candidate],
        enrichments: dict[str, dict[str, Any]],
        max_candidates: int,
        include_patterns: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        """
        Filter candidates for BusinessEvent derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by event patterns
        3. Exclude technical/utility handlers
        4. Use PageRank to find most important events
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_events = [
            c
            for c in filtered
            if self._is_likely_event(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_event(c.name, include_patterns, exclude_patterns)
        ]

        likely_events = filter_by_pagerank(likely_events, top_n=max_candidates // 2)

        remaining_slots = max_candidates - len(likely_events)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_events.extend(others)

        self.logger.debug(
            "BusinessEvent filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_events),
        )

        return likely_events[:max_candidates]

    def _is_likely_event(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a name suggests a business event."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for event patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = BusinessEventDerivation()

ELEMENT_TYPE = _instance.ELEMENT_TYPE
OUTBOUND_RULES = _instance.OUTBOUND_RULES
INBOUND_RULES = _instance.INBOUND_RULES


def filter_candidates(
    candidates: list[Candidate],
    enrichments: dict[str, dict[str, Any]],
    include_patterns: set[str],
    exclude_patterns: set[str],
    max_candidates: int,
) -> list[Candidate]:
    """Backward-compatible filter_candidates function."""
    return _instance.filter_candidates(
        candidates,
        enrichments,
        max_candidates,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )


def generate(
    graph_manager,
    archimate_manager,
    engine,
    llm_query_fn,
    query,
    instruction,
    example,
    max_candidates,
    batch_size,
    existing_elements,
    temperature=None,
    max_tokens=None,
    defer_relationships=False,
):
    """Backward-compatible generate function."""
    return _instance.generate(
        graph_manager=graph_manager,
        archimate_manager=archimate_manager,
        engine=engine,
        llm_query_fn=llm_query_fn,
        query=query,
        instruction=instruction,
        example=example,
        max_candidates=max_candidates,
        batch_size=batch_size,
        existing_elements=existing_elements,
        temperature=temperature,
        max_tokens=max_tokens,
        defer_relationships=defer_relationships,
    )


__all__ = [
    "ELEMENT_TYPE",
    "OUTBOUND_RULES",
    "INBOUND_RULES",
    "filter_candidates",
    "generate",
    "BusinessEventDerivation",
]
