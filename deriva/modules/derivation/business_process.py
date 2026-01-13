"""
BusinessProcess Derivation.

A BusinessProcess represents a sequence of business behaviors that achieves
a specific outcome such as a defined set of products or business services.

Graph signals:
- Method nodes (functions implementing business logic)
- Nodes with workflow/process naming patterns
- High betweenness centrality (orchestrates other components)
- Route handlers in web applications

Filtering strategy:
1. Query Method nodes from source code
2. Exclude utility/helper methods
3. Prioritize methods with business-relevant names
4. Focus on methods that coordinate activities

LLM role:
- Identify which methods represent business processes
- Generate meaningful process names
- Write documentation describing the process purpose

Relationships:
- OUTBOUND: BusinessProcess -> BusinessObject (Access) - processes access business objects
- INBOUND: None defined

ArchiMate Layer: Business Layer
ArchiMate Type: BusinessProcess

Typical Sources:
    - Method nodes with workflow/orchestration logic
    - Functions with names like process_*, handle_*, execute_*
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


class BusinessProcessDerivation(PatternBasedDerivation):
    """
    BusinessProcess element derivation.

    Uses pattern-based filtering to identify business processes
    from Method nodes representing workflow logic.
    """

    ELEMENT_TYPE = "BusinessProcess"

    OUTBOUND_RULES = [
        RelationshipRule(
            target_type="BusinessObject",
            rel_type="Access",
            description="Business processes access business objects",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = []

    def filter_candidates(
        self,
        candidates: list[Candidate],
        enrichments: dict[str, dict[str, Any]],
        max_candidates: int,
        include_patterns: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        **kwargs: Any,
    ) -> list[Candidate]:
        """Filter candidates for BusinessProcess derivation."""
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        # Pre-filter: exclude dunder methods
        filtered = [c for c in candidates if c.name and not c.name.startswith("__")]

        likely_processes = [
            c
            for c in filtered
            if self._is_likely_process(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_process(c.name, include_patterns, exclude_patterns)
        ]

        likely_processes = filter_by_pagerank(
            likely_processes, top_n=max_candidates // 2
        )

        remaining_slots = max_candidates - len(likely_processes)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_processes.extend(others)

        self.logger.debug(
            "BusinessProcess filter: %d total -> %d final",
            len(candidates),
            len(likely_processes),
        )

        return likely_processes[:max_candidates]

    def _is_likely_process(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a method name suggests a business process."""
        if not name:
            return False

        name_lower = name.lower()

        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = BusinessProcessDerivation()

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
    "BusinessProcessDerivation",
]
