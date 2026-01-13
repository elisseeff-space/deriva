"""
BusinessFunction Derivation.

A BusinessFunction represents a collection of business behavior based on a
chosen set of criteria (typically required business resources and/or competencies),
closely aligned to an organization, but not necessarily explicitly governed by
the organization.

Graph signals:
- Module/package structures representing business capabilities
- Groups of related methods/classes
- Service layers and domain modules
- High-level organizational code structures

Filtering strategy:
1. Query Module and Package nodes
2. Filter for business-relevant modules
3. Exclude utility/infrastructure modules
4. Focus on domain-specific capabilities

LLM role:
- Identify which modules represent business functions
- Generate meaningful function names
- Write documentation describing the business capability

ArchiMate Layer: Business Layer
ArchiMate Type: BusinessFunction

Typical Sources:
    - Module/package nodes representing business domains
    - High-level directory structures (e.g., orders/, invoices/)
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


class BusinessFunctionDerivation(PatternBasedDerivation):
    """
    BusinessFunction element derivation.

    Uses pattern-based filtering to identify business functions
    from Module and Package nodes representing business capabilities.
    """

    ELEMENT_TYPE = "BusinessFunction"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="BusinessObject",
            rel_type="Access",
            description="Business functions access business objects",
        ),
        RelationshipRule(
            target_type="BusinessProcess",
            rel_type="Composition",
            description="Business functions contain business processes",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="BusinessActor",
            rel_type="Assignment",
            description="Business actors perform business functions",
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
        Filter candidates for BusinessFunction derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by function/capability patterns
        3. Exclude utility/infrastructure modules
        4. Use PageRank to find most important functions
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_functions = [
            c
            for c in filtered
            if self._is_likely_function(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_function(c.name, include_patterns, exclude_patterns)
        ]

        likely_functions = filter_by_pagerank(
            likely_functions, top_n=max_candidates // 2
        )

        remaining_slots = max_candidates - len(likely_functions)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_functions.extend(others)

        self.logger.debug(
            "BusinessFunction filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_functions),
        )

        return likely_functions[:max_candidates]

    def _is_likely_function(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a module name suggests a business function."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for function patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = BusinessFunctionDerivation()

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
    "BusinessFunctionDerivation",
]
