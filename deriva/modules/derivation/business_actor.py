"""
BusinessActor Derivation.

A BusinessActor represents a business entity that is capable of performing
behavior. This includes users, roles, departments, or external parties.

Graph signals:
- TypeDefinition nodes with user/role/actor patterns
- BusinessConcept nodes representing people or organizations
- Authentication/authorization related code
- Route handlers with user context

Filtering strategy:
1. Query TypeDefinition and BusinessConcept nodes
2. Filter for actor/role/user patterns
3. Exclude technical/utility classes
4. Focus on entities that perform actions

LLM role:
- Identify which types represent actors
- Generate meaningful actor names
- Write documentation describing the actor's role

ArchiMate Layer: Business Layer
ArchiMate Type: BusinessActor

Typical Sources:
    - TypeDefinition nodes (User, Role, Admin, Customer classes)
    - BusinessConcept nodes representing organizational roles
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


class BusinessActorDerivation(PatternBasedDerivation):
    """
    BusinessActor element derivation.

    Uses pattern-based filtering to identify business actors
    from TypeDefinition and BusinessConcept nodes.
    """

    ELEMENT_TYPE = "BusinessActor"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="BusinessProcess",
            rel_type="Assignment",
            description="Business actors perform business processes",
        ),
        RelationshipRule(
            target_type="BusinessFunction",
            rel_type="Assignment",
            description="Business actors perform business functions",
        ),
        RelationshipRule(
            target_type="ApplicationInterface",
            rel_type="Serving",
            description="Business actors use application interfaces",
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
        """
        Filter candidates for BusinessActor derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by actor patterns
        3. Exclude technical/utility types
        4. Use PageRank to find most important actors
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_actors = [
            c
            for c in filtered
            if self._is_likely_actor(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_actor(c.name, include_patterns, exclude_patterns)
        ]

        likely_actors = filter_by_pagerank(
            likely_actors, top_n=max_candidates // 2, min_pagerank=0.001
        )

        remaining_slots = max_candidates - len(likely_actors)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(
                others, top_n=remaining_slots, min_pagerank=0.001
            )
            likely_actors.extend(others)

        self.logger.debug(
            "BusinessActor filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_actors),
        )

        return likely_actors[:max_candidates]

    def _is_likely_actor(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a type name suggests a business actor."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for actor patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = BusinessActorDerivation()

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
    "BusinessActorDerivation",
]
