"""
TechnologyService Derivation.

A TechnologyService represents an externally visible unit of functionality,
offered by a technology node (e.g., database, message queue, external API).

Graph signals:
- External dependency nodes (imported packages/libraries)
- Nodes with labels like ExternalDependency, Database, API
- High out-degree from application code (many things depend on it)
- Configuration files referencing external services

Filtering strategy:
- Start with ExternalDependency and similar labeled nodes
- Filter to high-importance dependencies (PageRank)
- Exclude standard library and utility packages
- Focus on infrastructure services (databases, queues, APIs)

LLM role:
- Classify which dependencies are technology services vs utilities
- Generate meaningful service names
- Write documentation describing the service's role

Relationships:
- OUTBOUND: TechnologyService -> DataObject (Access) - access databases/files
- OUTBOUND: TechnologyService -> ApplicationService (Serving) - serve app services
- INBOUND: DataObject -> TechnologyService (Realization) - config realizes tech

ArchiMate Layer: Technology Layer
ArchiMate Type: TechnologyService

Typical Sources:
    - ExternalDependency nodes (databases, message queues, APIs)
    - Technology nodes from LLM extraction
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


class TechnologyServiceDerivation(PatternBasedDerivation):
    """
    TechnologyService element derivation.

    Uses pattern-based filtering to identify technology services
    from ExternalDependency nodes.
    """

    ELEMENT_TYPE = "TechnologyService"

    OUTBOUND_RULES = [
        RelationshipRule(
            target_type="DataObject",
            rel_type="Access",
            description="Technology services access data objects (databases, files)",
        ),
        RelationshipRule(
            target_type="ApplicationService",
            rel_type="Serving",
            description="Technology services serve application services",
        ),
    ]

    INBOUND_RULES = [
        RelationshipRule(
            target_type="DataObject",
            rel_type="Realization",
            description="Data objects (config files) realize technology services",
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
        Filter candidates for TechnologyService derivation.

        Strategy:
        1. Enrich candidates with graph metrics
        2. Identify likely tech services using patterns
        3. Prioritize by PageRank
        4. Fill remaining slots from non-matches

        Args:
            candidates: Raw candidates from graph query
            enrichments: Graph enrichment data
            max_candidates: Maximum to return
            include_patterns: Patterns indicating likely tech services
            exclude_patterns: Patterns to exclude
            **kwargs: Additional unused kwargs

        Returns:
            Filtered list of candidates
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        # Enrich all candidates with graph metrics
        for c in candidates:
            enrich_candidate(c, enrichments)

        # Filter to candidates with names
        filtered = [c for c in candidates if c.name]

        # Split into likely tech services and others
        likely_tech = [
            c
            for c in filtered
            if self._is_likely_tech_service(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_tech_service(
                c.name, include_patterns, exclude_patterns
            )
        ]

        # Prioritize likely matches by PageRank
        likely_tech = filter_by_pagerank(
            likely_tech, top_n=max_candidates // 2, min_pagerank=0.001
        )

        # Fill remaining slots from others
        remaining_slots = max_candidates - len(likely_tech)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(
                others, top_n=remaining_slots, min_pagerank=0.001
            )
            likely_tech.extend(others)

        self.logger.debug(
            "TechnologyService filter: %d total -> %d final",
            len(candidates),
            len(likely_tech),
        )

        return likely_tech[:max_candidates]

    def _is_likely_tech_service(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """
        Check if a dependency suggests a technology service.

        Args:
            name: Dependency name
            include_patterns: Patterns that indicate a tech service
            exclude_patterns: Patterns to exclude

        Returns:
            True if name matches include patterns and not exclude patterns
        """
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusions first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check inclusions
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

# Create singleton instance for module-level function calls
_instance = TechnologyServiceDerivation()

# Export module-level constants (for services/derivation.py compatibility)
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
    """
    Backward-compatible filter_candidates function.

    Delegates to TechnologyServiceDerivation.filter_candidates().
    """
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
    """
    Backward-compatible generate function.

    Delegates to TechnologyServiceDerivation.generate().
    """
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
    # Backward-compatible exports
    "ELEMENT_TYPE",
    "OUTBOUND_RULES",
    "INBOUND_RULES",
    "filter_candidates",
    "generate",
    # New class export
    "TechnologyServiceDerivation",
]
