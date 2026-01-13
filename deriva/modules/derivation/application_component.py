"""
ApplicationComponent Derivation.

An ApplicationComponent represents a modular, deployable part of a system
that encapsulates its behavior and data.

Graph signals:
- Directory nodes (structural organization)
- Louvain community roots (cohesive modules)
- High PageRank (important/central directories)
- Path patterns: src/, app/, lib/, components/, modules/

Filtering strategy:
1. Query Directory nodes (excluding test/config/docs)
2. Get enrichment data (pagerank, louvain, kcore)
3. Filter to community roots or high-pagerank nodes
4. Limit to top N by PageRank
5. Send to LLM for final decision and naming

ArchiMate Layer: Application Layer
ArchiMate Type: ApplicationComponent

Typical Sources:
    - Directory nodes with src/, app/, lib/ paths
    - Louvain community root nodes (cohesive module boundaries)
"""

from __future__ import annotations

import logging
from typing import Any

from deriva.modules.derivation.base import (
    Candidate,
    RelationshipRule,
    filter_by_pagerank,
    get_community_roots,
)
from deriva.modules.derivation.element_base import ElementDerivationBase

logger = logging.getLogger(__name__)


class ApplicationComponentDerivation(ElementDerivationBase):
    """
    ApplicationComponent element derivation.

    Uses graph-based filtering (community roots + PageRank) rather than
    pattern-based filtering. This identifies natural component boundaries
    from the graph structure itself.
    """

    ELEMENT_TYPE = "ApplicationComponent"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="ApplicationService",
            rel_type="Composition",
            description="Application components contain application services",
        ),
        RelationshipRule(
            target_type="DataObject",
            rel_type="Access",
            description="Application components access data objects",
        ),
        RelationshipRule(
            target_type="ApplicationInterface",
            rel_type="Composition",
            description="Application components expose interfaces",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="TechnologyService",
            rel_type="Realization",
            description="Technology services realize application components",
        ),
        RelationshipRule(
            target_type="Node",
            rel_type="Serving",
            description="Nodes serve application components",
        ),
    ]

    # No get_filter_kwargs override needed - uses default empty dict
    # ApplicationComponent uses graph structure, not config patterns

    def filter_candidates(
        self,
        candidates: list[Candidate],
        enrichments: dict[str, dict[str, Any]],
        max_candidates: int,
        **kwargs: Any,
    ) -> list[Candidate]:
        """
        Apply graph-based filtering to reduce candidates for LLM.

        Strategy:
        1. Prioritize community roots (natural component boundaries)
        2. Include high-pagerank non-roots (important directories)
        3. Sort by pagerank and limit

        Args:
            candidates: List of candidates to filter
            enrichments: Graph enrichment data (pagerank, community, etc.)
            max_candidates: Maximum number of candidates to return
            **kwargs: Unused additional arguments

        Returns:
            Filtered list of candidates
        """
        if not candidates:
            return []

        # Get community roots - these are natural component boundaries
        roots = get_community_roots(candidates)

        # Also include high-pagerank nodes that aren't roots
        # (they might be important subdirectories)
        non_roots = [c for c in candidates if c not in roots]
        high_pagerank = filter_by_pagerank(non_roots, top_n=10)

        # Combine and deduplicate
        combined = list(roots)
        for c in high_pagerank:
            if c not in combined:
                combined.append(c)

        # Sort by pagerank (most important first) and limit
        combined = filter_by_pagerank(combined, top_n=max_candidates)

        self.logger.debug(
            "ApplicationComponent filter: %d total -> %d roots -> %d combined",
            len(candidates),
            len(roots),
            len(combined),
        )

        return combined


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

# Create singleton instance for module-level function calls
_instance = ApplicationComponentDerivation()

# Export module-level constants (for services/derivation.py compatibility)
ELEMENT_TYPE = _instance.ELEMENT_TYPE
OUTBOUND_RULES = _instance.OUTBOUND_RULES
INBOUND_RULES = _instance.INBOUND_RULES


def filter_candidates(
    candidates: list[Candidate],
    enrichments: dict[str, dict[str, Any]],
    max_candidates: int,
) -> list[Candidate]:
    """
    Backward-compatible filter_candidates function.

    Delegates to ApplicationComponentDerivation.filter_candidates().
    """
    return _instance.filter_candidates(candidates, enrichments, max_candidates)


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

    Delegates to ApplicationComponentDerivation.generate().
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
    "ApplicationComponentDerivation",
]
