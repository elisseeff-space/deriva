"""
SystemSoftware Derivation.

A SystemSoftware represents software that provides or contributes to an
environment for storing, executing, and using software or data deployed
within it.

Graph signals:
- Operating system references
- Runtime/platform dependencies
- Container base images
- Middleware and platform services

Filtering strategy:
1. Query ExternalDependency and File nodes
2. Filter for system/platform patterns
3. Exclude application-level libraries
4. Focus on infrastructure software

LLM role:
- Identify which dependencies are system software
- Generate meaningful software names
- Write documentation describing the software's role

ArchiMate Layer: Technology Layer
ArchiMate Type: SystemSoftware

Typical Sources:
    - Runtime/platform dependencies (Python, Node.js, JVM)
    - Container base images (alpine, debian, ubuntu)
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


class SystemSoftwareDerivation(PatternBasedDerivation):
    """
    SystemSoftware element derivation.

    Uses pattern-based filtering to identify system software
    from ExternalDependency and platform-related nodes.
    """

    ELEMENT_TYPE = "SystemSoftware"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="TechnologyService",
            rel_type="Realization",
            description="System software realizes technology services",
        ),
        RelationshipRule(
            target_type="DataObject",
            rel_type="Access",
            description="System software accesses data objects",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="Node",
            rel_type="Composition",
            description="Nodes contain system software",
        ),
        RelationshipRule(
            target_type="Device",
            rel_type="Composition",
            description="Devices contain system software",
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
        Filter candidates for SystemSoftware derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by system/platform patterns
        3. Exclude application-level libraries
        4. Use PageRank to find most important system software
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_system = [
            c
            for c in filtered
            if self._is_likely_system_software(
                c.name, include_patterns, exclude_patterns
            )
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_system_software(
                c.name, include_patterns, exclude_patterns
            )
        ]

        likely_system = filter_by_pagerank(likely_system, top_n=max_candidates // 2)

        remaining_slots = max_candidates - len(likely_system)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_system.extend(others)

        self.logger.debug(
            "SystemSoftware filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_system),
        )

        return likely_system[:max_candidates]

    def _is_likely_system_software(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a name suggests system software."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for system software patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = SystemSoftwareDerivation()

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
    "SystemSoftwareDerivation",
]
