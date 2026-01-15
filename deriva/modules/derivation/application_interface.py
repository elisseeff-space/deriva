"""
ApplicationInterface Derivation.

An ApplicationInterface represents a point of access where application services
are made available to a user, another application component, or a node.

Graph signals:
- API endpoint definitions (REST, GraphQL, gRPC)
- Public methods/functions exposed to external callers
- Route handlers in web frameworks
- Interface/protocol definitions

Filtering strategy:
1. Query Method nodes that represent endpoints/handlers
2. Filter for API/interface patterns
3. Exclude internal/private methods
4. Focus on externally accessible entry points

LLM role:
- Identify which methods represent interfaces
- Generate meaningful interface names
- Write documentation describing the interface purpose

ArchiMate Layer: Application Layer
ArchiMate Type: ApplicationInterface

Typical Sources:
    - API endpoint definitions (REST, GraphQL, gRPC)
    - Public interface/protocol method definitions
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


class ApplicationInterfaceDerivation(PatternBasedDerivation):
    """
    ApplicationInterface element derivation.

    Uses pattern-based filtering to identify application interfaces
    from API endpoints and public method definitions.
    """

    ELEMENT_TYPE = "ApplicationInterface"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="ApplicationService",
            rel_type="Serving",
            description="Application interfaces expose application services",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="ApplicationComponent",
            rel_type="Composition",
            description="Application components contain interfaces",
        ),
        RelationshipRule(
            target_type="BusinessActor",
            rel_type="Serving",
            description="Application interfaces serve business actors",
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
        Filter candidates for ApplicationInterface derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by interface patterns (API, endpoint, handler)
        3. Exclude internal/private methods
        4. Use PageRank to find most important interfaces
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        # Pre-filter: exclude private methods (starting with _)
        filtered = [c for c in candidates if c.name and not c.name.startswith("_")]

        likely_interfaces = [
            c
            for c in filtered
            if self.matches_patterns(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self.matches_patterns(c.name, include_patterns, exclude_patterns)
        ]

        likely_interfaces = filter_by_pagerank(
            likely_interfaces, top_n=max_candidates // 2
        )

        remaining_slots = max_candidates - len(likely_interfaces)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_interfaces.extend(others)

        self.logger.debug(
            "ApplicationInterface filter: %d total -> %d after exclude -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_interfaces),
        )

        return likely_interfaces[:max_candidates]
