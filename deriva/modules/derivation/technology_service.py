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
            if self.matches_patterns(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self.matches_patterns(c.name, include_patterns, exclude_patterns)
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
