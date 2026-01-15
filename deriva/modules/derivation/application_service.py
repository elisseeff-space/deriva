"""
ApplicationService Derivation.

An ApplicationService represents an explicitly defined exposed application
behavior. This includes API endpoints, web routes, and service interfaces.

Graph signals:
- Method nodes with route/endpoint patterns
- Functions decorated with @app.route, @api.get, etc.
- Controller/handler methods
- Methods in files named routes.py, api.py, views.py

Filtering strategy:
1. Query Method nodes
2. Filter for route/endpoint patterns
3. Look for HTTP method indicators (get, post, put, delete)
4. Focus on externally exposed interfaces

LLM role:
- Identify which methods are application services
- Generate meaningful service names
- Write documentation describing the service's purpose

Relationships:
- OUTBOUND: ApplicationService -> BusinessObject (Flow) - services transfer data
- INBOUND: TechnologyService -> ApplicationService (Serving) - tech serves app services

ArchiMate Layer: Application Layer
ArchiMate Type: ApplicationService

Typical Sources:
    - Method nodes with @route, @api decorators
    - Methods in routes.py, api.py, views.py, controllers.py
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


class ApplicationServiceDerivation(PatternBasedDerivation):
    """
    ApplicationService element derivation.

    Uses pattern-based filtering to identify API endpoints and
    service methods from Method nodes.
    """

    ELEMENT_TYPE = "ApplicationService"

    OUTBOUND_RULES = [
        RelationshipRule(
            target_type="BusinessObject",
            rel_type="Flow",
            description="Application services transfer/process business data",
        ),
    ]

    INBOUND_RULES = [
        RelationshipRule(
            target_type="TechnologyService",
            rel_type="Serving",
            description="Technology services serve application services",
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
        """Filter candidates for ApplicationService derivation."""
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        # Pre-filter: exclude dunder methods
        filtered = [c for c in candidates if c.name and not c.name.startswith("__")]

        # Separate likely services from others
        likely_services = [
            c
            for c in filtered
            if self.matches_patterns(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self.matches_patterns(c.name, include_patterns, exclude_patterns)
        ]

        # Sort by PageRank
        likely_services = filter_by_pagerank(likely_services, top_n=max_candidates // 2)

        remaining_slots = max_candidates - len(likely_services)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_services.extend(others)

        self.logger.debug(
            "ApplicationService filter: %d total -> %d final",
            len(candidates),
            len(likely_services),
        )

        return likely_services[:max_candidates]
