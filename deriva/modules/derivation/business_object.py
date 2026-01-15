"""
BusinessObject Derivation.

A BusinessObject represents a passive element that has relevance from a
business perspective. It represents things like data entities, domain
concepts, or business documents.

Graph signals:
- TypeDefinition nodes (classes/data models)
- BusinessConcept nodes (from LLM extraction)
- File nodes with model patterns (models.py, entities.py, schema.py)
- High in-degree (many references = important domain concept)

Filtering strategy:
1. Query TypeDefinition and BusinessConcept nodes
2. Exclude utility classes (helpers, mixins, base classes)
3. Prioritize by PageRank (central domain concepts)
4. Focus on nouns that represent business data

LLM role:
- Identify which type definitions are business-relevant
- Generate meaningful business names (not code names)
- Write documentation describing the business meaning

Relationships:
- OUTBOUND: BusinessObject -> BusinessObject (Composition/Aggregation)
- INBOUND: BusinessProcess -> BusinessObject (Access)
- INBOUND: ApplicationService -> BusinessObject (Flow)

ArchiMate Layer: Business Layer
ArchiMate Type: BusinessObject

Typical Sources:
    - TypeDefinition nodes (classes, dataclasses, models)
    - BusinessConcept nodes from LLM extraction
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


class BusinessObjectDerivation(PatternBasedDerivation):
    """
    BusinessObject element derivation.

    Uses pattern-based filtering to identify business-relevant types
    and concepts from TypeDefinition and BusinessConcept nodes.
    """

    ELEMENT_TYPE = "BusinessObject"

    OUTBOUND_RULES = [
        RelationshipRule(
            target_type="BusinessObject",
            rel_type="Composition",
            description="Business objects contain other business objects",
        ),
        RelationshipRule(
            target_type="BusinessObject",
            rel_type="Aggregation",
            description="Business objects reference other business objects",
        ),
    ]

    INBOUND_RULES = [
        RelationshipRule(
            target_type="BusinessProcess",
            rel_type="Access",
            description="Business processes access business objects",
        ),
        RelationshipRule(
            target_type="ApplicationService",
            rel_type="Flow",
            description="Application services flow data to business objects",
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
        Filter candidates for BusinessObject derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Pre-filter by name patterns (exclude utilities)
        3. Prioritize likely business objects
        4. Use PageRank/in-degree to find most important types
        5. Limit to max_candidates for LLM
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_business = [
            c
            for c in filtered
            if self._is_likely_business_object(
                c.name, include_patterns, exclude_patterns
            )
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_business_object(
                c.name, include_patterns, exclude_patterns
            )
        ]

        likely_business = filter_by_pagerank(
            likely_business, top_n=max_candidates // 2, min_pagerank=0.001
        )

        remaining_slots = max_candidates - len(likely_business)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(
                others, top_n=remaining_slots, min_pagerank=0.001
            )
            likely_business.extend(others)

        self.logger.debug(
            "BusinessObject filter: %d total -> %d after null check -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_business),
        )

        return likely_business[:max_candidates]

    def _is_likely_business_object(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a type name suggests a business object."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for business patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        # Default: include if it looks like a noun (starts with capital, no underscores)
        return name[0].isupper() and "_" not in name
