"""
DataObject Derivation.

A DataObject represents data structured for automated processing.
This includes database tables, files, messages, and other data structures.

Graph signals:
- File nodes (especially data files, configs, schemas)
- TypeDefinition nodes representing data structures
- Nodes related to persistence, storage, or data transfer

Filtering strategy:
1. Query File nodes with data-related types
2. Include schema/config/data files
3. Exclude source code and templates
4. Focus on structured data artifacts

LLM role:
- Identify which files/types represent data objects
- Generate meaningful data object names
- Write documentation describing the data purpose

Relationships:
- OUTBOUND: DataObject -> TechnologyService (Realization) - config realizes tech
- INBOUND: TechnologyService -> DataObject (Access) - tech accesses data
- INBOUND: ApplicationService -> DataObject (Flow) - app services flow data
- INBOUND: BusinessProcess -> DataObject (Access) - processes access data

ArchiMate Layer: Application Layer
ArchiMate Type: DataObject

Typical Sources:
    - File nodes with .json, .yaml, .xml, .csv extensions
    - Schema definition files and configuration files
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


class DataObjectDerivation(PatternBasedDerivation):
    """
    DataObject element derivation.

    Uses pattern-based filtering to identify data files and
    structured data artifacts from File nodes.
    """

    ELEMENT_TYPE = "DataObject"

    OUTBOUND_RULES = [
        RelationshipRule(
            target_type="TechnologyService",
            rel_type="Realization",
            description="Data objects (config/requirements) realize technology services",
        ),
    ]

    INBOUND_RULES = [
        RelationshipRule(
            target_type="TechnologyService",
            rel_type="Access",
            description="Technology services access data objects",
        ),
        RelationshipRule(
            target_type="ApplicationService",
            rel_type="Flow",
            description="Application services transfer data to/from data objects",
        ),
        RelationshipRule(
            target_type="BusinessProcess",
            rel_type="Access",
            description="Business processes access data objects",
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
        Filter candidates for DataObject derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by data file patterns
        3. Exclude source code and templates
        4. Use PageRank to find most important data files
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_data = [
            c
            for c in filtered
            if self._is_likely_data_object(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_data_object(
                c.name, include_patterns, exclude_patterns
            )
        ]

        likely_data = filter_by_pagerank(
            likely_data, top_n=max_candidates // 2, min_pagerank=0.001
        )

        remaining_slots = max_candidates - len(likely_data)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(
                others, top_n=remaining_slots, min_pagerank=0.001
            )
            likely_data.extend(others)

        self.logger.debug(
            "DataObject filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_data),
        )

        return likely_data[:max_candidates]

    def _is_likely_data_object(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a file name suggests a data object."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for data file patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = DataObjectDerivation()

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
    "DataObjectDerivation",
]
