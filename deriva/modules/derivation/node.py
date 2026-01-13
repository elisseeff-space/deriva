"""
Node Derivation.

A Node represents a computational or physical resource that hosts, manipulates,
or interacts with other computational or physical resources.

Graph signals:
- Server/container definitions
- Kubernetes deployments and services
- Cloud resource definitions (EC2, VMs)
- Docker Compose service definitions

Filtering strategy:
1. Query File nodes with deployment/infrastructure patterns
2. Filter for node/server patterns
3. Exclude application-level definitions
4. Focus on infrastructure nodes

LLM role:
- Identify which configs represent nodes
- Generate meaningful node names
- Write documentation describing the node's role

ArchiMate Layer: Technology Layer
ArchiMate Type: Node

Typical Sources:
    - Kubernetes deployment/service definitions
    - Docker Compose service configurations
    - Infrastructure-as-code server definitions
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


class NodeDerivation(PatternBasedDerivation):
    """
    Node element derivation.

    Uses pattern-based filtering to identify infrastructure nodes
    from deployment and infrastructure configuration files.
    """

    ELEMENT_TYPE = "Node"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="SystemSoftware",
            rel_type="Composition",
            description="Nodes contain system software",
        ),
        RelationshipRule(
            target_type="Device",
            rel_type="Composition",
            description="Nodes contain devices",
        ),
        RelationshipRule(
            target_type="ApplicationComponent",
            rel_type="Serving",
            description="Nodes serve application components",
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
        Filter candidates for Node derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by node/infrastructure patterns
        3. Exclude application-level definitions
        4. Use PageRank to find most important nodes
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_nodes = [
            c
            for c in filtered
            if self._is_likely_node(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_node(c.name, include_patterns, exclude_patterns)
        ]

        likely_nodes = filter_by_pagerank(likely_nodes, top_n=max_candidates // 2)

        remaining_slots = max_candidates - len(likely_nodes)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_nodes.extend(others)

        self.logger.debug(
            "Node filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_nodes),
        )

        return likely_nodes[:max_candidates]

    def _is_likely_node(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a name suggests an infrastructure node."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for node patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False


# =============================================================================
# Backward Compatibility - Module-level exports
# =============================================================================

_instance = NodeDerivation()

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
    "NodeDerivation",
]
