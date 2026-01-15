"""
Device Derivation.

A Device represents a physical IT resource upon which system software and
artifacts may be stored or deployed for execution.

Graph signals:
- Hardware configuration references
- Docker/container host definitions
- Physical server references in deployment configs
- Infrastructure-as-code resources (Terraform, CloudFormation)

Filtering strategy:
1. Query File nodes with infrastructure patterns
2. Filter for device/hardware patterns
3. Exclude software-only definitions
4. Focus on physical/virtual machine definitions

LLM role:
- Identify which configs represent devices
- Generate meaningful device names
- Write documentation describing the device purpose

ArchiMate Layer: Technology Layer
ArchiMate Type: Device

Typical Sources:
    - Infrastructure-as-code (Terraform, CloudFormation)
    - Docker/Kubernetes deployment configurations
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


class DeviceDerivation(PatternBasedDerivation):
    """
    Device element derivation.

    Uses pattern-based filtering to identify devices
    from infrastructure configuration files.
    """

    ELEMENT_TYPE = "Device"

    OUTBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="SystemSoftware",
            rel_type="Composition",
            description="Devices contain system software",
        ),
    ]

    INBOUND_RULES: list[RelationshipRule] = [
        RelationshipRule(
            target_type="Node",
            rel_type="Composition",
            description="Nodes contain devices",
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
        Filter candidates for Device derivation.

        Strategy:
        1. Enrich with graph metrics
        2. Filter by device/hardware patterns
        3. Exclude software-only definitions
        4. Use PageRank to find most important devices
        """
        include_patterns = include_patterns or set()
        exclude_patterns = exclude_patterns or set()

        for c in candidates:
            enrich_candidate(c, enrichments)

        filtered = [c for c in candidates if c.name]

        likely_devices = [
            c
            for c in filtered
            if self._is_likely_device(c.name, include_patterns, exclude_patterns)
        ]
        others = [
            c
            for c in filtered
            if not self._is_likely_device(c.name, include_patterns, exclude_patterns)
        ]

        likely_devices = filter_by_pagerank(likely_devices, top_n=max_candidates // 2)

        remaining_slots = max_candidates - len(likely_devices)
        if remaining_slots > 0 and others:
            others = filter_by_pagerank(others, top_n=remaining_slots)
            likely_devices.extend(others)

        self.logger.debug(
            "Device filter: %d total -> %d after null -> %d final candidates",
            len(candidates),
            len(filtered),
            len(likely_devices),
        )

        return likely_devices[:max_candidates]

    def _is_likely_device(
        self, name: str, include_patterns: set[str], exclude_patterns: set[str]
    ) -> bool:
        """Check if a name suggests a device."""
        if not name:
            return False

        name_lower = name.lower()

        # Check exclusion patterns first
        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Check for device patterns
        for pattern in include_patterns:
            if pattern in name_lower:
                return True

        return False
