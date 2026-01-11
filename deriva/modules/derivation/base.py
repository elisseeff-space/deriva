"""
Base utilities for hybrid derivation.

Provides shared functionality for per-element derivation files:
- Graph filtering and enrichment access
- LLM schemas for structured output
- Prompt building for elements and relationships
- Response parsing
- Element and relationship creation helpers
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from deriva.adapters.llm import FailedResponse, ResponseType
from deriva.common import current_timestamp, parse_json_array
from deriva.common.types import PipelineResult

if TYPE_CHECKING:
    from deriva.adapters.graph import GraphManager


def extract_response_content(response: Any) -> tuple[str, str | None]:
    """
    Extract content from an LLM response, handling all response types.

    Args:
        response: LLM response object (LiveResponse, CachedResponse, FailedResponse, or other)

    Returns:
        Tuple of (content, error) where error is None if successful
    """
    # Check for FailedResponse first
    if isinstance(response, FailedResponse):
        return "", f"LLM call failed: {response.error}"

    # Check for response_type attribute (dataclass-based responses)
    if hasattr(response, "response_type"):
        if response.response_type == ResponseType.FAILED:
            error = getattr(response, "error", "Unknown error")
            return "", f"LLM call failed: {error}"

    # Extract content from response
    if hasattr(response, "content"):
        return response.content, None

    # Fallback to string representation (shouldn't happen with proper response types)
    return str(response), None

logger = logging.getLogger(__name__)

# Essential properties to include in LLM prompts (reduces token usage)
# These are the most useful properties for element derivation
ESSENTIAL_PROPS: set[str] = {
    "name",
    "description",
    "typeName",
    "filePath",
    "conceptType",
    "conceptName",
    "techName",
    "dependencyName",
    "methodName",
    "className",
    "docstring",
}

# Properties to exclude from relationship prompts (invalidate cache)
EXCLUDED_FROM_CACHE: set[str] = {"derived_at"}

# Essential fields for relationship derivation (reduces tokens by ~50%)
RELATIONSHIP_ESSENTIAL_FIELDS: set[str] = {"identifier", "name", "element_type"}


def strip_for_relationship_prompt(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Strip elements to essential fields for relationship derivation.

    For relationship inference, we only need identifier, name, and element_type.
    Documentation and other properties are not needed and add tokens.

    Args:
        elements: List of element dictionaries

    Returns:
        List with only essential fields per element
    """
    return [
        {k: v for k, v in elem.items() if k in RELATIONSHIP_ESSENTIAL_FIELDS}
        for elem in elements
    ]


def strip_cache_breaking_props(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Strip properties that would invalidate cache from elements.

    The derived_at timestamp changes every run, causing cache misses
    even when the actual content is identical.

    Args:
        elements: List of element dictionaries

    Returns:
        Copy of elements with cache-breaking properties removed
    """
    result = []
    for elem in elements:
        clean = dict(elem)
        if "properties" in clean and isinstance(clean["properties"], dict):
            clean["properties"] = {
                k: v
                for k, v in clean["properties"].items()
                if k not in EXCLUDED_FROM_CACHE
            }
        result.append(clean)
    return result


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Candidate:
    """A candidate node for element derivation with enrichment data."""

    node_id: str
    name: str
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    # Graph enrichment data (populated from DuckDB)
    pagerank: float = 0.0
    louvain_community: str | None = None
    kcore_level: int = 0
    is_articulation_point: bool = False
    in_degree: int = 0
    out_degree: int = 0

    def to_dict(self, include_props: set[str] | None = None) -> dict[str, Any]:
        """Convert to dict for JSON serialization in LLM prompts.

        Args:
            include_props: Optional set of property keys to include.
                          If None, includes all properties.
                          Use ESSENTIAL_PROPS for minimal token usage.
        """
        # Filter properties if whitelist provided
        props = self.properties
        if include_props is not None:
            props = {k: v for k, v in self.properties.items() if k in include_props}

        return {
            "id": self.node_id,
            "name": self.name,
            "labels": self.labels,
            "properties": props,
            "pagerank": round(self.pagerank, 4),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
        }


@dataclass
class RelationshipRule:
    """A rule defining valid relationships for an element type."""

    target_type: (
        str  # For outbound: target element type. For inbound: source element type
    )
    rel_type: str  # ArchiMate relationship type (Serving, Access, etc.)
    description: str = ""  # Human-readable description


@dataclass
class GenerationResult:
    """Result from element generation (includes relationships)."""

    success: bool
    elements_created: int = 0
    relationships_created: int = 0
    created_elements: list[dict[str, Any]] = field(default_factory=list)
    created_relationships: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DerivationResult:
    """Result from element + relationship derivation (mirrors extraction pattern)."""

    success: bool
    elements: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Graph Enrichment Access
# =============================================================================


def get_enrichments_from_neo4j(
    graph_manager: "GraphManager",
) -> dict[str, dict[str, Any]]:
    """
    Get all graph enrichment data from Neo4j node properties.

    The prep phase stores enrichments (PageRank, Louvain, k-core, etc.)
    as properties on Neo4j nodes. This function reads them back.

    Args:
        graph_manager: Connected GraphManager instance

    Returns:
        Dict mapping node_id to enrichment data
    """
    query = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label STARTS WITH 'Graph:')
          AND n.active = true
        RETURN n.id as node_id,
               n.pagerank as pagerank,
               n.louvain_community as louvain_community,
               n.kcore_level as kcore_level,
               n.is_articulation_point as is_articulation_point,
               n.in_degree as in_degree,
               n.out_degree as out_degree
    """
    try:
        rows = graph_manager.query(query)
        return {
            row["node_id"]: {
                "pagerank": row.get("pagerank") or 0.0,
                "louvain_community": row.get("louvain_community"),
                "kcore_level": row.get("kcore_level") or 0,
                "is_articulation_point": row.get("is_articulation_point") or False,
                "in_degree": row.get("in_degree") or 0,
                "out_degree": row.get("out_degree") or 0,
            }
            for row in rows
            if row.get("node_id")
        }
    except Exception as e:
        logger.warning(f"Failed to get enrichments from Neo4j: {e}")
        return {}


# Backward compatibility alias (deprecated)
def get_enrichments(engine: Any) -> dict[str, dict[str, Any]]:
    """Deprecated: Use get_enrichments_from_neo4j() instead."""
    logger.warning(
        "get_enrichments(engine) is deprecated - enrichments should be read from Neo4j"
    )
    return {}


def enrich_candidate(
    candidate: Candidate, enrichments: dict[str, dict[str, Any]]
) -> None:
    """Add enrichment data to a candidate in-place."""
    data = enrichments.get(candidate.node_id, {})
    candidate.pagerank = data.get("pagerank", 0.0)
    candidate.louvain_community = data.get("louvain_community")
    candidate.kcore_level = data.get("kcore_level", 0)
    candidate.is_articulation_point = data.get("is_articulation_point", False)
    candidate.in_degree = data.get("in_degree", 0)
    candidate.out_degree = data.get("out_degree", 0)


# =============================================================================
# Graph Filtering
# =============================================================================


def filter_by_pagerank(
    candidates: list[Candidate],
    top_n: int | None = None,
    percentile: float | None = None,
    min_pagerank: float | None = None,
) -> list[Candidate]:
    """
    Filter candidates by PageRank score.

    Args:
        candidates: List of candidates with pagerank populated
        top_n: Keep top N candidates
        percentile: Keep top X percentile (0-100)
        min_pagerank: Minimum absolute PageRank score to include (applied first)

    Returns:
        Filtered and sorted candidates (highest pagerank first)
    """
    # Apply minimum threshold first (removes low-importance nodes)
    if min_pagerank is not None:
        candidates = [c for c in candidates if c.pagerank >= min_pagerank]

    sorted_candidates = sorted(candidates, key=lambda c: -c.pagerank)

    if top_n is not None:
        return sorted_candidates[:top_n]

    if percentile is not None:
        cutoff_idx = max(1, int(len(sorted_candidates) * (100 - percentile) / 100))
        return sorted_candidates[:cutoff_idx]

    return sorted_candidates


def filter_by_labels(
    candidates: list[Candidate],
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
) -> list[Candidate]:
    """
    Filter candidates by node labels.

    Args:
        candidates: List of candidates
        include_labels: Only keep candidates with ANY of these labels
        exclude_labels: Remove candidates with ANY of these labels
    """
    result = candidates

    if include_labels:
        result = [c for c in result if any(lbl in c.labels for lbl in include_labels)]

    if exclude_labels:
        result = [
            c for c in result if not any(lbl in c.labels for lbl in exclude_labels)
        ]

    return result


def filter_by_community(
    candidates: list[Candidate],
    community_ids: set[str] | None = None,
    only_roots: bool = False,
) -> list[Candidate]:
    """
    Filter candidates by Louvain community.

    Args:
        candidates: List of candidates
        community_ids: Only keep candidates in these communities
        only_roots: Only keep community root nodes (node_id == louvain_community)
    """
    result = candidates

    if community_ids is not None:
        result = [c for c in result if c.louvain_community in community_ids]

    if only_roots:
        result = [c for c in result if c.node_id == c.louvain_community]

    return result


def get_community_roots(candidates: list[Candidate]) -> list[Candidate]:
    """Get candidates that are Louvain community roots."""
    return [c for c in candidates if c.node_id == c.louvain_community]


def get_articulation_points(candidates: list[Candidate]) -> list[Candidate]:
    """Get candidates that are articulation points (bridge nodes)."""
    return [c for c in candidates if c.is_articulation_point]


# =============================================================================
# Token Estimation & Context Limiting (Phase 4)
# =============================================================================

# Default model context limits (tokens) - conservative estimates
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1-mini": 128000,
    "gpt-4.1-nano": 128000,
    "claude-3": 200000,
    "claude-haiku": 200000,
    "claude-sonnet": 200000,
    "claude-opus": 200000,
    "devstral": 32000,
    "mistral": 32000,
    "default": 16000,  # Conservative default
}


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is accurate within ~10% for most LLM tokenizers.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def get_model_context_limit(model_name: str) -> int:
    """
    Get the context limit for a model.

    Args:
        model_name: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet")

    Returns:
        Context limit in tokens
    """
    model_lower = model_name.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_lower:
            return limit
    return MODEL_CONTEXT_LIMITS["default"]


def limit_existing_elements(
    elements: list[dict[str, Any]],
    max_elements: int = 50,
    sort_by_confidence: bool = True,
) -> list[dict[str, Any]]:
    """
    Limit existing elements to top-N by importance.

    When deriving relationships, we don't need ALL existing elements -
    just the most important ones. This reduces tokens by up to 80%.

    Args:
        elements: List of element dictionaries
        max_elements: Maximum number of elements to keep (default 50)
        sort_by_confidence: If True, sort by confidence descending (default True)

    Returns:
        Filtered list of elements
    """
    if len(elements) <= max_elements:
        return elements

    if sort_by_confidence:
        # Sort by confidence (from properties), highest first
        sorted_elements = sorted(
            elements,
            key=lambda e: e.get("properties", {}).get("confidence", 0.5),
            reverse=True,
        )
        return sorted_elements[:max_elements]

    # Simple truncation if not sorting
    return elements[:max_elements]


def stratified_sample_elements(
    elements: list[dict[str, Any]],
    max_per_type: int = 10,
    relevant_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Sample elements with stratification by element type.

    Ensures representation from each relevant element type while
    limiting total count. Improves relationship diversity.

    Args:
        elements: List of element dictionaries
        max_per_type: Maximum elements per type (default 10)
        relevant_types: Only include these types (None = all types)

    Returns:
        Stratified sample of elements
    """
    if not elements:
        return []

    # Group by element_type
    by_type: dict[str, list[dict[str, Any]]] = {}
    for elem in elements:
        etype = elem.get("element_type", "Unknown")
        if relevant_types is None or etype in relevant_types:
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(elem)

    # Take top max_per_type from each type (sorted by confidence)
    sampled = []
    for etype, type_elements in by_type.items():
        sorted_type = sorted(
            type_elements,
            key=lambda e: e.get("properties", {}).get("confidence", 0.5),
            reverse=True,
        )
        sampled.extend(sorted_type[:max_per_type])

    return sampled


def get_connected_source_ids(
    graph_manager: "GraphManager",
    source_ids: list[str],
    max_hops: int = 2,
) -> set[str]:
    """
    Get graph node IDs connected to the given source nodes.

    Queries the graph for nodes within max_hops of the source nodes.
    This enables graph-aware filtering of existing elements.

    Args:
        graph_manager: GraphManager instance for querying
        source_ids: List of source node IDs to find connections for
        max_hops: Maximum path length to consider (default 2)

    Returns:
        Set of connected node IDs (includes source_ids)
    """
    if not source_ids or not graph_manager:
        return set(source_ids) if source_ids else set()

    # Build Cypher query for neighbors within max_hops
    # Using variable-length path pattern for efficiency
    query = """
        MATCH (n)
        WHERE n.id IN $source_ids
        MATCH (n)-[*1..%d]-(neighbor)
        WHERE neighbor.active = true OR neighbor.active IS NULL
        RETURN DISTINCT neighbor.id as id
    """ % max_hops

    try:
        results = graph_manager.query(query, {"source_ids": source_ids})
        connected = {row["id"] for row in results if row.get("id")}
        # Include original source IDs
        connected.update(source_ids)
        return connected
    except Exception as e:
        logger.warning("Graph query for connected nodes failed: %s", e)
        # Fall back to just the source IDs
        return set(source_ids)


def filter_by_graph_proximity(
    elements: list[dict[str, Any]],
    connected_ids: set[str],
) -> list[dict[str, Any]]:
    """
    Filter elements to only those with source nodes in the connected set.

    This implements graph-aware pre-filtering: only include elements
    that are graph neighbors of the new elements being processed.

    Args:
        elements: List of element dictionaries with properties.source
        connected_ids: Set of connected graph node IDs

    Returns:
        Filtered list of elements with graph proximity
    """
    if not connected_ids:
        return elements

    filtered = []
    for elem in elements:
        source = elem.get("properties", {}).get("source")
        if source and source in connected_ids:
            filtered.append(elem)

    return filtered


def check_prompt_size(
    prompt: str,
    model_name: str = "default",
    warn_threshold: float = 0.8,
) -> tuple[int, bool]:
    """
    Check if prompt size is within model limits.

    Args:
        prompt: The prompt string to check
        model_name: Model name for context limit lookup
        warn_threshold: Fraction of limit to warn at (default 0.8 = 80%)

    Returns:
        Tuple of (estimated_tokens, is_over_threshold)
    """
    estimated = estimate_tokens(prompt)
    limit = get_model_context_limit(model_name)
    threshold = int(limit * warn_threshold)

    if estimated > threshold:
        logger.warning(
            "Prompt size %d tokens exceeds %d%% of %s limit (%d)",
            estimated,
            int(warn_threshold * 100),
            model_name,
            limit,
        )
        return estimated, True

    return estimated, False


# =============================================================================
# Batching
# =============================================================================


def calculate_dynamic_batch_size(
    num_candidates: int,
    min_batch: int = 10,
    max_batch: int = 25,
) -> int:
    """
    Calculate optimal batch size based on candidate count.

    For small candidate sets, use larger batches to reduce LLM calls.
    For large sets, use moderate batches to balance context and calls.

    Args:
        num_candidates: Total number of candidates
        min_batch: Minimum batch size (default 10)
        max_batch: Maximum batch size (default 25)

    Returns:
        Calculated batch size
    """
    if num_candidates <= min_batch:
        return num_candidates  # Single batch for small sets
    # Use ~3-4 batches for most datasets
    dynamic = max(min_batch, num_candidates // 3)
    return min(max_batch, dynamic)


def adjust_batch_for_tokens(
    current_batch_size: int,
    estimated_tokens: int,
    model_name: str = "default",
    target_utilization: float = 0.7,
    min_batch: int = 5,
) -> int:
    """
    Adjust batch size based on estimated token count.

    If the estimated tokens exceed target utilization of model limit,
    reduce batch size proportionally to fit within limits.

    Args:
        current_batch_size: Current batch size
        estimated_tokens: Estimated tokens for current batch
        model_name: Model name for context limit lookup
        target_utilization: Target fraction of context limit (default 0.7 = 70%)
        min_batch: Minimum batch size (default 5)

    Returns:
        Adjusted batch size
    """
    limit = get_model_context_limit(model_name)
    target = int(limit * target_utilization)

    if estimated_tokens <= target:
        return current_batch_size

    # Scale down proportionally
    scale_factor = target / estimated_tokens
    adjusted = int(current_batch_size * scale_factor)

    # Clamp to minimum
    adjusted = max(min_batch, adjusted)

    if adjusted < current_batch_size:
        logger.info(
            "Reducing batch size %d -> %d due to token limit (est: %d, target: %d)",
            current_batch_size,
            adjusted,
            estimated_tokens,
            target,
        )

    return adjusted


def batch_candidates(
    candidates: list[Candidate],
    batch_size: int | None = None,
    group_by_community: bool = True,
) -> list[list[Candidate]]:
    """
    Split candidates into batches for LLM processing.

    Args:
        candidates: List of candidates to batch
        batch_size: Maximum items per batch. If None, uses dynamic sizing
                   based on candidate count (recommended).
        group_by_community: If True, group candidates by Louvain community first,
                           keeping related nodes together (default True)

    Returns:
        List of batches
    """
    if not candidates:
        return []

    # Use dynamic batch sizing if not specified
    if batch_size is None:
        batch_size = calculate_dynamic_batch_size(len(candidates))

    if not group_by_community:
        # Simple sequential batching
        batches = []
        for i in range(0, len(candidates), batch_size):
            batches.append(candidates[i : i + batch_size])
        return batches

    # Group by Louvain community first for coherent batches
    by_community: dict[str | None, list[Candidate]] = {}
    for c in candidates:
        comm = c.louvain_community
        if comm not in by_community:
            by_community[comm] = []
        by_community[comm].append(c)

    # Build batches keeping communities together when possible
    batches: list[list[Candidate]] = []
    current_batch: list[Candidate] = []

    for community_candidates in by_community.values():
        for c in community_candidates:
            current_batch.append(c)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
# Query Helpers
# =============================================================================


def query_candidates(
    graph_manager: "GraphManager",
    cypher_query: str,
    enrichments: dict[str, dict[str, Any]] | None = None,
) -> list[Candidate]:
    """
    Execute a Cypher query and return enriched candidates.

    The query should return: id, name, labels, properties
    """
    results = graph_manager.query(cypher_query)
    candidates = []

    for row in results:
        candidate = Candidate(
            node_id=row.get("id", ""),
            name=row.get("name", ""),
            labels=row.get("labels", []),
            properties=row.get("properties", {}),
        )
        if enrichments:
            enrich_candidate(candidate, enrichments)
        candidates.append(candidate)

    return candidates


# =============================================================================
# LLM Schemas
# =============================================================================

DERIVATION_SCHEMA: dict[str, Any] = {
    "name": "derivation_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string"},
                        "name": {"type": "string"},
                        "documentation": {"type": "string"},
                        "source": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "identifier",
                        "name",
                        "documentation",
                        "source",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["elements"],
        "additionalProperties": False,
    },
}

RELATIONSHIP_SCHEMA: dict[str, Any] = {
    "name": "relationship_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "relationship_type": {"type": "string"},
                        "name": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "source",
                        "target",
                        "relationship_type",
                        "name",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["relationships"],
        "additionalProperties": False,
    },
}


# =============================================================================
# Prompt Building - Elements
# =============================================================================


def build_derivation_prompt(
    candidates: list[Candidate] | list[dict[str, Any]],
    instruction: str,
    example: str,
    element_type: str,
    existing_identifiers: list[str] | None = None,
    existing_elements_summary: dict[str, list[str]] | None = None,
) -> str:
    """
    Build LLM prompt for element derivation.

    Args:
        candidates: Pre-filtered candidate nodes (Candidate objects or dicts)
        instruction: Element-specific derivation instructions
        example: Example output format
        element_type: ArchiMate element type
        existing_identifiers: Optional list of already-created identifiers to avoid
        existing_elements_summary: Optional dict mapping element_type to list of names,
                                  for naming alignment across types
    """
    # Convert Candidate objects to dicts with minimal properties (reduces tokens)
    if candidates and isinstance(candidates[0], Candidate):
        candidate_list = cast(list[Candidate], candidates)
        data = [c.to_dict(include_props=ESSENTIAL_PROPS) for c in candidate_list]
    else:
        data = candidates

    # Use compact JSON (no indentation) to reduce token usage by ~20-30%
    data_json = json.dumps(data, separators=(",", ":"), default=str)

    # Build forbidden names section if existing identifiers provided
    forbidden_section = ""
    if existing_identifiers:
        forbidden_section = f"""
## Already Created (DO NOT duplicate these identifiers)
{json.dumps(existing_identifiers, separators=(",", ":"))}
"""

    # Build cross-element reference section for naming alignment
    context_section = ""
    if existing_elements_summary:
        lines = []
        for etype, names in existing_elements_summary.items():
            if names:
                lines.append(f"- {etype}: {', '.join(names[:10])}")  # Limit to 10
        if lines:
            context_section = f"""
## Existing Elements (for naming alignment)
{chr(10).join(lines)}
"""

    return f"""You are deriving ArchiMate {element_type} elements from source code graph data.

## Instructions
{instruction}

## Candidate Nodes
These nodes have been pre-filtered as potential {element_type} candidates.
Each includes graph metrics (pagerank, degree) to help assess importance.

```json
{data_json}
```
{forbidden_section}{context_section}
## Example Output
{example}

## Rules
1. Only create elements from the provided candidates
2. Use the node "id" as the "source" field to link back
3. Provide meaningful names (not just the node name)
4. Add documentation explaining the element's purpose
5. Set confidence based on how well the candidate matches {element_type}
6. If no candidates are suitable, return {{"elements": []}}
7. Output stable, deterministic results - same inputs should produce same outputs

Return a JSON object with an "elements" array.
"""


# =============================================================================
# Prompt Building - Relationships
# =============================================================================


def build_relationship_prompt(elements: list[dict[str, Any]]) -> str:
    """Build LLM prompt for relationship derivation."""
    # Use compact JSON to reduce token usage
    elements_json = json.dumps(elements, separators=(",", ":"), default=str)
    valid_ids = [e.get("identifier", "") for e in elements if e.get("identifier")]

    return f"""Derive relationships between these ArchiMate elements:

```json
{elements_json}
```

CRITICAL RULES:
1. You must ONLY use identifiers from this exact list: {json.dumps(valid_ids)}
2. Do NOT invent new identifiers or modify existing ones
3. Use identifiers exactly as shown (case-sensitive, character-for-character)
4. Only create relationships where BOTH source AND target exist in the list

VALID RELATIONSHIP TYPES (use ONLY these exact names):
- Composition: element consists of other elements
- Aggregation: element combines other elements
- Serving: element provides services to another
- Realization: element realizes/implements another
- Access: element reads/writes data objects
- Flow: transfer of information between elements
- Assignment: allocates responsibility between elements

INVALID TYPES (NEVER use these):
- Association (use Serving or Flow instead)
- Dependency (use Serving instead)
- Uses (use Serving instead)

Output stable, deterministic results.
Return {{"relationships": []}} with source, target, relationship_type for each.
"""


def build_element_relationship_prompt(
    source_elements: list[dict[str, Any]],
    target_elements: list[dict[str, Any]],
    source_element_type: str,
    valid_relationships: list[dict[str, Any]],
    instruction: str | None = None,
    example: str | None = None,
) -> str:
    """Build LLM prompt for element-type-specific relationship derivation."""
    # Use compact JSON to reduce token usage
    sources_json = json.dumps(source_elements, separators=(",", ":"), default=str)
    targets_json = json.dumps(target_elements, separators=(",", ":"), default=str)

    source_ids = [
        e.get("identifier", "") for e in source_elements if e.get("identifier")
    ]
    target_ids = [
        e.get("identifier", "") for e in target_elements if e.get("identifier")
    ]

    rel_rules = []
    for rel in valid_relationships:
        targets = ", ".join(rel["allowed_targets"])
        rel_rules.append(
            f"- {rel['relationship_type']}: {rel['description']} → can target: [{targets}]"
        )
    rel_rules_text = (
        "\n".join(rel_rules) if rel_rules else "No valid relationship types."
    )

    default_instruction = f"""Derive relationships FROM the {source_element_type} elements.
Only create relationships where the source is from the Source Elements list.
Use only the relationship types valid for {source_element_type} as specified below."""

    default_example = """{"relationships": [
  {"source": "source_id", "target": "target_id", "relationship_type": "Serving", "confidence": 0.8}
]}"""

    return f"""You are deriving ArchiMate relationships FROM {source_element_type} elements.

## Instructions
{instruction or default_instruction}

## Source Elements (type: {source_element_type})
```json
{sources_json}
```

## Target Elements (all types)
```json
{targets_json}
```

## Valid Relationship Types for {source_element_type}
{rel_rules_text}

## Example Output Format
```json
{example or default_example}
```

CRITICAL RULES:
1. Source identifiers MUST be from: {json.dumps(source_ids)}
2. Target identifiers MUST be from: {json.dumps(target_ids)}
3. Use identifiers exactly as shown (case-sensitive)
4. Only use relationship types listed above

Return {{"relationships": []}} with source, target, relationship_type for each.
"""


# =============================================================================
# Response Parsing
# =============================================================================


def parse_derivation_response(response: str) -> dict[str, Any]:
    """Parse LLM response for elements."""
    return parse_json_array(response, "elements").to_dict()


def parse_relationship_response(response: str) -> dict[str, Any]:
    """Parse LLM response for relationships."""
    return parse_json_array(response, "relationships").to_dict()


# =============================================================================
# Element Building
# =============================================================================


def sanitize_identifier(identifier: str) -> str:
    """
    Sanitize identifier to be valid XML NCName.

    - Lowercase everything
    - Replace spaces, hyphens, colons with underscores
    - Remove non-alphanumeric characters (except underscore)
    - Ensure starts with letter/underscore
    """
    sanitized = identifier.lower().replace(" ", "_").replace("-", "_").replace(":", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"id_{sanitized}"
    return sanitized


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    """Clamp confidence score to valid [0.0, 1.0] range."""
    try:
        conf = float(value) if value is not None else default
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return default


def build_element(derived: dict[str, Any], element_type: str) -> dict[str, Any]:
    """Build ArchiMate element dict from LLM output."""
    identifier = derived.get("identifier")
    name = derived.get("name")

    if not identifier or not name:
        return {"success": False, "errors": ["Missing identifier or name"]}

    identifier = sanitize_identifier(identifier)
    # Clamp confidence to [0.0, 1.0] range (LLM may return out-of-range values)
    confidence = clamp_confidence(derived.get("confidence"))

    return {
        "success": True,
        "data": {
            "identifier": identifier,
            "name": name,
            "element_type": element_type,
            "documentation": derived.get("documentation", ""),
            "properties": {
                "source": derived.get("source"),
                "confidence": confidence,
                "derived_at": current_timestamp(),
            },
        },
    }


# =============================================================================
# Per-Element Relationship Derivation
# =============================================================================


def build_per_element_relationship_prompt(
    source_elements: list[dict[str, Any]],
    target_elements: list[dict[str, Any]],
    source_element_type: str,
    instruction: str,
    example: str | None = None,
    valid_relationship_types: list[str] | None = None,
) -> str:
    """
    Build LLM prompt for per-element relationship derivation.

    This is used when each element module derives its own relationships,
    providing focused context and better consistency.

    Args:
        source_elements: Elements of the current type (relationships FROM these)
        target_elements: All available target elements
        source_element_type: The ArchiMate element type being processed
        instruction: Custom instruction from database config
        example: Example output from database config
        valid_relationship_types: Allowed relationship types (from config)
    """
    # Strip to essential fields for relationship derivation (reduces tokens ~50%)
    clean_sources = strip_for_relationship_prompt(source_elements)
    clean_targets = strip_for_relationship_prompt(target_elements)
    # Use compact JSON to reduce token usage
    source_json = json.dumps(clean_sources, separators=(",", ":"), default=str)
    target_json = json.dumps(clean_targets, separators=(",", ":"), default=str)

    # Note: identifier lists removed - they're already in the JSON above (saves tokens)
    prompt = f"""You are deriving ArchiMate relationships FROM {source_element_type} elements.

{instruction}

SOURCE ELEMENTS (derive relationships FROM these):
```json
{source_json}
```

TARGET ELEMENTS (derive relationships TO these):
```json
{target_json}
```

RULES:
- Use identifiers EXACTLY as shown in the elements above (case-sensitive)
- Source must be from SOURCE ELEMENTS, target from TARGET ELEMENTS
"""

    if valid_relationship_types:
        prompt += f"""
ALLOWED RELATIONSHIP TYPES (use ONLY these):
{json.dumps(valid_relationship_types)}
"""

    if example:
        prompt += f"""
EXAMPLE OUTPUT:
```json
{example}
```
"""

    prompt += """
Output stable, deterministic results.
Return {"relationships": []} with source, target, relationship_type, confidence for each.
"""
    return prompt


def build_unified_relationship_prompt(
    new_elements: list[dict[str, Any]],
    existing_elements: list[dict[str, Any]],
    element_type: str,
    outbound_rules: list[RelationshipRule],
    inbound_rules: list[RelationshipRule],
) -> str:
    """
    Build LLM prompt for unified relationship derivation (both directions).

    This is used after creating a batch of elements to derive:
    - OUTBOUND: relationships FROM new_elements TO existing_elements
    - INBOUND: relationships FROM existing_elements TO new_elements

    Args:
        new_elements: Elements just created in this batch
        existing_elements: Elements from previous derivation steps
        element_type: The ArchiMate element type just created
        outbound_rules: Rules for relationships FROM this type
        inbound_rules: Rules for relationships TO this type (from other types)

    Returns:
        Prompt string for LLM
    """
    if not new_elements:
        return ""

    # Strip to essential fields for relationship derivation (reduces tokens ~50%)
    clean_new = strip_for_relationship_prompt(new_elements)
    clean_existing = strip_for_relationship_prompt(existing_elements)

    # Use compact JSON to reduce token usage
    new_json = json.dumps(clean_new, separators=(",", ":"), default=str)
    existing_json = json.dumps(clean_existing, separators=(",", ":"), default=str)

    # Build outbound rules text
    outbound_text = ""
    if outbound_rules:
        outbound_lines = []
        for rule in outbound_rules:
            targets_of_type = [
                e
                for e in existing_elements
                if e.get("element_type") == rule.target_type
            ]
            if targets_of_type:
                outbound_lines.append(
                    f"- {element_type} --[{rule.rel_type}]--> {rule.target_type}: {rule.description}"
                )
        if outbound_lines:
            outbound_text = "OUTBOUND (FROM new elements TO existing):\n" + "\n".join(
                outbound_lines
            )

    # Build inbound rules text
    inbound_text = ""
    if inbound_rules:
        inbound_lines = []
        for rule in inbound_rules:
            sources_of_type = [
                e
                for e in existing_elements
                if e.get("element_type") == rule.target_type
            ]
            if sources_of_type:
                inbound_lines.append(
                    f"- {rule.target_type} --[{rule.rel_type}]--> {element_type}: {rule.description}"
                )
        if inbound_lines:
            inbound_text = "INBOUND (FROM existing elements TO new):\n" + "\n".join(
                inbound_lines
            )

    # Note: identifier lists and valid_rel_types removed - they're in the JSON/rules (saves tokens)
    prompt = f"""You are deriving ArchiMate relationships for newly created {element_type} elements.

## New {element_type} Elements (just created)
```json
{new_json}
```

## Existing Elements (from previous steps)
```json
{existing_json}
```

## Relationship Rules
{outbound_text}

{inbound_text}

## Rules
1. ONLY create relationships that match the rules above
2. Use identifiers EXACTLY as shown in the elements above (case-sensitive)
3. Source/target must exist in the elements listed above
4. Set confidence 0.5-1.0 based on clarity
5. Maximum 3 relationships per new element
6. If no valid relationships exist, return empty array
7. Output stable, deterministic results

Return {{"relationships": []}} with source, target, relationship_type, confidence for each.
"""
    return prompt


def derive_batch_relationships(
    new_elements: list[dict[str, Any]],
    existing_elements: list[dict[str, Any]],
    element_type: str,
    outbound_rules: list[RelationshipRule],
    inbound_rules: list[RelationshipRule],
    llm_query_fn: Any,
    temperature: float | None = None,
    max_tokens: int | None = None,
    graph_manager: "GraphManager | None" = None,
) -> list[dict[str, Any]]:
    """
    Derive relationships for a batch of newly created elements.

    Handles both outbound (FROM new TO existing) and inbound (FROM existing TO new).

    Args:
        new_elements: Elements just created in this batch
        existing_elements: Elements from previous derivation steps
        element_type: The ArchiMate element type just created
        outbound_rules: Rules for relationships FROM this type
        inbound_rules: Rules for relationships TO this type
        llm_query_fn: Function to call LLM
        temperature: Optional temperature override
        max_tokens: Optional max_tokens override
        graph_manager: Optional GraphManager for graph-aware filtering.
                      If provided, filters existing_elements to only include
                      those with graph proximity to new_elements.

    Returns:
        List of validated relationship dicts
    """
    # Early returns to avoid unnecessary processing
    if not new_elements:
        return []

    if not outbound_rules and not inbound_rules:
        return []  # No rules defined, skip LLM call entirely

    if not existing_elements:
        return []  # No targets for relationships

    # Check if there are any applicable rules with available targets/sources
    has_outbound_targets = any(
        any(e.get("element_type") == rule.target_type for e in existing_elements)
        for rule in outbound_rules
    )
    has_inbound_sources = any(
        any(e.get("element_type") == rule.target_type for e in existing_elements)
        for rule in inbound_rules
    )

    if not has_outbound_targets and not has_inbound_sources:
        logger.debug("No applicable relationship rules for %s batch", element_type)
        return []

    # Filter existing_elements to only include relevant types (reduces prompt size)
    relevant_types = {r.target_type for r in outbound_rules} | {
        r.target_type for r in inbound_rules
    }
    filtered_existing = [
        e for e in existing_elements if e.get("element_type") in relevant_types
    ]

    # Phase 4.3: Apply graph-aware pre-filtering if graph_manager provided
    # This keeps only elements with graph proximity to new_elements
    if graph_manager and len(filtered_existing) > 20:
        # Extract source IDs from new elements
        new_source_ids = [
            e.get("properties", {}).get("source")
            for e in new_elements
            if e.get("properties", {}).get("source")
        ]
        if new_source_ids:
            # Get connected node IDs (within 2 hops)
            connected_ids = get_connected_source_ids(
                graph_manager, new_source_ids, max_hops=2
            )
            before_count = len(filtered_existing)
            filtered_existing = filter_by_graph_proximity(
                filtered_existing, connected_ids
            )
            logger.debug(
                "Graph-aware filtering: %d -> %d elements (connected to %d sources)",
                before_count,
                len(filtered_existing),
                len(connected_ids),
            )

    # Phase 4: Apply stratified sampling to limit context size
    # This keeps representation from each type while reducing tokens
    if len(filtered_existing) > 50:
        filtered_existing = stratified_sample_elements(
            filtered_existing,
            max_per_type=10,
            relevant_types=list(relevant_types),
        )
        logger.debug(
            "Stratified sampling: reduced to %d elements", len(filtered_existing)
        )

    logger.debug(
        "Filtered existing elements: %d -> %d (relevant types: %s)",
        len(existing_elements),
        len(filtered_existing),
        relevant_types,
    )

    prompt = build_unified_relationship_prompt(
        new_elements=new_elements,
        existing_elements=filtered_existing,
        element_type=element_type,
        outbound_rules=outbound_rules,
        inbound_rules=inbound_rules,
    )

    if not prompt:
        return []

    # Phase 4: Check prompt size and warn if too large
    estimated_tokens, over_threshold = check_prompt_size(prompt)
    if over_threshold:
        logger.warning(
            "Large prompt for %s relationships (%d tokens). Consider fewer elements.",
            element_type,
            estimated_tokens,
        )

    llm_kwargs = {}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    try:
        response = llm_query_fn(prompt, RELATIONSHIP_SCHEMA, **llm_kwargs)
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )
    except Exception as e:
        logger.error("LLM error deriving %s relationships: %s", element_type, e)
        return []

    parse_result = parse_relationship_response(response_content)

    if not parse_result["success"]:
        logger.warning(
            "Failed to parse %s relationships: %s",
            element_type,
            parse_result.get("errors"),
        )
        return []

    # Validate relationships (use filtered_existing for consistency with prompt)
    new_ids = {e.get("identifier", "") for e in new_elements}
    existing_ids = {e.get("identifier", "") for e in filtered_existing}
    all_ids = new_ids | existing_ids

    # Build valid relationship type set
    valid_types = {r.rel_type for r in outbound_rules} | {
        r.rel_type for r in inbound_rules
    }

    relationships = []
    for rel_data in parse_result.get("data", []):
        source = rel_data.get("source")
        target = rel_data.get("target")
        rel_type = rel_data.get("relationship_type")

        # Both endpoints must exist
        if source not in all_ids or target not in all_ids:
            logger.debug(
                "Skipping relationship: endpoint not found (%s -> %s)", source, target
            )
            continue

        # At least one endpoint must be from new elements
        if source not in new_ids and target not in new_ids:
            logger.debug("Skipping relationship: neither endpoint is new")
            continue

        # Validate relationship type
        if rel_type not in valid_types:
            logger.debug("Skipping relationship: invalid type %s", rel_type)
            continue

        relationships.append(
            {
                "source": source,
                "target": target,
                "relationship_type": rel_type,
                "confidence": rel_data.get("confidence", 0.5),
            }
        )

    logger.info(
        "Derived %d relationships for %s batch", len(relationships), element_type
    )
    return relationships


def derive_element_relationships(
    source_elements: list[dict[str, Any]],
    target_elements: list[dict[str, Any]],
    source_element_type: str,
    llm_query_fn: Any,
    instruction: str,
    example: str | None = None,
    valid_relationship_types: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """
    Derive relationships FROM a specific element type.

    This is called by each element module after generating its elements,
    to derive relationships to all available target elements.

    Args:
        source_elements: Elements just created by this module
        target_elements: All elements available as targets (from previous modules)
        source_element_type: The element type being processed
        llm_query_fn: Function to call LLM
        instruction: Prompt instruction from database config
        example: Example output from database config
        valid_relationship_types: Allowed relationship types
        temperature: LLM temperature override
        max_tokens: LLM max_tokens override

    Returns:
        List of relationship dicts ready for persistence
    """
    if not source_elements or not target_elements:
        return []

    prompt = build_per_element_relationship_prompt(
        source_elements=source_elements,
        target_elements=target_elements,
        source_element_type=source_element_type,
        instruction=instruction,
        example=example,
        valid_relationship_types=valid_relationship_types,
    )

    llm_kwargs = {}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    try:
        response = llm_query_fn(prompt, RELATIONSHIP_SCHEMA, **llm_kwargs)
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )
    except Exception as e:
        logger.error(f"LLM error deriving {source_element_type} relationships: {e}")
        return []

    parse_result = parse_relationship_response(response_content)

    if not parse_result["success"]:
        logger.warning(
            f"Failed to parse {source_element_type} relationships: {parse_result.get('errors')}"
        )
        return []

    # Validate and filter relationships
    source_ids = {e.get("identifier", "") for e in source_elements}
    target_ids = {e.get("identifier", "") for e in target_elements}
    valid_types = set(valid_relationship_types) if valid_relationship_types else None

    relationships = []
    for rel_data in parse_result.get("data", []):
        source = rel_data.get("source")
        target = rel_data.get("target")
        rel_type = rel_data.get("relationship_type")

        # Validate source is from this element type
        if source not in source_ids:
            logger.debug(
                f"Skipping relationship: source {source} not in {source_element_type}"
            )
            continue

        # Validate target exists
        if target not in target_ids:
            logger.debug(f"Skipping relationship: target {target} not found")
            continue

        # Validate relationship type
        if valid_types and rel_type not in valid_types:
            logger.debug(
                f"Skipping relationship: type {rel_type} not allowed for {source_element_type}"
            )
            continue

        relationships.append(
            {
                "source": source,
                "target": target,
                "relationship_type": rel_type,
                "confidence": rel_data.get("confidence", 0.5),
            }
        )

    logger.info(
        f"Derived {len(relationships)} relationships FROM {source_element_type}"
    )
    return relationships


# =============================================================================
# Consolidated Relationship Derivation (Phase 4.6)
# =============================================================================


def derive_consolidated_relationships(
    all_elements: list[dict[str, Any]],
    relationship_rules: dict[str, tuple[list[RelationshipRule], list[RelationshipRule]]],
    llm_query_fn: Any,
    graph_manager: "GraphManager | None" = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """
    Derive relationships for all elements in a single consolidated pass.

    This is used when defer_relationships=True in element generation.
    Instead of deriving relationships per-batch, this function processes
    all elements together for better context and consistency.

    Args:
        all_elements: All elements created during generation phase
        relationship_rules: Dict mapping element_type to (outbound_rules, inbound_rules)
        llm_query_fn: Function to call LLM
        graph_manager: Optional GraphManager for graph-aware filtering
        temperature: Optional temperature override
        max_tokens: Optional max_tokens override

    Returns:
        List of all derived relationship dicts
    """
    if not all_elements:
        return []

    all_relationships = []

    # Group elements by type
    by_type: dict[str, list[dict[str, Any]]] = {}
    for elem in all_elements:
        etype = elem.get("element_type", "Unknown")
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(elem)

    logger.info(
        "Consolidated relationship derivation: %d elements across %d types",
        len(all_elements),
        len(by_type),
    )

    # Process each element type
    for element_type, type_elements in by_type.items():
        if element_type not in relationship_rules:
            continue

        outbound_rules, inbound_rules = relationship_rules[element_type]
        if not outbound_rules and not inbound_rules:
            continue

        # Get all other elements as potential targets
        other_elements = [e for e in all_elements if e.get("element_type") != element_type]

        relationships = derive_batch_relationships(
            new_elements=type_elements,
            existing_elements=other_elements,
            element_type=element_type,
            outbound_rules=outbound_rules,
            inbound_rules=inbound_rules,
            llm_query_fn=llm_query_fn,
            temperature=temperature,
            max_tokens=max_tokens,
            graph_manager=graph_manager,
        )

        all_relationships.extend(relationships)
        logger.debug(
            "Derived %d relationships for %s (%d elements)",
            len(relationships),
            element_type,
            len(type_elements),
        )

    logger.info("Total relationships derived: %d", len(all_relationships))
    return all_relationships


# =============================================================================
# Result Creation
# =============================================================================


def create_result(
    success: bool,
    errors: list[str] | None = None,
    stats: dict[str, Any] | None = None,
) -> PipelineResult:
    """Create a simple pipeline result."""
    return {
        "success": success,
        "errors": errors or [],
        "stats": stats or {},
        "timestamp": current_timestamp(),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "ESSENTIAL_PROPS",
    "RELATIONSHIP_ESSENTIAL_FIELDS",
    # Utility functions
    "strip_for_relationship_prompt",
    # Data structures
    "Candidate",
    "RelationshipRule",
    "GenerationResult",
    "DerivationResult",
    # Enrichment
    "get_enrichments",
    "get_enrichments_from_neo4j",
    "enrich_candidate",
    # Filtering
    "filter_by_pagerank",
    "filter_by_labels",
    "filter_by_community",
    "get_community_roots",
    "get_articulation_points",
    # Token estimation & context limiting (Phase 4)
    "MODEL_CONTEXT_LIMITS",
    "estimate_tokens",
    "get_model_context_limit",
    "limit_existing_elements",
    "stratified_sample_elements",
    "check_prompt_size",
    # Graph-aware filtering (Phase 4.3)
    "get_connected_source_ids",
    "filter_by_graph_proximity",
    # Batching
    "calculate_dynamic_batch_size",
    "adjust_batch_for_tokens",
    "batch_candidates",
    # Query
    "query_candidates",
    # Schemas
    "DERIVATION_SCHEMA",
    "RELATIONSHIP_SCHEMA",
    # Prompts
    "build_derivation_prompt",
    "build_relationship_prompt",
    "build_element_relationship_prompt",
    "build_per_element_relationship_prompt",
    "build_unified_relationship_prompt",
    # Response handling
    "extract_response_content",
    # Parsing
    "parse_derivation_response",
    "parse_relationship_response",
    # Element building
    "clamp_confidence",
    "sanitize_identifier",
    "build_element",
    # Relationship derivation
    "derive_element_relationships",
    "derive_batch_relationships",
    "derive_consolidated_relationships",
    # Results
    "create_result",
]
