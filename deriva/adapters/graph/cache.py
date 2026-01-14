"""
Caching functionality for graph operations.

Provides caching for expensive Neo4j queries like enrichment fetching.
Uses graph state hash to detect when cache should be invalidated.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from deriva.common.cache_utils import BaseDiskCache, hash_inputs

# Default graph cache directory (can be overridden via GRAPH_CACHE_DIR env var)
GRAPH_CACHE_DIR = os.getenv("GRAPH_CACHE_DIR", "workspace/cache/graph")

if TYPE_CHECKING:
    from deriva.adapters.graph.manager import GraphManager

logger = logging.getLogger(__name__)


def compute_graph_hash(graph_manager: "GraphManager") -> str:
    """
    Compute a hash representing the current graph state.

    The hash is based on:
    - Count of active nodes
    - Count of edges
    - Namespace (to differentiate graphs)

    This provides a fast way to detect if the graph has changed
    since the last cache operation.

    Args:
        graph_manager: Connected GraphManager instance

    Returns:
        SHA256 hash string representing graph state
    """
    try:
        # Get node and edge counts for active nodes
        stats_query = """
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label STARTS WITH 'Graph:')
              AND n.active = true
            WITH count(n) as node_count
            OPTIONAL MATCH ()-[r]->()
            WHERE type(r) STARTS WITH 'Graph:'
            RETURN node_count, count(r) as edge_count
        """
        results = graph_manager.query(stats_query)
        if results:
            node_count = results[0].get("node_count", 0)
            edge_count = results[0].get("edge_count", 0)
        else:
            node_count = 0
            edge_count = 0

        # Include namespace in hash
        namespace = getattr(graph_manager, "namespace", "Graph")

        return hash_inputs(namespace, node_count, edge_count)

    except Exception as e:
        logger.warning(f"Failed to compute graph hash: {e}")
        # Return a unique hash that won't match anything cached
        return hash_inputs("error", str(e))


class EnrichmentCache(BaseDiskCache):
    """
    Cache for graph enrichment data (PageRank, Louvain, k-core, etc.).

    Enrichments are expensive to compute and don't change unless
    the graph structure changes. This cache stores enrichment data
    keyed by graph state hash.

    Example:
        cache = EnrichmentCache()
        graph_hash = compute_graph_hash(graph_manager)

        if cached := cache.get_enrichments(graph_hash):
            return cached

        # Compute enrichments from Neo4j
        enrichments = get_enrichments_from_neo4j(graph_manager)
        cache.set_enrichments(graph_hash, enrichments)
        return enrichments
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize enrichment cache.

        Args:
            cache_dir: Directory to store cache files (default: GRAPH_CACHE_DIR/enrichments)
        """
        if cache_dir is None:
            cache_dir = f"{GRAPH_CACHE_DIR}/enrichments"
        super().__init__(cache_dir)

    def get_enrichments(self, graph_hash: str) -> dict[str, dict[str, Any]] | None:
        """
        Get cached enrichments for a graph state.

        Args:
            graph_hash: Hash from compute_graph_hash()

        Returns:
            Dict mapping node_id to enrichment data, or None if not cached
        """
        cached = self.get(graph_hash)
        if cached is not None:
            logger.debug(f"Enrichment cache HIT for graph hash {graph_hash[:8]}...")
            return cached.get("enrichments")
        logger.debug(f"Enrichment cache MISS for graph hash {graph_hash[:8]}...")
        return None

    def set_enrichments(
        self, graph_hash: str, enrichments: dict[str, dict[str, Any]]
    ) -> None:
        """
        Cache enrichments for a graph state.

        Args:
            graph_hash: Hash from compute_graph_hash()
            enrichments: Dict mapping node_id to enrichment data
        """
        self.set(graph_hash, {"enrichments": enrichments, "graph_hash": graph_hash})
        logger.debug(
            f"Cached {len(enrichments)} enrichments for graph hash {graph_hash[:8]}..."
        )


class QueryCache(BaseDiskCache):
    """
    Cache for Cypher query results.

    Caches query results keyed by query string + graph state hash.
    Useful for queries that are repeated multiple times per run.

    Example:
        cache = QueryCache()
        graph_hash = compute_graph_hash(graph_manager)

        cache_key = cache.generate_key(cypher_query, graph_hash)
        if cached := cache.get(cache_key):
            return cached["results"]

        results = graph_manager.query(cypher_query)
        cache.set(cache_key, {"results": results})
        return results
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize query cache.

        Args:
            cache_dir: Directory to store cache files (default: GRAPH_CACHE_DIR/queries)
        """
        if cache_dir is None:
            cache_dir = f"{GRAPH_CACHE_DIR}/queries"
        super().__init__(cache_dir)

    @staticmethod
    def generate_key(query: str, graph_hash: str) -> str:
        """
        Generate cache key for a query.

        Args:
            query: Cypher query string
            graph_hash: Hash from compute_graph_hash()

        Returns:
            SHA256 hash as cache key
        """
        return hash_inputs(query, graph_hash)

    def get_results(self, query: str, graph_hash: str) -> list[dict[str, Any]] | None:
        """
        Get cached query results.

        Args:
            query: Cypher query string
            graph_hash: Hash from compute_graph_hash()

        Returns:
            Query results or None if not cached
        """
        cache_key = self.generate_key(query, graph_hash)
        cached = self.get(cache_key)
        if cached is not None:
            return cached.get("results")
        return None

    def set_results(
        self, query: str, graph_hash: str, results: list[dict[str, Any]]
    ) -> None:
        """
        Cache query results.

        Args:
            query: Cypher query string
            graph_hash: Hash from compute_graph_hash()
            results: Query results to cache
        """
        cache_key = self.generate_key(query, graph_hash)
        self.set(cache_key, {"results": results, "query": query[:200]})


__all__ = [
    "compute_graph_hash",
    "EnrichmentCache",
    "QueryCache",
]
