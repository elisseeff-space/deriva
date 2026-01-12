"""
Base classes for element derivation modules.

Provides the common generate() flow that all element modules share,
eliminating ~80% code duplication across 13 element modules.

Usage:
    # For pattern-based filtering (most modules):
    class BusinessObjectDerivation(PatternBasedDerivation):
        ELEMENT_TYPE = "BusinessObject"
        OUTBOUND_RULES = [...]
        INBOUND_RULES = [...]

        def filter_candidates(self, candidates, enrichments, max_candidates,
                              include_patterns=None, exclude_patterns=None, **kwargs):
            # Module-specific filtering logic
            ...

    # For graph-based filtering (e.g., ApplicationComponent):
    class ApplicationComponentDerivation(ElementDerivationBase):
        ELEMENT_TYPE = "ApplicationComponent"
        OUTBOUND_RULES = [...]
        INBOUND_RULES = [...]

        def filter_candidates(self, candidates, enrichments, max_candidates, **kwargs):
            # Uses community roots, PageRank, etc.
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from deriva.adapters.archimate.models import Element, Relationship
from deriva.modules.derivation.base import (
    DERIVATION_SCHEMA,
    Candidate,
    GenerationResult,
    RelationshipRule,
    batch_candidates,
    build_derivation_prompt,
    build_element,
    derive_batch_relationships,
    extract_response_content,
    get_enrichments_from_neo4j,
    parse_derivation_response,
    query_candidates,
)

if TYPE_CHECKING:
    from deriva.adapters.archimate import ArchimateManager
    from deriva.adapters.graph import GraphManager


class ElementDerivationBase(ABC):
    """
    Abstract base class for element derivation modules.

    Provides the common generate() flow shared by all element types.
    Subclasses must define:
    - ELEMENT_TYPE: The ArchiMate element type name
    - OUTBOUND_RULES: Relationships FROM this element type
    - INBOUND_RULES: Relationships TO this element type
    - filter_candidates(): Module-specific candidate filtering

    The generate() method handles:
    - Enrichment retrieval from Neo4j
    - Candidate querying
    - Batching
    - LLM calls for element derivation
    - Element creation in ArchimateManager
    - Relationship derivation
    """

    # Subclasses MUST override these
    ELEMENT_TYPE: str
    OUTBOUND_RULES: list[RelationshipRule]
    INBOUND_RULES: list[RelationshipRule]

    def __init__(self) -> None:
        """Initialize the derivation class."""
        self.logger = logging.getLogger(self.__class__.__module__)

    @abstractmethod
    def filter_candidates(
        self,
        candidates: list[Candidate],
        enrichments: dict[str, dict[str, Any]],
        max_candidates: int,
        **kwargs: Any,
    ) -> list[Candidate]:
        """
        Filter candidates for this element type.

        Each module implements its own filtering strategy based on:
        - Pattern matching (include/exclude patterns from config)
        - Graph structure (community roots, PageRank)
        - Domain-specific heuristics

        Args:
            candidates: Raw candidates from graph query
            enrichments: Graph enrichment data (pagerank, community, kcore, etc.)
            max_candidates: Maximum candidates to return
            **kwargs: Additional module-specific parameters
                     (e.g., include_patterns, exclude_patterns for pattern-based modules)

        Returns:
            Filtered list of candidates, limited to max_candidates
        """
        ...

    def get_filter_kwargs(self, engine: Any) -> dict[str, Any]:
        """
        Get additional kwargs for filter_candidates().

        Pattern-based modules override this to load patterns from config.
        Graph-based modules can use the default (empty dict).

        Args:
            engine: DuckDB connection for config queries

        Returns:
            Dict of kwargs to pass to filter_candidates()
        """
        return {}

    def generate(
        self,
        graph_manager: "GraphManager",
        archimate_manager: "ArchimateManager",
        engine: Any,
        llm_query_fn: Callable[..., Any],
        query: str,
        instruction: str,
        example: str,
        max_candidates: int,
        batch_size: int,
        existing_elements: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        defer_relationships: bool = False,
    ) -> GenerationResult:
        """
        Generate elements of this type.

        This is the common flow shared by all element modules:
        1. Get filter kwargs (patterns for pattern-based modules)
        2. Get enrichments from Neo4j
        3. Query candidates
        4. Filter candidates (module-specific)
        5. Batch candidates
        6. For each batch: LLM call -> parse -> create elements -> derive relationships

        Args:
            graph_manager: GraphManager for querying graph nodes
            archimate_manager: ArchimateManager for creating elements
            engine: DuckDB connection for config
            llm_query_fn: Function to call LLM
            query: Cypher query to find candidates
            instruction: LLM instruction for derivation
            example: Example output for LLM
            max_candidates: Maximum candidates to process
            batch_size: Candidates per LLM batch
            existing_elements: Already-created elements for relationship derivation
            temperature: Optional LLM temperature override
            max_tokens: Optional LLM max_tokens override
            defer_relationships: If True, skip relationship derivation

        Returns:
            GenerationResult with success status, counts, and any errors
        """
        result = GenerationResult(success=True)

        # Get filter kwargs (patterns for pattern-based modules)
        filter_kwargs = self.get_filter_kwargs(engine)

        # Get enrichments and query candidates
        enrichments = get_enrichments_from_neo4j(graph_manager)

        try:
            candidates = query_candidates(graph_manager, query, enrichments)
        except Exception as e:
            return GenerationResult(
                success=False,
                errors=[f"Query failed for {self.ELEMENT_TYPE}: {e}"],
            )

        if not candidates:
            self.logger.info("No candidates found for %s", self.ELEMENT_TYPE)
            return result

        self.logger.info(
            "Found %d candidates for %s", len(candidates), self.ELEMENT_TYPE
        )

        # Filter candidates (module-specific)
        filtered = self.filter_candidates(
            candidates, enrichments, max_candidates, **filter_kwargs
        )

        if not filtered:
            self.logger.info("No candidates passed filtering for %s", self.ELEMENT_TYPE)
            return result

        self.logger.info(
            "Filtered to %d candidates for LLM (%s)", len(filtered), self.ELEMENT_TYPE
        )

        # Batch and process
        batches = batch_candidates(filtered, batch_size)

        llm_kwargs: dict[str, Any] = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens

        for batch_num, batch in enumerate(batches, 1):
            self._process_batch(
                batch_num=batch_num,
                batch=batch,
                instruction=instruction,
                example=example,
                llm_query_fn=llm_query_fn,
                llm_kwargs=llm_kwargs,
                archimate_manager=archimate_manager,
                graph_manager=graph_manager,
                existing_elements=existing_elements,
                temperature=temperature,
                max_tokens=max_tokens,
                defer_relationships=defer_relationships,
                result=result,
            )

        self.logger.info(
            "Created %d %s elements and %d relationships",
            result.elements_created,
            self.ELEMENT_TYPE,
            result.relationships_created,
        )
        return result

    def _process_batch(
        self,
        batch_num: int,
        batch: list[Candidate],
        instruction: str,
        example: str,
        llm_query_fn: Callable[..., Any],
        llm_kwargs: dict[str, Any],
        archimate_manager: "ArchimateManager",
        graph_manager: "GraphManager",
        existing_elements: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        defer_relationships: bool,
        result: GenerationResult,
    ) -> None:
        """
        Process a single batch of candidates.

        Handles LLM call, response parsing, element creation, and
        relationship derivation for one batch.

        Args:
            batch_num: Batch number (for error reporting)
            batch: List of candidates in this batch
            instruction: LLM instruction
            example: LLM example
            llm_query_fn: LLM query function
            llm_kwargs: Additional LLM parameters
            archimate_manager: For creating elements
            graph_manager: For relationship derivation
            existing_elements: For relationship derivation
            temperature: LLM temperature
            max_tokens: LLM max tokens
            defer_relationships: Skip relationship derivation if True
            result: GenerationResult to update with counts and errors
        """
        # Build prompt
        prompt = build_derivation_prompt(
            candidates=batch,
            instruction=instruction,
            example=example,
            element_type=self.ELEMENT_TYPE,
        )

        # Call LLM
        try:
            response = llm_query_fn(prompt, DERIVATION_SCHEMA, **llm_kwargs)
            response_content, error = extract_response_content(response)
            if error:
                result.errors.append(
                    f"LLM error in batch {batch_num} ({self.ELEMENT_TYPE}): {error}"
                )
                return
        except Exception as e:
            result.errors.append(
                f"LLM error in batch {batch_num} ({self.ELEMENT_TYPE}): {e}"
            )
            return

        # Parse response
        parse_result = parse_derivation_response(response_content)
        if not parse_result["success"]:
            result.errors.extend(
                [
                    f"{self.ELEMENT_TYPE} batch {batch_num}: {e}"
                    for e in parse_result.get("errors", [])
                ]
            )
            return

        # Build enrichment lookup for this batch
        batch_enrichments = {
            c.node_id: {
                "pagerank": c.pagerank,
                "louvain_community": c.louvain_community,
            }
            for c in batch
        }

        # Create elements
        batch_elements: list[dict[str, Any]] = []
        for derived in parse_result.get("data", []):
            element_result = build_element(
                derived, self.ELEMENT_TYPE, batch_enrichments
            )

            if not element_result["success"]:
                result.errors.extend(element_result.get("errors", []))
                continue

            element_data = element_result["data"]

            try:
                element = Element(
                    name=element_data["name"],
                    element_type=element_data["element_type"],
                    identifier=element_data["identifier"],
                    documentation=element_data.get("documentation"),
                    properties=element_data.get("properties", {}),
                )
                archimate_manager.add_element(element)
                result.elements_created += 1
                result.created_elements.append(element_data)
                batch_elements.append(element_data)
            except Exception as e:
                result.errors.append(
                    f"Failed to create {self.ELEMENT_TYPE} element "
                    f"{element_data.get('identifier', 'unknown')}: {e}"
                )

        # Derive relationships
        if batch_elements and existing_elements and not defer_relationships:
            self._derive_relationships(
                batch_elements=batch_elements,
                existing_elements=existing_elements,
                llm_query_fn=llm_query_fn,
                temperature=temperature,
                max_tokens=max_tokens,
                graph_manager=graph_manager,
                archimate_manager=archimate_manager,
                result=result,
            )

    def _derive_relationships(
        self,
        batch_elements: list[dict[str, Any]],
        existing_elements: list[dict[str, Any]],
        llm_query_fn: Callable[..., Any],
        temperature: float | None,
        max_tokens: int | None,
        graph_manager: "GraphManager",
        archimate_manager: "ArchimateManager",
        result: GenerationResult,
    ) -> None:
        """
        Derive relationships for newly created elements.

        Args:
            batch_elements: Elements created in this batch
            existing_elements: All previously created elements
            llm_query_fn: LLM query function
            temperature: LLM temperature
            max_tokens: LLM max tokens
            graph_manager: For graph-based relationship derivation
            archimate_manager: For creating relationships
            result: GenerationResult to update
        """
        relationships = derive_batch_relationships(
            new_elements=batch_elements,
            existing_elements=existing_elements,
            element_type=self.ELEMENT_TYPE,
            outbound_rules=self.OUTBOUND_RULES,
            inbound_rules=self.INBOUND_RULES,
            llm_query_fn=llm_query_fn,
            temperature=temperature,
            max_tokens=max_tokens,
            graph_manager=graph_manager,
        )

        for rel_data in relationships:
            try:
                relationship = Relationship(
                    source=rel_data["source"],
                    target=rel_data["target"],
                    relationship_type=rel_data["relationship_type"],
                    properties={"confidence": rel_data.get("confidence", 0.5)},
                )
                archimate_manager.add_relationship(relationship)
                result.relationships_created += 1
                result.created_relationships.append(rel_data)
            except Exception as e:
                result.errors.append(
                    f"Failed to create {self.ELEMENT_TYPE} relationship: {e}"
                )


class PatternBasedDerivation(ElementDerivationBase):
    """
    Mixin for element modules that use pattern-based filtering.

    Most element modules (10 out of 13) use include/exclude patterns
    loaded from the config database. This mixin provides the default
    implementation of get_filter_kwargs() to load those patterns.

    Modules using this base class receive include_patterns and
    exclude_patterns as kwargs to their filter_candidates() method.
    """

    def get_filter_kwargs(self, engine: Any) -> dict[str, Any]:
        """
        Load include/exclude patterns from config database.

        Args:
            engine: DuckDB connection

        Returns:
            Dict with include_patterns and exclude_patterns sets
        """
        from deriva.services import config

        try:
            patterns = config.get_derivation_patterns(engine, self.ELEMENT_TYPE)
            return {
                "include_patterns": patterns.get("include", set()),
                "exclude_patterns": patterns.get("exclude", set()),
            }
        except ValueError:
            # No patterns configured - return empty sets
            self.logger.debug(
                "No derivation patterns found for %s, using empty sets",
                self.ELEMENT_TYPE,
            )
            return {
                "include_patterns": set(),
                "exclude_patterns": set(),
            }


__all__ = [
    "ElementDerivationBase",
    "PatternBasedDerivation",
]
