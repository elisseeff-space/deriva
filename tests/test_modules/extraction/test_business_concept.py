"""Tests for modules.extraction.business_concept module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from deriva.modules.extraction.business_concept import (
    BUSINESS_CONCEPT_SCHEMA,
    _dir_name_to_concept_name,
    build_extraction_prompt,
    extract_seed_concepts_from_structure,
    get_existing_concepts_from_graph,
)


class TestBusinessConceptSchema:
    """Tests for the JSON schema definition."""

    def test_schema_has_required_structure(self):
        """Schema should have the expected structure."""
        assert BUSINESS_CONCEPT_SCHEMA["name"] == "business_concepts_extraction"
        assert BUSINESS_CONCEPT_SCHEMA["strict"] is True
        schema = cast(dict[str, Any], BUSINESS_CONCEPT_SCHEMA["schema"])
        assert "concepts" in schema["properties"]

    def test_schema_concept_fields(self):
        """Schema should define all concept fields."""
        schema = cast(dict[str, Any], BUSINESS_CONCEPT_SCHEMA["schema"])
        items = schema["properties"]["concepts"]["items"]
        required_fields = items["required"]

        assert "conceptName" in required_fields
        assert "conceptType" in required_fields
        assert "description" in required_fields
        assert "confidence" in required_fields

    def test_concept_type_enum_values(self):
        """Schema should have correct enum values for conceptType."""
        schema = cast(dict[str, Any], BUSINESS_CONCEPT_SCHEMA["schema"])
        items = schema["properties"]["concepts"]["items"]
        enum_values = items["properties"]["conceptType"]["enum"]

        assert "actor" in enum_values
        assert "service" in enum_values
        assert "process" in enum_values
        assert "entity" in enum_values
        assert "event" in enum_values


class TestDirNameToConceptName:
    """Tests for _dir_name_to_concept_name helper function."""

    def test_converts_snake_case(self):
        """Should convert snake_case to PascalCase."""
        assert _dir_name_to_concept_name("data_processing") == "DataProcessing"
        assert _dir_name_to_concept_name("user_management") == "UserManagement"

    def test_converts_kebab_case(self):
        """Should convert kebab-case to PascalCase."""
        assert _dir_name_to_concept_name("message-queue") == "MessageQueue"
        assert _dir_name_to_concept_name("api-gateway") == "ApiGateway"

    def test_handles_single_word(self):
        """Should capitalize single word directories."""
        assert _dir_name_to_concept_name("orders") == "Orders"
        assert _dir_name_to_concept_name("analytics") == "Analytics"

    def test_handles_empty_string(self):
        """Should handle empty string."""
        assert _dir_name_to_concept_name("") == ""

    def test_handles_mixed_separators(self):
        """Should handle mixed underscores and hyphens."""
        assert _dir_name_to_concept_name("data-stream_processor") == "DataStreamProcessor"


class TestExtractSeedConceptsFromStructure:
    """Tests for extract_seed_concepts_from_structure function."""

    def test_extracts_from_directories(self):
        """Should extract concepts from directory names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directories
            (Path(tmpdir) / "orders").mkdir()
            (Path(tmpdir) / "customers").mkdir()
            (Path(tmpdir) / "processing_pipeline").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            names = [c["conceptName"] for c in concepts]
            assert "Orders" in names
            assert "Customers" in names
            assert "ProcessingPipeline" in names

    def test_skips_common_directories(self):
        """Should skip non-business directories like .git, node_modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directories that should be skipped
            (Path(tmpdir) / ".git").mkdir()
            (Path(tmpdir) / "node_modules").mkdir()
            (Path(tmpdir) / "__pycache__").mkdir()
            (Path(tmpdir) / "tests").mkdir()
            (Path(tmpdir) / "venv").mkdir()
            # Create one that should be included
            (Path(tmpdir) / "orders").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            names = [c["conceptName"] for c in concepts]
            assert "Orders" in names
            assert len(concepts) == 1  # Only 'orders' should be included

    def test_classifies_process_keywords(self):
        """Should classify directories with process keywords as 'process'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data_processing").mkdir()
            (Path(tmpdir) / "stream_pipeline").mkdir()
            (Path(tmpdir) / "job_scheduler").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            for concept in concepts:
                assert concept["conceptType"] == "process"

    def test_classifies_entity_keywords(self):
        """Should classify directories with entity keywords as 'entity'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data_model").mkdir()
            (Path(tmpdir) / "user_repository").mkdir()
            (Path(tmpdir) / "cache_storage").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            for concept in concepts:
                assert concept["conceptType"] == "entity"

    def test_classifies_unknown_as_capability(self):
        """Should classify unknown directories as 'capability'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "orders").mkdir()
            (Path(tmpdir) / "customers").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            for concept in concepts:
                assert concept["conceptType"] == "capability"

    def test_includes_source_info(self):
        """Should include source information for each concept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "orders").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            assert len(concepts) == 1
            assert concepts[0]["source"] == "directory:orders/"

    def test_skips_short_names(self):
        """Should skip directories with names <= 2 characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "ab").mkdir()
            (Path(tmpdir) / "xy").mkdir()
            (Path(tmpdir) / "orders").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            names = [c["conceptName"] for c in concepts]
            assert "Orders" in names
            assert "Ab" not in names
            assert "Xy" not in names

    def test_skips_hidden_directories(self):
        """Should skip directories starting with dot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".hidden").mkdir()
            (Path(tmpdir) / ".config").mkdir()
            (Path(tmpdir) / "orders").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir)

            names = [c["conceptName"] for c in concepts]
            assert "Orders" in names
            assert len(concepts) == 1

    def test_disabled_when_include_directories_false(self):
        """Should return empty when include_directories is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "orders").mkdir()
            (Path(tmpdir) / "customers").mkdir()

            concepts = extract_seed_concepts_from_structure(tmpdir, include_directories=False)

            assert len(concepts) == 0


class TestGetExistingConceptsFromGraph:
    """Tests for get_existing_concepts_from_graph function."""

    def test_returns_concepts_from_query(self):
        """Should return concepts from graph query."""
        mock_manager = MagicMock()
        mock_manager.run_cypher.return_value = [
            {"conceptName": "OrderManagement", "conceptType": "entity", "source": "graph"},
            {"conceptName": "CustomerService", "conceptType": "service", "source": "graph"},
        ]

        concepts = get_existing_concepts_from_graph(mock_manager, "myrepo")

        assert len(concepts) == 2
        assert concepts[0]["conceptName"] == "OrderManagement"
        assert concepts[1]["conceptName"] == "CustomerService"

    def test_filters_none_concept_names(self):
        """Should filter out results with None conceptName."""
        mock_manager = MagicMock()
        mock_manager.run_cypher.return_value = [
            {"conceptName": "ValidConcept", "conceptType": "entity", "source": "graph"},
            {"conceptName": None, "conceptType": "entity", "source": "graph"},
        ]

        concepts = get_existing_concepts_from_graph(mock_manager, "myrepo")

        assert len(concepts) == 1
        assert concepts[0]["conceptName"] == "ValidConcept"

    def test_returns_empty_on_exception(self):
        """Should return empty list when query fails."""
        mock_manager = MagicMock()
        mock_manager.run_cypher.side_effect = Exception("Connection failed")

        concepts = get_existing_concepts_from_graph(mock_manager, "myrepo")

        assert concepts == []

    def test_uses_correct_query_params(self):
        """Should pass correct parameters to query."""
        mock_manager = MagicMock()
        mock_manager.run_cypher.return_value = []

        get_existing_concepts_from_graph(mock_manager, "testrepo", limit=25)

        mock_manager.run_cypher.assert_called_once()
        call_args = mock_manager.run_cypher.call_args
        assert call_args[0][1]["repo_name"] == "testrepo"
        assert call_args[0][1]["limit"] == 25


class TestBuildExtractionPrompt:
    """Tests for build_extraction_prompt function."""

    def test_includes_file_content(self):
        """Should include file content in prompt."""
        prompt = build_extraction_prompt(
            file_content="class OrderService:\n    pass",
            file_path="src/orders/service.py",
            instruction="Extract business concepts",
            example='{"concepts": []}',
        )

        assert "OrderService" in prompt

    def test_includes_file_path(self):
        """Should include file path in prompt."""
        prompt = build_extraction_prompt(
            file_content="test content",
            file_path="src/orders/service.py",
            instruction="Extract concepts",
            example="{}",
        )

        assert "src/orders/service.py" in prompt

    def test_includes_instruction(self):
        """Should include instruction in prompt."""
        prompt = build_extraction_prompt(
            file_content="content",
            file_path="test.py",
            instruction="Focus on domain entities and business processes",
            example="{}",
        )

        assert "Focus on domain entities" in prompt

    def test_includes_example(self):
        """Should include example output in prompt."""
        prompt = build_extraction_prompt(
            file_content="content",
            file_path="test.py",
            instruction="Extract",
            example='{"concepts": [{"conceptName": "Order"}]}',
        )

        assert "Order" in prompt

    def test_includes_existing_concepts_when_provided(self):
        """Should include existing concepts section when provided."""
        existing: list[dict[str, Any]] = [
            {"conceptName": "CustomerManagement", "conceptType": "entity", "confidence": 0.9},
        ]
        prompt = build_extraction_prompt(
            file_content="content",
            file_path="test.py",
            instruction="Extract",
            example="{}",
            existing_concepts=existing,  # type: ignore[arg-type]
        )

        assert "CustomerManagement" in prompt
        assert "Previously Identified Concepts" in prompt

    def test_filters_low_confidence_existing_concepts(self):
        """Should filter out low confidence existing concepts."""
        existing: list[dict[str, Any]] = [
            {"conceptName": "HighConfidence", "conceptType": "entity", "confidence": 0.9},
            {"conceptName": "LowConfidence", "conceptType": "entity", "confidence": 0.5},
        ]
        prompt = build_extraction_prompt(
            file_content="content",
            file_path="test.py",
            instruction="Extract",
            example="{}",
            existing_concepts=existing,  # type: ignore[arg-type]
        )

        assert "HighConfidence" in prompt
        assert "LowConfidence" not in prompt

    def test_handles_empty_existing_concepts(self):
        """Should handle empty existing concepts list."""
        prompt = build_extraction_prompt(
            file_content="content",
            file_path="test.py",
            instruction="Extract",
            example="{}",
            existing_concepts=[],
        )

        assert "Previously Identified Concepts" not in prompt
