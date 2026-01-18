"""Tests for modules.extraction.edges module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deriva.modules.extraction.edges import (
    ALL_EDGE_TYPES,
    EdgeType,
    PYTHON_STDLIB,
    _build_method_lookup,
    _extract_type_names,
    _resolve_absolute_import,
    _resolve_import,
    _resolve_relative_import,
    extract_edges_batch,
    extract_edges_from_file,
)


# =============================================================================
# EdgeType Enum Tests
# =============================================================================


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_imports_value(self):
        """Should have IMPORTS edge type."""
        assert EdgeType.IMPORTS.value == "IMPORTS"

    def test_uses_value(self):
        """Should have USES edge type."""
        assert EdgeType.USES.value == "USES"

    def test_calls_value(self):
        """Should have CALLS edge type."""
        assert EdgeType.CALLS.value == "CALLS"

    def test_decorated_by_value(self):
        """Should have DECORATED_BY edge type."""
        assert EdgeType.DECORATED_BY.value == "DECORATED_BY"

    def test_references_value(self):
        """Should have REFERENCES edge type."""
        assert EdgeType.REFERENCES.value == "REFERENCES"


class TestAllEdgeTypes:
    """Tests for ALL_EDGE_TYPES constant."""

    def test_contains_all_edge_types(self):
        """Should contain all edge types."""
        assert EdgeType.IMPORTS in ALL_EDGE_TYPES
        assert EdgeType.USES in ALL_EDGE_TYPES
        assert EdgeType.CALLS in ALL_EDGE_TYPES
        assert EdgeType.DECORATED_BY in ALL_EDGE_TYPES
        assert EdgeType.REFERENCES in ALL_EDGE_TYPES

    def test_has_five_types(self):
        """Should have exactly 5 edge types."""
        assert len(ALL_EDGE_TYPES) == 5


# =============================================================================
# _resolve_relative_import Tests
# =============================================================================


class TestResolveRelativeImport:
    """Tests for _resolve_relative_import function."""

    def test_resolves_single_dot_import_same_dir(self):
        """Should resolve single-dot relative import in same directory."""
        all_files = {"models.py"}
        result = _resolve_relative_import(
            module=".models",
            current_file="main.py",
            all_file_paths=all_files,
        )
        assert result == "models.py"

    def test_returns_none_for_not_found(self):
        """Should return None when import cannot be resolved."""
        all_files = {"other.py"}
        result = _resolve_relative_import(
            module=".models",
            current_file="main.py",
            all_file_paths=all_files,
        )
        assert result is None

    def test_handles_empty_module_after_dots(self):
        """Should handle import with just dots (package import)."""
        all_files = {"__init__.py"}
        result = _resolve_relative_import(
            module=".",
            current_file="main.py",
            all_file_paths=all_files,
        )
        # Should look for __init__.py in current dir
        assert result == "__init__.py" or result is None


# =============================================================================
# _resolve_absolute_import Tests
# =============================================================================


class TestResolveAbsoluteImport:
    """Tests for _resolve_absolute_import function."""

    def test_resolves_simple_module(self):
        """Should resolve simple module name."""
        all_files = {"utils.py"}
        result = _resolve_absolute_import(
            module="utils",
            all_file_paths=all_files,
        )
        assert result == "utils.py"

    def test_resolves_nested_module(self):
        """Should resolve nested module path."""
        all_files = {"src/utils/helpers.py"}
        result = _resolve_absolute_import(
            module="src.utils.helpers",
            all_file_paths=all_files,
        )
        assert result == "src/utils/helpers.py"

    def test_resolves_package_init(self):
        """Should resolve to package __init__.py."""
        all_files = {"mypackage/__init__.py"}
        result = _resolve_absolute_import(
            module="mypackage",
            all_file_paths=all_files,
        )
        assert result == "mypackage/__init__.py"

    def test_returns_none_for_not_found(self):
        """Should return None when module not found."""
        all_files = {"other.py"}
        result = _resolve_absolute_import(
            module="utils",
            all_file_paths=all_files,
        )
        assert result is None

    def test_resolves_submodule_to_parent(self):
        """Should resolve submodule to parent file if exists."""
        all_files = {"utils.py"}
        result = _resolve_absolute_import(
            module="utils.helper",  # utils.helper -> utils.py
            all_file_paths=all_files,
        )
        assert result == "utils.py"


# =============================================================================
# _resolve_import Tests
# =============================================================================


class TestResolveImport:
    """Tests for _resolve_import function."""

    def test_resolves_internal_import(self):
        """Should identify internal imports."""
        all_files = {"src/models.py"}
        external_packages = {"flask", "django"}
        result = _resolve_import(
            module="src.models",
            current_file="src/main.py",
            all_file_paths=all_files,
            external_packages=external_packages,
        )
        assert result["type"] == "internal"
        assert result["target_path"] == "src/models.py"

    def test_resolves_external_import(self):
        """Should identify external imports."""
        all_files = {"src/main.py"}
        external_packages = {"flask", "django"}
        result = _resolve_import(
            module="flask.views",
            current_file="src/main.py",
            all_file_paths=all_files,
            external_packages=external_packages,
        )
        assert result["type"] == "external"
        assert result["package"] == "flask"

    def test_resolves_stdlib_import(self):
        """Should identify stdlib imports."""
        all_files = {"src/main.py"}
        external_packages = {"flask"}
        result = _resolve_import(
            module="os.path",
            current_file="src/main.py",
            all_file_paths=all_files,
            external_packages=external_packages,
        )
        assert result["type"] == "stdlib"
        assert result["module"] == "os"

    def test_resolves_relative_import(self):
        """Should resolve relative imports."""
        all_files = {"src/models.py"}
        external_packages = set()
        result = _resolve_import(
            module=".models",
            current_file="src/main.py",
            all_file_paths=all_files,
            external_packages=external_packages,
        )
        assert result["type"] == "internal"
        assert result["target_path"] == "src/models.py"

    def test_handles_unresolved_relative(self):
        """Should handle unresolved relative imports."""
        all_files = {"src/main.py"}
        external_packages = set()
        result = _resolve_import(
            module=".nonexistent",
            current_file="src/main.py",
            all_file_paths=all_files,
            external_packages=external_packages,
        )
        assert result["type"] == "unknown"
        assert "relative_not_found" in result.get("reason", "")

    def test_defaults_to_external_for_unknown(self):
        """Should default to external for unknown packages."""
        all_files = {"src/main.py"}
        external_packages = set()
        result = _resolve_import(
            module="unknown_package",
            current_file="src/main.py",
            all_file_paths=all_files,
            external_packages=external_packages,
        )
        assert result["type"] == "external"
        assert result["package"] == "unknown_package"


# =============================================================================
# _build_method_lookup Tests
# =============================================================================


class TestBuildMethodLookup:
    """Tests for _build_method_lookup function."""

    def test_builds_lookup_from_methods(self):
        """Should build lookup dict from methods."""
        mock_method1 = MagicMock()
        mock_method1.name = "func_a"
        mock_method1.class_name = None
        mock_method1.line_start = 1
        mock_method1.line_end = 10

        mock_method2 = MagicMock()
        mock_method2.name = "method_b"
        mock_method2.class_name = "MyClass"
        mock_method2.line_start = 20
        mock_method2.line_end = 30

        methods = [mock_method1, mock_method2]

        result = _build_method_lookup(methods)

        # Keys are just method names, not "ClassName.method"
        assert "func_a" in result
        assert "method_b" in result

    def test_handles_empty_list(self):
        """Should return empty dict for empty list."""
        result = _build_method_lookup([])
        assert not result  # Empty dict is falsey

    def test_handles_duplicate_names(self):
        """Should append duplicate names to list."""
        mock_method1 = MagicMock()
        mock_method1.name = "func"
        mock_method1.class_name = None
        mock_method1.line_start = 1
        mock_method1.line_end = 10

        mock_method2 = MagicMock()
        mock_method2.name = "func"
        mock_method2.class_name = "MyClass"
        mock_method2.line_start = 20
        mock_method2.line_end = 30

        methods = [mock_method1, mock_method2]

        result = _build_method_lookup(methods)

        assert "func" in result
        # Both methods should be in the list
        assert len(result["func"]) == 2


# =============================================================================
# _extract_type_names Tests
# =============================================================================


class TestExtractTypeNames:
    """Tests for _extract_type_names function."""

    def test_extracts_simple_type(self):
        """Should extract simple type name."""
        result = _extract_type_names("MyClass")
        assert "MyClass" in result

    def test_extracts_from_generic(self):
        """Should extract type from generic annotation."""
        result = _extract_type_names("List[MyClass]")
        assert "MyClass" in result
        assert "List" not in result  # List is a container

    def test_extracts_multiple_types(self):
        """Should extract multiple types from union."""
        result = _extract_type_names("Union[TypeA, TypeB]")
        assert "TypeA" in result
        assert "TypeB" in result

    def test_handles_optional(self):
        """Should handle Optional type."""
        result = _extract_type_names("Optional[MyType]")
        assert "MyType" in result
        assert "Optional" not in result

    def test_removes_quotes(self):
        """Should remove quotes from forward references."""
        result = _extract_type_names('"MyForwardRef"')
        assert "MyForwardRef" in result

    def test_ignores_builtin_containers(self):
        """Should ignore builtin container types."""
        result = _extract_type_names("Dict[str, List[int]]")
        # Only PascalCase names that aren't containers
        assert "Dict" not in result
        assert "List" not in result

    def test_extracts_from_nested_generic(self):
        """Should extract from deeply nested generics."""
        result = _extract_type_names("List[Dict[str, UserModel]]")
        assert "UserModel" in result

    def test_returns_empty_for_primitives(self):
        """Should return empty set for primitive types only."""
        result = _extract_type_names("str")
        assert len(result) == 0

    def test_handles_callable(self):
        """Should handle Callable type annotations."""
        result = _extract_type_names("Callable[[InputType], OutputType]")
        assert "InputType" in result
        assert "OutputType" in result


# =============================================================================
# Python Stdlib Tests
# =============================================================================


class TestPythonStdlib:
    """Tests for PYTHON_STDLIB constant."""

    def test_contains_common_modules(self):
        """Should contain common stdlib modules."""
        assert "os" in PYTHON_STDLIB
        assert "sys" in PYTHON_STDLIB
        assert "json" in PYTHON_STDLIB
        assert "re" in PYTHON_STDLIB
        assert "pathlib" in PYTHON_STDLIB
        assert "typing" in PYTHON_STDLIB
        assert "collections" in PYTHON_STDLIB
        assert "datetime" in PYTHON_STDLIB

    def test_does_not_contain_external_packages(self):
        """Should not contain external packages."""
        assert "flask" not in PYTHON_STDLIB
        assert "django" not in PYTHON_STDLIB
        assert "numpy" not in PYTHON_STDLIB
        assert "pandas" not in PYTHON_STDLIB


# =============================================================================
# extract_edges_from_file Tests
# =============================================================================


class TestExtractEdgesFromFile:
    """Tests for extract_edges_from_file function."""

    def test_returns_result_dict(self):
        """Should return a result dictionary."""
        mock_tsm = MagicMock()
        mock_tsm.get_handler.return_value = None

        with patch(
            "deriva.modules.extraction.edges.TreeSitterManager", return_value=mock_tsm
        ):
            result = extract_edges_from_file(
                file_path="test.py",
                file_content="print('hello')",
                repo_name="test-repo",
                external_packages=set(),
                all_file_paths={"test.py"},
            )

        assert "success" in result
        assert "data" in result
        assert "errors" in result

    def test_handles_unsupported_language(self):
        """Should handle unsupported file types gracefully."""
        mock_tsm = MagicMock()
        mock_tsm.get_handler.return_value = None

        with patch(
            "deriva.modules.extraction.edges.TreeSitterManager", return_value=mock_tsm
        ):
            result = extract_edges_from_file(
                file_path="test.txt",  # Unsupported
                file_content="some text",
                repo_name="test-repo",
                external_packages=set(),
                all_file_paths={"test.txt"},
            )

        assert result["success"] is True
        assert len(result["data"]["edges"]) == 0


# =============================================================================
# extract_edges_batch Tests
# =============================================================================


class TestExtractEdgesBatch:
    """Tests for extract_edges_batch function."""

    def test_returns_result_dict(self):
        """Should return a result dictionary."""
        result = extract_edges_batch(
            files=[],
            repo_name="test-repo",
            repo_path="/repo",
        )

        assert "success" in result
        assert "data" in result
        assert "errors" in result
        assert "stats" in result

    def test_handles_empty_files(self):
        """Should handle empty file list."""
        result = extract_edges_batch(
            files=[],
            repo_name="test-repo",
            repo_path="/repo",
        )

        assert result["success"] is True
        assert len(result["data"]["edges"]) == 0

    def test_stats_includes_required_fields(self):
        """Should include required stats fields."""
        result = extract_edges_batch(
            files=[],
            repo_name="test-repo",
            repo_path="/repo",
        )

        stats = result["stats"]
        assert "files_processed" in stats
        assert "total_edges" in stats
        assert "imports" in stats
        assert "calls" in stats
