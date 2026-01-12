"""Tests for adapters.treesitter.manager module."""

from __future__ import annotations

from deriva.adapters.treesitter import TreeSitterManager


class TestTreeSitterManagerInit:
    """Tests for TreeSitterManager initialization."""

    def test_init_creates_empty_parsers_cache(self):
        """Should initialize with empty parsers cache."""
        manager = TreeSitterManager()
        assert manager._parsers == {}

    def test_supported_languages_constant(self):
        """Should have supported languages defined."""
        assert "python" in TreeSitterManager.SUPPORTED_LANGUAGES
        assert "javascript" in TreeSitterManager.SUPPORTED_LANGUAGES
        assert "java" in TreeSitterManager.SUPPORTED_LANGUAGES
        assert "csharp" in TreeSitterManager.SUPPORTED_LANGUAGES

    def test_extension_map_constant(self):
        """Should have extension map defined."""
        assert TreeSitterManager.EXTENSION_MAP[".py"] == "python"
        assert TreeSitterManager.EXTENSION_MAP[".js"] == "javascript"
        assert TreeSitterManager.EXTENSION_MAP[".java"] == "java"
        assert TreeSitterManager.EXTENSION_MAP[".cs"] == "csharp"


class TestDetectLanguage:
    """Tests for detect_language class method."""

    def test_detects_python(self):
        """Should detect Python files."""
        assert TreeSitterManager.detect_language("test.py") == "python"
        assert TreeSitterManager.detect_language("test.pyw") == "python"
        assert TreeSitterManager.detect_language("test.pyi") == "python"

    def test_detects_javascript(self):
        """Should detect JavaScript files."""
        assert TreeSitterManager.detect_language("test.js") == "javascript"
        assert TreeSitterManager.detect_language("test.mjs") == "javascript"
        assert TreeSitterManager.detect_language("test.jsx") == "javascript"

    def test_detects_typescript(self):
        """Should detect TypeScript files."""
        assert TreeSitterManager.detect_language("test.ts") == "typescript"
        assert TreeSitterManager.detect_language("test.tsx") == "typescript"

    def test_detects_java(self):
        """Should detect Java files."""
        assert TreeSitterManager.detect_language("Test.java") == "java"

    def test_detects_csharp(self):
        """Should detect C# files."""
        assert TreeSitterManager.detect_language("Test.cs") == "csharp"

    def test_returns_none_for_unknown(self):
        """Should return None for unknown extensions."""
        assert TreeSitterManager.detect_language("test.unknown") is None
        assert TreeSitterManager.detect_language("test.rb") is None
        assert TreeSitterManager.detect_language("test") is None

    def test_case_insensitive(self):
        """Should handle case insensitivity."""
        assert TreeSitterManager.detect_language("test.PY") == "python"
        assert TreeSitterManager.detect_language("test.Py") == "python"


class TestSupportsLanguage:
    """Tests for supports_language class method."""

    def test_supports_python(self):
        """Should support Python."""
        assert TreeSitterManager.supports_language("python") is True
        assert TreeSitterManager.supports_language("Python") is True
        assert TreeSitterManager.supports_language("PYTHON") is True

    def test_supports_javascript(self):
        """Should support JavaScript."""
        assert TreeSitterManager.supports_language("javascript") is True

    def test_supports_java(self):
        """Should support Java."""
        assert TreeSitterManager.supports_language("java") is True

    def test_supports_csharp(self):
        """Should support C#."""
        assert TreeSitterManager.supports_language("csharp") is True

    def test_does_not_support_unknown(self):
        """Should not support unknown languages."""
        assert TreeSitterManager.supports_language("ruby") is False
        assert TreeSitterManager.supports_language("go") is False


class TestGetSupportedLanguages:
    """Tests for get_supported_languages class method."""

    def test_returns_list(self):
        """Should return a list."""
        languages = TreeSitterManager.get_supported_languages()
        assert isinstance(languages, list)

    def test_includes_known_languages(self):
        """Should include known languages."""
        languages = TreeSitterManager.get_supported_languages()
        assert "python" in languages


class TestExtractTypes:
    """Tests for extract_types method."""

    def test_extracts_python_class(self):
        """Should extract Python class definitions."""
        manager = TreeSitterManager()
        source = '''
class MyClass:
    """A sample class."""
    pass
'''
        types = manager.extract_types(source, language="python")

        assert len(types) == 1
        assert types[0].name == "MyClass"
        assert types[0].kind == "class"

    def test_extracts_python_function(self):
        """Should extract Python function definitions."""
        manager = TreeSitterManager()
        source = '''
def my_function():
    """A sample function."""
    pass
'''
        types = manager.extract_types(source, language="python")

        assert len(types) == 1
        assert types[0].name == "my_function"
        assert types[0].kind == "function"

    def test_extracts_class_with_bases(self):
        """Should extract class inheritance."""
        manager = TreeSitterManager()
        source = """
class Child(Parent):
    pass
"""
        types = manager.extract_types(source, language="python")

        assert len(types) == 1
        assert types[0].name == "Child"
        assert "Parent" in types[0].bases

    def test_returns_empty_for_unknown_language(self):
        """Should return empty list for unknown language."""
        manager = TreeSitterManager()
        source = "class MyClass: pass"

        types = manager.extract_types(source, language="unknown")
        assert types == []

    def test_returns_empty_for_no_language(self):
        """Should return empty list when language cannot be determined."""
        manager = TreeSitterManager()
        source = "class MyClass: pass"

        types = manager.extract_types(source)
        assert types == []

    def test_detects_language_from_path(self):
        """Should detect language from file path."""
        manager = TreeSitterManager()
        source = """
class MyClass:
    pass
"""
        types = manager.extract_types(source, file_path="test.py")

        assert len(types) == 1
        assert types[0].name == "MyClass"


class TestExtractMethods:
    """Tests for extract_methods method."""

    def test_extracts_class_methods(self):
        """Should extract methods from classes."""
        manager = TreeSitterManager()
        source = """
class MyClass:
    def my_method(self):
        pass
"""
        methods = manager.extract_methods(source, language="python")

        assert len(methods) >= 1
        method_names = [m.name for m in methods]
        assert "my_method" in method_names

    def test_extracts_standalone_functions(self):
        """Should extract standalone functions."""
        manager = TreeSitterManager()
        source = """
def standalone():
    pass
"""
        methods = manager.extract_methods(source, language="python")

        assert len(methods) == 1
        assert methods[0].name == "standalone"
        assert methods[0].class_name is None

    def test_returns_empty_for_unknown_language(self):
        """Should return empty list for unknown language."""
        manager = TreeSitterManager()
        source = "def test(): pass"

        methods = manager.extract_methods(source, language="unknown")
        assert methods == []


class TestExtractImports:
    """Tests for extract_imports method."""

    def test_extracts_simple_import(self):
        """Should extract simple import statements."""
        manager = TreeSitterManager()
        source = """
import os
"""
        imports = manager.extract_imports(source, language="python")

        assert len(imports) == 1
        assert imports[0].module == "os"

    def test_extracts_from_import(self):
        """Should extract from import statements."""
        manager = TreeSitterManager()
        source = """
from pathlib import Path
"""
        imports = manager.extract_imports(source, language="python")

        assert len(imports) == 1
        assert imports[0].module == "pathlib"
        assert "Path" in imports[0].names
        assert imports[0].is_from_import is True

    def test_extracts_multiple_imports(self):
        """Should extract multiple import statements."""
        manager = TreeSitterManager()
        source = """
import os
import sys
from typing import List, Dict
"""
        imports = manager.extract_imports(source, language="python")

        assert len(imports) == 3
        modules = [i.module for i in imports]
        assert "os" in modules
        assert "sys" in modules
        assert "typing" in modules

    def test_returns_empty_for_unknown_language(self):
        """Should return empty list for unknown language."""
        manager = TreeSitterManager()
        source = "import os"

        imports = manager.extract_imports(source, language="unknown")
        assert imports == []


class TestExtractAll:
    """Tests for extract_all method."""

    def test_extracts_all_elements(self):
        """Should extract types, methods, and imports."""
        manager = TreeSitterManager()
        source = """
import os

class MyClass:
    def my_method(self):
        pass
"""
        result = manager.extract_all(source, language="python")

        assert "types" in result
        assert "methods" in result
        assert "imports" in result
        assert len(result["types"]) >= 1
        assert len(result["methods"]) >= 1
        assert len(result["imports"]) == 1

    def test_returns_empty_dicts_for_unknown_language(self):
        """Should return empty lists for unknown language."""
        manager = TreeSitterManager()
        source = "code"

        result = manager.extract_all(source, language="unknown")

        assert result["types"] == []
        assert result["methods"] == []
        assert result["imports"] == []


class TestResolveLanguage:
    """Tests for _resolve_language private method."""

    def test_explicit_language_takes_precedence(self):
        """Should use explicit language over file path."""
        manager = TreeSitterManager()

        # Even though file is .js, explicit language should be used
        lang = manager._resolve_language("test.js", "python")
        assert lang == "python"

    def test_uses_file_path_when_no_explicit(self):
        """Should use file path when no explicit language."""
        manager = TreeSitterManager()

        lang = manager._resolve_language("test.py", None)
        assert lang == "python"

    def test_returns_none_when_no_info(self):
        """Should return None when no language info available."""
        manager = TreeSitterManager()

        lang = manager._resolve_language(None, None)
        assert lang is None

    def test_typescript_maps_to_javascript(self):
        """Should map TypeScript to JavaScript."""
        manager = TreeSitterManager()

        lang = manager._resolve_language(None, "typescript")
        assert lang == "javascript"

        lang = manager._resolve_language("test.ts", None)
        assert lang == "javascript"


class TestParserCaching:
    """Tests for parser caching behavior."""

    def test_parser_is_cached(self):
        """Should cache parsers for reuse."""
        manager = TreeSitterManager()
        source = "class Test: pass"

        # First extraction creates parser
        manager.extract_types(source, language="python")
        assert "python" in manager._parsers

        # Second extraction reuses parser
        manager.extract_types(source, language="python")
        assert len(manager._parsers) == 1
