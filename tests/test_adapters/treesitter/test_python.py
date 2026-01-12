"""Tests for Python language extractor."""

from __future__ import annotations

import pytest

from deriva.adapters.treesitter import TreeSitterManager


@pytest.fixture
def manager():
    """Provide a TreeSitterManager for tests."""
    return TreeSitterManager()


class TestPythonTypes:
    """Tests for Python type extraction."""

    def test_extracts_simple_class(self, manager):
        """Should extract basic class definition."""
        types = manager.extract_types("class User: pass", language="python")

        assert len(types) == 1
        assert types[0].name == "User"
        assert types[0].kind == "class"

    def test_extracts_class_with_inheritance(self, manager):
        """Should extract class with base classes."""
        types = manager.extract_types("class Admin(User, PermissionMixin): pass", language="python")

        assert len(types) == 1
        assert types[0].name == "Admin"
        assert set(types[0].bases) == {"User", "PermissionMixin"}

    def test_extracts_class_docstring(self, manager):
        """Should extract class docstring."""
        source = 'class Service:\n    """A service class."""\n    pass'
        types = manager.extract_types(source, language="python")

        assert types[0].docstring is not None
        assert "service" in types[0].docstring.lower()

    def test_extracts_decorated_class(self, manager):
        """Should extract decorated class."""
        source = "@dataclass\nclass Config:\n    name: str"
        types = manager.extract_types(source, language="python")

        assert len(types) == 1
        assert types[0].name == "Config"

    def test_extracts_top_level_function(self, manager):
        """Should extract top-level function as type."""
        types = manager.extract_types("def process_data(items): return items", language="python")

        assert len(types) == 1
        assert types[0].name == "process_data"
        assert types[0].kind == "function"

    def test_extracts_async_function(self, manager):
        """Should detect async functions."""
        types = manager.extract_types("async def fetch_user(user_id): pass", language="python")

        assert len(types) == 1
        assert types[0].name == "fetch_user"
        assert types[0].is_async is True

    def test_extracts_decorated_function(self, manager):
        """Should extract decorated top-level function."""
        types = manager.extract_types("@cache\ndef expensive_operation(): return 1", language="python")

        assert len(types) == 1
        assert types[0].name == "expensive_operation"

    def test_extracts_type_alias(self, manager):
        """Should extract Python 3.12+ type aliases."""
        source = "type UserId = int\ntype Callback = Callable[[int], str]"
        types = manager.extract_types(source, language="python")
        type_aliases = [t for t in types if t.kind == "type_alias"]

        assert len(type_aliases) == 2

    def test_extracts_multiple_classes(self, manager):
        """Should extract multiple class definitions."""
        source = "class First: pass\nclass Second: pass\nclass Third: pass"
        types = manager.extract_types(source, language="python")

        assert {t.name for t in types} == {"First", "Second", "Third"}


class TestPythonMethods:
    """Tests for Python method extraction."""

    def test_extracts_class_methods(self, manager):
        """Should extract methods from class with correct class_name."""
        source = "class Calculator:\n    def add(self, a, b): return a + b\n    def subtract(self, a, b): return a - b"
        methods = manager.extract_methods(source, language="python")

        assert {m.name for m in methods} == {"add", "subtract"}
        assert all(m.class_name == "Calculator" for m in methods)

    def test_extracts_standalone_function(self, manager):
        """Should extract standalone function with no class."""
        methods = manager.extract_methods("def helper_function(x): return x * 2", language="python")

        assert len(methods) == 1
        assert methods[0].name == "helper_function"
        assert methods[0].class_name is None

    def test_extracts_method_parameters(self, manager):
        """Should extract method parameters."""
        source = "class Service:\n    def process(self, data: list, config: dict = None) -> bool: pass"
        methods = manager.extract_methods(source, language="python")

        assert len(methods) == 1
        assert len(methods[0].parameters) >= 1

    def test_extracts_return_annotation(self, manager):
        """Should extract return type annotation."""
        source = "class Repository:\n    def find_by_id(self, id: int) -> User: pass"
        methods = manager.extract_methods(source, language="python")

        assert methods[0].return_annotation is not None

    def test_extracts_async_method(self, manager):
        """Should detect async methods."""
        source = "class AsyncService:\n    async def fetch(self, url): return url"
        methods = manager.extract_methods(source, language="python")

        assert methods[0].is_async is True

    def test_extracts_staticmethod(self, manager):
        """Should detect @staticmethod decorator."""
        source = "class Utils:\n    @staticmethod\n    def format_date(date): return date"
        methods = manager.extract_methods(source, language="python")

        assert methods[0].is_static is True

    def test_extracts_classmethod(self, manager):
        """Should detect @classmethod decorator."""
        source = "class Factory:\n    @classmethod\n    def create(cls): return cls()"
        methods = manager.extract_methods(source, language="python")

        assert methods[0].is_classmethod is True

    def test_extracts_property(self, manager):
        """Should detect @property decorator."""
        source = "class Person:\n    @property\n    def full_name(self): return self.name"
        methods = manager.extract_methods(source, language="python")

        assert methods[0].is_property is True

    def test_extracts_method_docstring(self, manager):
        """Should extract method docstring."""
        source = 'class Service:\n    def process(self, data):\n        """Process the data."""\n        return data'
        methods = manager.extract_methods(source, language="python")

        assert "Process" in methods[0].docstring

    def test_extracts_decorated_standalone_function(self, manager):
        """Should extract decorated top-level functions as methods."""
        source = "@cache\ndef expensive_computation(data): return data"
        methods = manager.extract_methods(source, language="python")

        assert methods[0].name == "expensive_computation"
        assert methods[0].class_name is None


class TestPythonImports:
    """Tests for Python import extraction."""

    def test_extracts_simple_import(self, manager):
        """Should extract simple import statement."""
        imports = manager.extract_imports("import os", language="python")

        assert len(imports) == 1
        assert imports[0].module == "os"
        assert imports[0].is_from_import is False

    def test_extracts_import_with_alias(self, manager):
        """Should extract import with alias."""
        imports = manager.extract_imports("import numpy as np", language="python")

        assert imports[0].module == "numpy"
        assert imports[0].alias == "np"

    def test_extracts_from_import(self, manager):
        """Should extract from-import statement."""
        imports = manager.extract_imports("from pathlib import Path", language="python")

        assert imports[0].module == "pathlib"
        assert "Path" in imports[0].names
        assert imports[0].is_from_import is True

    def test_extracts_multiple_names_from_import(self, manager):
        """Should extract multiple names from single import."""
        imports = manager.extract_imports("from typing import List, Dict, Optional", language="python")

        assert imports[0].module == "typing"
        assert len(imports[0].names) >= 1

    def test_extracts_star_import(self, manager):
        """Should extract wildcard import."""
        imports = manager.extract_imports("from module import *", language="python")

        assert imports[0].module == "module"
        assert imports[0].is_from_import is True

    def test_extracts_dotted_import(self, manager):
        """Should extract dotted module path."""
        imports = manager.extract_imports("import os.path\nfrom collections.abc import Mapping", language="python")

        assert len(imports) == 2
        modules = {i.module for i in imports}
        assert "os.path" in modules or "collections.abc" in modules

    def test_extracts_relative_import(self, manager):
        """Should extract relative imports."""
        source = "from . import utils\nfrom ..core import base"
        imports = manager.extract_imports(source, language="python")

        assert len(imports) >= 1

    def test_extracts_multiple_import_statements(self, manager):
        """Should extract all import statements."""
        source = "import os\nimport sys\nimport json\nfrom pathlib import Path\nfrom typing import Any"
        imports = manager.extract_imports(source, language="python")

        assert len(imports) == 5


class TestPythonEdgeCases:
    """Tests for edge cases in Python extraction."""

    def test_handles_empty_file(self, manager):
        """Should handle empty file."""
        assert manager.extract_types("", language="python") == []
        assert manager.extract_methods("", language="python") == []
        assert manager.extract_imports("", language="python") == []

    def test_handles_comments_only(self, manager):
        """Should handle file with only comments."""
        assert manager.extract_types("# This is a comment\n# Another comment", language="python") == []

    def test_handles_nested_classes(self, manager):
        """Should handle nested class definitions."""
        source = "class Outer:\n    class Inner:\n        def method(self): pass"
        types = manager.extract_types(source, language="python")

        assert "Outer" in {t.name for t in types}

    def test_handles_unicode_identifiers(self, manager):
        """Should handle unicode in identifiers."""
        source = "class Données:\n    def traiter(self): pass"
        types = manager.extract_types(source, language="python")

        assert len(types) >= 1

    def test_handles_syntax_errors_gracefully(self, manager):
        """Should handle syntax errors without crashing."""
        source = "class Incomplete {\n    def broken"
        types = manager.extract_types(source, language="python")
        assert isinstance(types, list)
