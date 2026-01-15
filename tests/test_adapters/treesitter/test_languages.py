"""Consolidated tests for all language extractors (Java, JavaScript, C#, Python).

This file replaces the individual test_java.py, test_javascript.py, test_csharp.py,
and test_python.py files with parameterized tests that reduce duplication.
"""

from __future__ import annotations

import pytest

from deriva.adapters.treesitter import TreeSitterManager

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def manager():
    """Provide a TreeSitterManager instance for all tests."""
    return TreeSitterManager()


# =============================================================================
# LANGUAGE-SPECIFIC CODE SAMPLES
# =============================================================================

SAMPLES = {
    "java": {
        "class": ("public class UserService { private String name; }", "UserService", "class"),
        "class_inheritance": ("public class Admin extends User { private int level; }", "Admin"),
        "interface": ("public interface Repository<T> { T findById(Long id); }", "Repository", "interface"),
        "enum": ("public enum Status { PENDING, ACTIVE, COMPLETED }", "Status"),
        "abstract_class": ("public abstract class BaseEntity { public abstract void validate(); }", "BaseEntity"),
        "inner_class": ("public class Outer { public class Inner { private int value; } }", "Outer"),
        "method_public": ("public class Calc { public int add(int a, int b) { return a + b; } }", "add"),
        "method_private": ("public class Svc { private void helper() { } }", "helper"),
        "method_static": ("public class Utils { public static String format(String s) { return s; } }", "format"),
        "constructor": ("public class Person { public Person(String name) { } }", "Person", "Person"),
        "overloaded": ("public class P { public void print(String s) {} public void print(int i) {} }", "print", 2),
        "import_single": ("import java.util.List;", 1),
        "import_wildcard": ("import java.util.*;", 1),
        "import_multiple": ("import java.util.List;\nimport java.util.Map;\nimport java.io.IOException;", 3),
        "annotated_class": ('@Entity\n@Table(name = "users")\npublic class User { }', "User"),
        "record": ("public record User(String name, int age) { }", "User"),
    },
    "javascript": {
        "class": ("class UserService { constructor() { this.users = []; } }", "UserService", "class"),
        "class_inheritance": ("class Admin extends User { constructor() { super(); } }", "Admin"),
        "function": ("function processData(data) { return data; }", "processData", "function"),
        "async_function": ("async function fetchUser(id) { return await fetch(id); }", "fetchUser", True),
        "arrow_function": ("const multiply = (x, y) => { return x * y; };", "multiply", "function"),
        "async_arrow": ("const fetchData = async (url) => { return await fetch(url); };", "fetchData", True),
        "exported_class": ("export class Service { run() { return true; } }", "Service"),
        "method_class": ("class Calc { add(a, b) { return a + b; } subtract(a, b) { return a - b; } }", ["add", "subtract"]),
        "constructor": ("class Service { constructor(config) { this.config = config; } }", "constructor"),
        "async_method": ("class Api { async get(url) { return await fetch(url); } }", "get", True),
        "static_method": ("class MathUtils { static square(x) { return x * x; } }", "square"),
        "import_default": ("import React from 'react';", "react"),
        "import_named": ("import { useState, useEffect } from 'react';", 1),
        "import_namespace": ("import * as utils from './utils';", 1),
        "import_require": ("const express = require('express');", "express"),
        "jsx_function": ("function Component() { return <div>Hello</div>; }", "Component", "function"),
        "private_fields": ("class Counter { #count = 0; increment() { this.#count++; } }", "increment", "Counter"),
    },
    "csharp": {
        "class": ("public class UserService { private string _name; }", "UserService", "class"),
        "class_inheritance": ("public class Admin : User { public int Level { get; set; } }", "Admin"),
        "interface": ("public interface IRepository<T> { T GetById(int id); }", "IRepository", "interface"),
        "struct": ("public struct Point { public int X; public int Y; }", "Point"),
        "enum": ("public enum Status { Pending, Active, Completed }", "Status"),
        "abstract_class": ("public abstract class BaseEntity { public abstract void Validate(); }", "BaseEntity"),
        "record": ("public record User(string Name, int Age);", "User"),
        "method_public": ("public class Calc { public int Add(int a, int b) { return a + b; } }", "Add"),
        "method_private": ("public class Svc { private void Helper() { } }", "Helper"),
        "method_static": ("public class Utils { public static string Format(string s) { return s; } }", "Format"),
        "method_async": ("public class Api { public async Task<User> GetUserAsync(int id) { return null; } }", "GetUserAsync"),
        "constructor": ("public class Person { public Person(string name) { } public string Name { get; } }", "Person", "Person"),
        "import_using": ("using System;", 1),
        "import_multiple": ("using System;\nusing System.Collections.Generic;\nusing System.Linq;", 3),
        "import_alias": ("using Console = System.Console;", 1),
        "namespace_class": ("namespace MyApp.Services { public class UserService { } }", "UserService"),
        "file_scoped_ns": ("namespace MyApp.Services;\npublic class UserService { }", "UserService"),
        "attributes": ('[Serializable]\n[Table("users")]\npublic class User { }', "User"),
        "partial_class": ("public partial class User { public string Name { get; set; } }", "User"),
    },
    "python": {
        "class": ("class User: pass", "User", "class"),
        "class_inheritance": ("class Admin(User, PermissionMixin): pass", "Admin", {"User", "PermissionMixin"}),
        "class_docstring": ('class Service:\n    """A service class."""\n    pass', "Service", "service"),
        "decorated_class": ("@dataclass\nclass Config:\n    name: str", "Config"),
        "function": ("def process_data(items): return items", "process_data", "function"),
        "async_function": ("async def fetch_user(user_id): pass", "fetch_user", True),
        "type_alias": ("type UserId = int\ntype Callback = Callable[[int], str]", 2),
        "multiple_classes": ("class First: pass\nclass Second: pass\nclass Third: pass", {"First", "Second", "Third"}),
        "method_class": ("class Calc:\n    def add(self, a, b): return a + b\n    def sub(self, a, b): return a - b", ["add", "sub"], "Calc"),
        "standalone_fn": ("def helper_function(x): return x * 2", "helper_function", None),
        "async_method": ("class Svc:\n    async def fetch(self, url): return url", "fetch", True),
        "staticmethod": ("class Utils:\n    @staticmethod\n    def format_date(date): return date", "format_date", "static"),
        "classmethod": ("class Factory:\n    @classmethod\n    def create(cls): return cls()", "create", "classmethod"),
        "property": ("class Person:\n    @property\n    def full_name(self): return self.name", "full_name", "property"),
        "import_simple": ("import os", "os", False),
        "import_alias": ("import numpy as np", "numpy", "np"),
        "import_from": ("from pathlib import Path", "pathlib", ["Path"]),
        "import_multiple": ("import os\nimport sys\nimport json\nfrom pathlib import Path\nfrom typing import Any", 5),
        "nested_class": ("class Outer:\n    class Inner:\n        def method(self): pass", "Outer"),
        "unicode": ("class Données:\n    def traiter(self): pass", "Données"),
    },
}


# =============================================================================
# TYPE EXTRACTION TESTS
# =============================================================================


class TestTypeExtraction:
    """Tests for type extraction across all languages."""

    @pytest.mark.parametrize("language", ["java", "javascript", "csharp", "python"])
    def test_extracts_class(self, manager, language):
        """Should extract basic class definition."""
        source, name, kind = SAMPLES[language]["class"]
        types = manager.extract_types(source, language=language)

        assert len(types) >= 1
        target = next((t for t in types if t.name == name), None)
        assert target is not None, f"Expected to find {name} in {[t.name for t in types]}"
        assert target.kind == kind

    @pytest.mark.parametrize("language", ["java", "javascript", "csharp", "python"])
    def test_extracts_class_with_inheritance(self, manager, language):
        """Should extract class with inheritance/extends."""
        sample = SAMPLES[language].get("class_inheritance")
        if not sample:
            pytest.skip(f"No inheritance sample for {language}")

        source, name = sample[0], sample[1]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None, f"Expected to find {name}"
        # Python has bases attribute to check
        if language == "python" and len(sample) > 2:
            assert set(target.bases) == sample[2]

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("java", "interface"),
            ("csharp", "interface"),
        ],
    )
    def test_extracts_interface(self, manager, language, sample_key):
        """Should extract interface definition."""
        source, name, kind = SAMPLES[language][sample_key]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None
        assert target.kind == kind

    @pytest.mark.parametrize("language", ["java", "csharp"])
    def test_extracts_enum(self, manager, language):
        """Should extract enum definition."""
        source, name = SAMPLES[language]["enum"][:2]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    @pytest.mark.parametrize("language", ["java", "csharp"])
    def test_extracts_abstract_class(self, manager, language):
        """Should extract abstract class."""
        source, name = SAMPLES[language]["abstract_class"][:2]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    @pytest.mark.parametrize("language", ["java", "csharp"])
    def test_extracts_record(self, manager, language):
        """Should extract record type."""
        source, name = SAMPLES[language]["record"][:2]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    def test_csharp_extracts_struct(self, manager):
        """Should extract C# struct."""
        source, name = SAMPLES["csharp"]["struct"][:2]
        types = manager.extract_types(source, language="csharp")

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("javascript", "function"),
            ("javascript", "async_function"),
            ("javascript", "arrow_function"),
            ("python", "function"),
            ("python", "async_function"),
        ],
    )
    def test_extracts_function_as_type(self, manager, language, sample_key):
        """Should extract top-level functions as types."""
        sample = SAMPLES[language][sample_key]
        source, name = sample[0], sample[1]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None
        # Check async flag if provided
        if len(sample) > 2 and sample[2] is True:
            assert target.is_async is True

    def test_python_extracts_type_alias(self, manager):
        """Should extract Python type aliases."""
        source, count = SAMPLES["python"]["type_alias"][:2]
        types = manager.extract_types(source, language="python")
        aliases = [t for t in types if t.kind == "type_alias"]

        assert len(aliases) == count

    def test_python_extracts_multiple_classes(self, manager):
        """Should extract multiple class definitions."""
        source, expected_names = SAMPLES["python"]["multiple_classes"][:2]
        types = manager.extract_types(source, language="python")

        assert {t.name for t in types} == expected_names


# =============================================================================
# METHOD EXTRACTION TESTS
# =============================================================================


class TestMethodExtraction:
    """Tests for method extraction across all languages."""

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("java", "method_public"),
            ("javascript", "method_class"),
            ("csharp", "method_public"),
            ("python", "method_class"),
        ],
    )
    def test_extracts_class_methods(self, manager, language, sample_key):
        """Should extract methods from a class."""
        sample = SAMPLES[language][sample_key]
        source = sample[0]
        expected = sample[1]  # Either a name or list of names
        methods = manager.extract_methods(source, language=language)

        if isinstance(expected, list):
            names = {m.name for m in methods}
            assert all(e in names for e in expected), f"Expected {expected} in {names}"
        else:
            target = next((m for m in methods if m.name == expected), None)
            assert target is not None

    @pytest.mark.parametrize("language", ["java", "csharp"])
    def test_extracts_private_method(self, manager, language):
        """Should extract private methods."""
        source, name = SAMPLES[language]["method_private"][:2]
        methods = manager.extract_methods(source, language=language)

        target = next((m for m in methods if m.name == name), None)
        assert target is not None

    @pytest.mark.parametrize("language", ["java", "javascript", "csharp"])
    def test_extracts_static_method(self, manager, language):
        """Should extract static methods."""
        sample_key = "method_static" if language != "javascript" else "static_method"
        source, name = SAMPLES[language][sample_key][:2]
        methods = manager.extract_methods(source, language=language)

        target = next((m for m in methods if m.name == name), None)
        assert target is not None

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("java", "constructor"),
            ("javascript", "constructor"),
            ("csharp", "constructor"),
        ],
    )
    def test_extracts_constructor(self, manager, language, sample_key):
        """Should extract constructor."""
        sample = SAMPLES[language][sample_key]
        source, name = sample[0], sample[1]
        methods = manager.extract_methods(source, language=language)

        target = next((m for m in methods if m.name == name), None)
        assert target is not None
        # Check class_name if provided
        if len(sample) > 2:
            assert target.class_name == sample[2]

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("javascript", "async_method"),
            ("csharp", "method_async"),
            ("python", "async_method"),
        ],
    )
    def test_extracts_async_method(self, manager, language, sample_key):
        """Should detect async methods."""
        sample = SAMPLES[language][sample_key]
        source, name = sample[0], sample[1]
        methods = manager.extract_methods(source, language=language)

        target = next((m for m in methods if m.name == name), None)
        assert target is not None
        if language == "python" or language == "javascript":
            assert target.is_async is True

    def test_java_extracts_overloaded_methods(self, manager):
        """Should extract overloaded methods in Java."""
        source, name, count = SAMPLES["java"]["overloaded"]
        methods = manager.extract_methods(source, language="java")

        matching = [m for m in methods if m.name == name]
        assert len(matching) == count

    def test_python_extracts_standalone_function(self, manager):
        """Should extract standalone function with no class."""
        source, name, class_name = SAMPLES["python"]["standalone_fn"]
        methods = manager.extract_methods(source, language="python")

        target = next((m for m in methods if m.name == name), None)
        assert target is not None
        assert target.class_name is class_name

    @pytest.mark.parametrize(
        "sample_key,expected_attr",
        [
            ("staticmethod", "is_static"),
            ("classmethod", "is_classmethod"),
            ("property", "is_property"),
        ],
    )
    def test_python_extracts_decorated_methods(self, manager, sample_key, expected_attr):
        """Should detect Python method decorators."""
        source, name = SAMPLES["python"][sample_key][:2]
        methods = manager.extract_methods(source, language="python")

        target = next((m for m in methods if m.name == name), None)
        assert target is not None
        assert getattr(target, expected_attr) is True


# =============================================================================
# IMPORT EXTRACTION TESTS
# =============================================================================


class TestImportExtraction:
    """Tests for import extraction across all languages."""

    @pytest.mark.parametrize(
        "language,sample_key,expected_count",
        [
            ("java", "import_single", 1),
            ("java", "import_wildcard", 1),
            ("java", "import_multiple", 3),
            ("javascript", "import_named", 1),
            ("javascript", "import_namespace", 1),
            ("csharp", "import_using", 1),
            ("csharp", "import_multiple", 3),
            ("csharp", "import_alias", 1),
            ("python", "import_multiple", 5),
        ],
    )
    def test_extracts_imports_count(self, manager, language, sample_key, expected_count):
        """Should extract correct number of imports."""
        source = SAMPLES[language][sample_key][0]
        imports = manager.extract_imports(source, language=language)

        assert len(imports) >= expected_count

    def test_python_extracts_simple_import(self, manager):
        """Should extract simple Python import."""
        source, module, is_from = SAMPLES["python"]["import_simple"]
        imports = manager.extract_imports(source, language="python")

        assert len(imports) == 1
        assert imports[0].module == module
        assert imports[0].is_from_import is is_from

    def test_python_extracts_import_with_alias(self, manager):
        """Should extract Python import with alias."""
        source, module, alias = SAMPLES["python"]["import_alias"]
        imports = manager.extract_imports(source, language="python")

        assert imports[0].module == module
        assert imports[0].alias == alias

    def test_python_extracts_from_import(self, manager):
        """Should extract Python from-import."""
        source, module, names = SAMPLES["python"]["import_from"]
        imports = manager.extract_imports(source, language="python")

        assert imports[0].module == module
        assert all(n in imports[0].names for n in names)
        assert imports[0].is_from_import is True

    def test_javascript_extracts_default_import(self, manager):
        """Should extract JavaScript default import."""
        source, module = SAMPLES["javascript"]["import_default"][:2]
        imports = manager.extract_imports(source, language="javascript")

        assert len(imports) >= 1
        assert any(module in i.module for i in imports)

    def test_javascript_extracts_require(self, manager):
        """Should extract CommonJS require statements."""
        source, module = SAMPLES["javascript"]["import_require"][:2]
        imports = manager.extract_imports(source, language="javascript")

        modules = [i.module for i in imports]
        assert any(module in m for m in modules)


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases across all languages."""

    @pytest.mark.parametrize("language", ["java", "javascript", "csharp", "python"])
    def test_handles_empty_file(self, manager, language):
        """Should handle empty file without error."""
        types = manager.extract_types("", language=language)
        assert types == []

        methods = manager.extract_methods("", language=language)
        assert methods == []

        imports = manager.extract_imports("", language=language)
        assert imports == []

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("java", "annotated_class"),
            ("csharp", "attributes"),
            ("python", "decorated_class"),
        ],
    )
    def test_handles_decorated_annotated_classes(self, manager, language, sample_key):
        """Should handle classes with decorators/annotations/attributes."""
        source, name = SAMPLES[language][sample_key][:2]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    @pytest.mark.parametrize(
        "language,sample_key",
        [
            ("csharp", "namespace_class"),
            ("csharp", "file_scoped_ns"),
        ],
    )
    def test_handles_namespaces(self, manager, language, sample_key):
        """Should handle namespace declarations."""
        source, name = SAMPLES[language][sample_key][:2]
        types = manager.extract_types(source, language=language)

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    def test_java_handles_inner_class(self, manager):
        """Should handle Java inner classes."""
        source, name = SAMPLES["java"]["inner_class"][:2]
        types = manager.extract_types(source, language="java")

        # Should extract at least the outer class
        assert len(types) >= 1

    def test_javascript_handles_jsx(self, manager):
        """Should handle JSX syntax."""
        source, name, kind = SAMPLES["javascript"]["jsx_function"]
        types = manager.extract_types(source, language="javascript")

        target = next((t for t in types if t.name == name), None)
        assert target is not None
        assert target.kind == kind

    def test_javascript_handles_private_fields(self, manager):
        """Should handle private class fields."""
        source, name, class_name = SAMPLES["javascript"]["private_fields"]
        methods = manager.extract_methods(source, language="javascript")

        target = next((m for m in methods if m.name == name), None)
        assert target is not None
        assert target.class_name == class_name

    def test_csharp_handles_partial_class(self, manager):
        """Should handle partial classes."""
        source, name = SAMPLES["csharp"]["partial_class"][:2]
        types = manager.extract_types(source, language="csharp")

        target = next((t for t in types if t.name == name), None)
        assert target is not None

    def test_python_handles_nested_class(self, manager):
        """Should handle nested class definitions."""
        source, name = SAMPLES["python"]["nested_class"][:2]
        types = manager.extract_types(source, language="python")

        assert name in {t.name for t in types}

    def test_python_handles_unicode_identifiers(self, manager):
        """Should handle unicode in identifiers."""
        source, name = SAMPLES["python"]["unicode"][:2]
        types = manager.extract_types(source, language="python")

        assert len(types) >= 1

    def test_python_handles_syntax_errors(self, manager):
        """Should handle syntax errors gracefully."""
        source = "class Incomplete {\n    def broken"
        types = manager.extract_types(source, language="python")
        assert isinstance(types, list)

    def test_python_handles_comments_only(self, manager):
        """Should handle file with only comments."""
        types = manager.extract_types("# Comment\n# Another", language="python")
        assert types == []
