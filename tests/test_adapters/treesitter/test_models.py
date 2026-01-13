"""Tests for adapters.treesitter.models module."""

from __future__ import annotations

from deriva.adapters.treesitter.models import ExtractedImport, ExtractedMethod, ExtractedType


class TestExtractedType:
    """Tests for ExtractedType dataclass."""

    def test_basic_creation(self):
        """Should create with required fields."""
        t = ExtractedType(name="MyClass", kind="class", line_start=1, line_end=10)

        assert t.name == "MyClass"
        assert t.kind == "class"
        assert t.line_start == 1
        assert t.line_end == 10

    def test_default_values(self):
        """Should have sensible defaults."""
        t = ExtractedType(name="MyClass", kind="class", line_start=1, line_end=10)

        assert t.docstring is None
        assert t.bases == []
        assert t.decorators == []
        assert t.is_async is False
        assert t.visibility is None

    def test_with_optional_fields(self):
        """Should accept optional fields."""
        t = ExtractedType(
            name="AsyncHandler",
            kind="class",
            line_start=1,
            line_end=20,
            docstring="A handler class.",
            bases=["BaseHandler", "Mixin"],
            decorators=["dataclass"],
            is_async=True,
            visibility="public",
        )

        assert t.docstring == "A handler class."
        assert t.bases == ["BaseHandler", "Mixin"]
        assert t.decorators == ["dataclass"]
        assert t.is_async is True
        assert t.visibility == "public"

    def test_kind_options(self):
        """Should support various kinds."""
        kinds = ["class", "interface", "enum", "struct", "function", "type_alias", "package"]

        for kind in kinds:
            t = ExtractedType(name="Test", kind=kind, line_start=1, line_end=1)
            assert t.kind == kind


class TestExtractedMethod:
    """Tests for ExtractedMethod dataclass."""

    def test_basic_creation(self):
        """Should create with required fields."""
        m = ExtractedMethod(name="my_method", class_name="MyClass", line_start=5, line_end=10)

        assert m.name == "my_method"
        assert m.class_name == "MyClass"
        assert m.line_start == 5
        assert m.line_end == 10

    def test_default_values(self):
        """Should have sensible defaults."""
        m = ExtractedMethod(name="test", class_name=None, line_start=1, line_end=5)

        assert m.docstring is None
        assert m.parameters == []
        assert m.return_annotation is None
        assert m.decorators == []
        assert m.is_async is False
        assert m.is_static is False
        assert m.is_classmethod is False
        assert m.is_property is False
        assert m.visibility is None

    def test_standalone_function(self):
        """Should allow None class_name for standalone functions."""
        m = ExtractedMethod(name="helper", class_name=None, line_start=1, line_end=5)

        assert m.class_name is None

    def test_with_optional_fields(self):
        """Should accept optional fields."""
        m = ExtractedMethod(
            name="async_getter",
            class_name="Service",
            line_start=10,
            line_end=20,
            docstring="Gets data asynchronously.",
            parameters=[{"name": "self"}, {"name": "id", "type": "int"}],
            return_annotation="Data",
            decorators=["property", "async"],
            is_async=True,
            is_static=False,
            is_classmethod=False,
            is_property=True,
            visibility="private",
        )

        assert m.docstring == "Gets data asynchronously."
        assert len(m.parameters) == 2
        assert m.return_annotation == "Data"
        assert "property" in m.decorators
        assert m.is_async is True
        assert m.is_property is True
        assert m.visibility == "private"


class TestExtractedImport:
    """Tests for ExtractedImport dataclass."""

    def test_basic_creation(self):
        """Should create with required fields."""
        i = ExtractedImport(module="os", names=[])

        assert i.module == "os"
        assert i.names == []

    def test_default_values(self):
        """Should have sensible defaults."""
        i = ExtractedImport(module="sys", names=[])

        assert i.alias is None
        assert i.line == 0
        assert i.is_from_import is False

    def test_simple_import(self):
        """Should represent 'import module'."""
        i = ExtractedImport(module="json", names=[], line=1)

        assert i.module == "json"
        assert i.names == []
        assert i.is_from_import is False

    def test_from_import(self):
        """Should represent 'from module import names'."""
        i = ExtractedImport(
            module="typing",
            names=["List", "Dict", "Optional"],
            line=1,
            is_from_import=True,
        )

        assert i.module == "typing"
        assert "List" in i.names
        assert "Dict" in i.names
        assert "Optional" in i.names
        assert i.is_from_import is True

    def test_import_with_alias(self):
        """Should support aliased imports."""
        i = ExtractedImport(module="numpy", names=[], alias="np", line=1)

        assert i.module == "numpy"
        assert i.alias == "np"
