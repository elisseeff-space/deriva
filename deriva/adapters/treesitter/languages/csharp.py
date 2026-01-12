"""C# language extractor using tree-sitter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tree_sitter

from ..models import ExtractedImport, ExtractedMethod, ExtractedType
from .base import LanguageExtractor
from . import register_extractor


class CSharpExtractor(LanguageExtractor):
    """Extractor for C# source code using tree-sitter-c-sharp."""

    @property
    def language_name(self) -> str:
        return "csharp"

    def get_language(self) -> Any:
        """Return the tree-sitter C# language."""
        import tree_sitter_c_sharp

        return tree_sitter_c_sharp.language()

    def extract_types(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedType]:
        """Extract type definitions (classes, interfaces, structs, enums, records)."""
        types: list[ExtractedType] = []
        root = tree.root_node

        type_nodes = {
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "enum_declaration",
            "record_declaration",
            "record_struct_declaration",
        }

        for node in self.walk_tree(root, type_nodes):
            types.append(self._extract_type(node, source))

        return types

    def extract_methods(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedMethod]:
        """Extract methods, constructors, and properties."""
        methods: list[ExtractedMethod] = []
        root = tree.root_node

        type_nodes = {
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "record_declaration",
            "record_struct_declaration",
        }

        for type_node in self.walk_tree(root, type_nodes):
            class_name = self._get_type_name(type_node, source)
            body = self.find_child_by_field(type_node, "body")
            if not body:
                # Records may have a different structure
                body = self.find_child_by_type(type_node, "declaration_list")
            if body:
                for child in body.children:
                    if child.type == "method_declaration":
                        methods.append(self._extract_method(child, source, class_name))
                    elif child.type == "constructor_declaration":
                        methods.append(
                            self._extract_constructor(child, source, class_name)
                        )
                    elif child.type == "property_declaration":
                        methods.append(
                            self._extract_property(child, source, class_name)
                        )

        return methods

    def extract_imports(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedImport]:
        """Extract using directives."""
        imports: list[ExtractedImport] = []
        root = tree.root_node

        for node in self.walk_tree(root, {"using_directive"}):
            imports.append(self._extract_using(node, source))

        return imports

    # =========================================================================
    # Private extraction helpers
    # =========================================================================

    def _extract_type(self, node: tree_sitter.Node, source: bytes) -> ExtractedType:
        """Extract a type definition."""
        name = self._get_type_name(node, source)
        kind = self._get_type_kind(node)
        bases = self._get_bases(node, source)
        decorators = self._get_attributes(node, source)
        visibility = self._get_visibility(node, source)

        return ExtractedType(
            name=name,
            kind=kind,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=None,
            bases=bases,
            decorators=decorators,
            is_async=False,
            visibility=visibility,
        )

    def _extract_method(
        self, node: tree_sitter.Node, source: bytes, class_name: str
    ) -> ExtractedMethod:
        """Extract a method declaration."""
        name_node = self.find_child_by_field(node, "name")
        name = self.get_node_text(name_node, source) if name_node else ""

        decorators = self._get_attributes(node, source)
        visibility = self._get_visibility(node, source)
        parameters = self._extract_parameters(node, source)

        # Get return type
        return_type_node = self.find_child_by_field(node, "type")
        return_annotation = (
            self.get_node_text(return_type_node, source) if return_type_node else None
        )

        # Check modifiers
        is_static = self._has_modifier(node, source, "static")
        is_async = self._has_modifier(node, source, "async")

        return ExtractedMethod(
            name=name,
            class_name=class_name,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=None,
            parameters=parameters,
            return_annotation=return_annotation,
            decorators=decorators,
            is_async=is_async,
            is_static=is_static,
            is_classmethod=False,
            is_property=False,
            visibility=visibility,
        )

    def _extract_constructor(
        self, node: tree_sitter.Node, source: bytes, class_name: str
    ) -> ExtractedMethod:
        """Extract a constructor declaration."""
        name_node = self.find_child_by_field(node, "name")
        name = self.get_node_text(name_node, source) if name_node else class_name

        decorators = self._get_attributes(node, source)
        visibility = self._get_visibility(node, source)
        parameters = self._extract_parameters(node, source)

        return ExtractedMethod(
            name=name,
            class_name=class_name,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=None,
            parameters=parameters,
            return_annotation=None,
            decorators=decorators,
            is_async=False,
            is_static=False,
            is_classmethod=False,
            is_property=False,
            visibility=visibility,
        )

    def _extract_property(
        self, node: tree_sitter.Node, source: bytes, class_name: str
    ) -> ExtractedMethod:
        """Extract a property declaration."""
        name_node = self.find_child_by_field(node, "name")
        name = self.get_node_text(name_node, source) if name_node else ""

        decorators = self._get_attributes(node, source)
        visibility = self._get_visibility(node, source)

        # Get property type
        type_node = self.find_child_by_field(node, "type")
        return_annotation = self.get_node_text(type_node, source) if type_node else None

        is_static = self._has_modifier(node, source, "static")

        return ExtractedMethod(
            name=name,
            class_name=class_name,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=None,
            parameters=[],
            return_annotation=return_annotation,
            decorators=decorators,
            is_async=False,
            is_static=is_static,
            is_classmethod=False,
            is_property=True,
            visibility=visibility,
        )

    def _extract_using(self, node: tree_sitter.Node, source: bytes) -> ExtractedImport:
        """Extract a using directive."""
        # Check for alias
        alias_node = self.find_child_by_field(node, "alias")
        alias = self.get_node_text(alias_node, source) if alias_node else None

        # Get namespace
        name_node = self.find_child_by_field(node, "name")
        if not name_node:
            # Try qualified_name or identifier
            name_node = self.find_child_by_type(node, "qualified_name")
            if not name_node:
                name_node = self.find_child_by_type(node, "identifier")

        module = self.get_node_text(name_node, source) if name_node else ""

        # Check for static using
        is_static = any(
            self.get_node_text(child, source) == "static" for child in node.children
        )

        return ExtractedImport(
            module=module,
            names=[],
            alias=alias,
            line=self.get_line_start(node),
            is_from_import=is_static,
        )

    # =========================================================================
    # Utility helpers
    # =========================================================================

    def _get_type_name(self, node: tree_sitter.Node, source: bytes) -> str:
        """Get the name of a type declaration."""
        name_node = self.find_child_by_field(node, "name")
        return self.get_node_text(name_node, source) if name_node else ""

    def _get_type_kind(self, node: tree_sitter.Node) -> str:
        """Get the kind of type declaration."""
        kind_map = {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "struct_declaration": "struct",
            "enum_declaration": "enum",
            "record_declaration": "record",
            "record_struct_declaration": "record",
        }
        return kind_map.get(node.type, "class")

    def _get_bases(self, node: tree_sitter.Node, source: bytes) -> list[str]:
        """Get base types from base_list."""
        bases: list[str] = []

        base_list = self.find_child_by_field(node, "bases")
        if not base_list:
            base_list = self.find_child_by_type(node, "base_list")

        if base_list:
            for child in base_list.children:
                if child.type in (
                    "identifier",
                    "qualified_name",
                    "generic_name",
                    "simple_base_type",
                ):
                    bases.append(self.get_node_text(child, source))

        return bases

    def _get_attributes(self, node: tree_sitter.Node, source: bytes) -> list[str]:
        """Get attributes from a declaration."""
        attributes: list[str] = []

        for child in node.children:
            if child.type == "attribute_list":
                for attr in child.children:
                    if attr.type == "attribute":
                        attr_text = self.get_node_text(attr, source)
                        attributes.append(attr_text)

        return attributes

    def _get_visibility(self, node: tree_sitter.Node, source: bytes) -> str | None:
        """Get visibility modifier."""
        visibility_keywords = {"public", "private", "protected", "internal"}

        for child in node.children:
            if child.type == "modifier":
                text = self.get_node_text(child, source)
                if text in visibility_keywords:
                    return text

        return None

    def _has_modifier(
        self, node: tree_sitter.Node, source: bytes, modifier: str
    ) -> bool:
        """Check if a node has a specific modifier."""
        for child in node.children:
            if child.type == "modifier":
                if self.get_node_text(child, source) == modifier:
                    return True
        return False

    def _extract_parameters(
        self, node: tree_sitter.Node, source: bytes
    ) -> list[dict[str, Any]]:
        """Extract method parameters."""
        params: list[dict[str, Any]] = []

        params_node = self.find_child_by_field(node, "parameters")
        if not params_node:
            params_node = self.find_child_by_type(node, "parameter_list")

        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "parameter":
                type_node = self.find_child_by_field(child, "type")
                name_node = self.find_child_by_field(child, "name")

                type_str = self.get_node_text(type_node, source) if type_node else None
                name = self.get_node_text(name_node, source) if name_node else ""

                # Check for default value
                default_node = self.find_child_by_field(child, "default")
                has_default = default_node is not None

                # Check for params keyword (varargs)
                is_params = any(
                    self.get_node_text(c, source) == "params" for c in child.children
                )
                if is_params:
                    name = f"params {name}"

                # Check for ref/out/in
                for modifier in ("ref", "out", "in"):
                    if any(
                        self.get_node_text(c, source) == modifier
                        for c in child.children
                    ):
                        name = f"{modifier} {name}"
                        break

                params.append(
                    {
                        "name": name,
                        "annotation": type_str,
                        "has_default": has_default,
                    }
                )

        return params


# Register this extractor
register_extractor("csharp", CSharpExtractor)
