"""Java language extractor using tree-sitter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tree_sitter

from ..models import ExtractedImport, ExtractedMethod, ExtractedType
from .base import LanguageExtractor
from . import register_extractor


class JavaExtractor(LanguageExtractor):
    """Extractor for Java source code using tree-sitter-java."""

    @property
    def language_name(self) -> str:
        return "java"

    def get_language(self) -> Any:
        """Return the tree-sitter Java language."""
        import tree_sitter_java
        return tree_sitter_java.language()

    def extract_types(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedType]:
        """Extract type definitions (classes, interfaces, enums, records)."""
        types: list[ExtractedType] = []
        root = tree.root_node

        type_nodes = {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
        }

        for node in self.walk_tree(root, type_nodes):
            types.append(self._extract_type(node, source))

        return types

    def extract_methods(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedMethod]:
        """Extract methods and constructors."""
        methods: list[ExtractedMethod] = []
        root = tree.root_node

        type_nodes = {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }

        for type_node in self.walk_tree(root, type_nodes):
            class_name = self._get_type_name(type_node, source)
            body = self.find_child_by_field(type_node, "body")
            if body:
                for child in body.children:
                    if child.type == "method_declaration":
                        methods.append(self._extract_method(child, source, class_name))
                    elif child.type == "constructor_declaration":
                        methods.append(self._extract_constructor(child, source, class_name))

        return methods

    def extract_imports(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedImport]:
        """Extract import declarations."""
        imports: list[ExtractedImport] = []
        root = tree.root_node

        for node in root.children:
            if node.type == "import_declaration":
                imports.append(self._extract_import(node, source))

        return imports

    # =========================================================================
    # Private extraction helpers
    # =========================================================================

    def _extract_type(
        self, node: tree_sitter.Node, source: bytes
    ) -> ExtractedType:
        """Extract a type definition (class, interface, enum, record)."""
        name = self._get_type_name(node, source)
        kind = self._get_type_kind(node)
        bases = self._get_bases(node, source)
        decorators = self._get_annotations(node, source)
        visibility = self._get_visibility(node, source)
        docstring = self._get_javadoc(node, source)

        return ExtractedType(
            name=name,
            kind=kind,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=docstring,
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

        decorators = self._get_annotations(node, source)
        visibility = self._get_visibility(node, source)
        parameters = self._extract_parameters(node, source)

        # Get return type
        return_type_node = self.find_child_by_field(node, "type")
        return_annotation = self.get_node_text(return_type_node, source) if return_type_node else None

        # Check modifiers
        is_static = self._has_modifier(node, source, "static")
        is_async = False  # Java doesn't have async in the same way

        docstring = self._get_javadoc(node, source)

        return ExtractedMethod(
            name=name,
            class_name=class_name,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=docstring,
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

        decorators = self._get_annotations(node, source)
        visibility = self._get_visibility(node, source)
        parameters = self._extract_parameters(node, source)
        docstring = self._get_javadoc(node, source)

        return ExtractedMethod(
            name=name,
            class_name=class_name,
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=docstring,
            parameters=parameters,
            return_annotation=None,  # Constructors don't have return types
            decorators=decorators,
            is_async=False,
            is_static=False,
            is_classmethod=False,
            is_property=False,
            visibility=visibility,
        )

    def _extract_import(
        self, node: tree_sitter.Node, source: bytes
    ) -> ExtractedImport:
        """Extract an import declaration."""
        # Check for static import
        is_static = any(
            self.get_node_text(child, source) == "static"
            for child in node.children
        )

        # Get the full import path
        path_node = self.find_child_by_type(node, "scoped_identifier")
        if not path_node:
            path_node = self.find_child_by_type(node, "identifier")

        full_path = self.get_node_text(path_node, source) if path_node else ""

        # Check for wildcard
        is_wildcard = any(
            child.type == "asterisk" or self.get_node_text(child, source) == "*"
            for child in node.children
        )

        # Split into module and imported name
        if "." in full_path:
            parts = full_path.rsplit(".", 1)
            module = parts[0]
            names = [parts[1]] if not is_wildcard else ["*"]
        else:
            module = full_path
            names = ["*"] if is_wildcard else []

        return ExtractedImport(
            module=module,
            names=names,
            alias=None,
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
            "enum_declaration": "enum",
            "record_declaration": "record",
            "annotation_type_declaration": "annotation",
        }
        return kind_map.get(node.type, "class")

    def _get_bases(self, node: tree_sitter.Node, source: bytes) -> list[str]:
        """Get extended/implemented types."""
        bases: list[str] = []

        # Handle extends
        superclass = self.find_child_by_field(node, "superclass")
        if superclass:
            for child in superclass.children:
                if child.type == "type_identifier":
                    bases.append(self.get_node_text(child, source))
                elif child.type == "generic_type":
                    bases.append(self.get_node_text(child, source))

        # Handle implements
        interfaces = self.find_child_by_field(node, "interfaces")
        if interfaces:
            for child in interfaces.children:
                if child.type == "type_identifier":
                    bases.append(self.get_node_text(child, source))
                elif child.type == "generic_type":
                    bases.append(self.get_node_text(child, source))
                elif child.type == "type_list":
                    for item in child.children:
                        if item.type in ("type_identifier", "generic_type"):
                            bases.append(self.get_node_text(item, source))

        return bases

    def _get_annotations(self, node: tree_sitter.Node, source: bytes) -> list[str]:
        """Get annotations from a declaration."""
        annotations: list[str] = []

        # Check for modifiers node which may contain annotations
        modifiers = self.find_child_by_type(node, "modifiers")
        if modifiers:
            for child in modifiers.children:
                if child.type in ("annotation", "marker_annotation"):
                    ann_text = self.get_node_text(child, source)
                    if ann_text.startswith("@"):
                        ann_text = ann_text[1:]
                    annotations.append(ann_text)

        # Also check direct children for annotations
        for child in node.children:
            if child.type in ("annotation", "marker_annotation"):
                ann_text = self.get_node_text(child, source)
                if ann_text.startswith("@"):
                    ann_text = ann_text[1:]
                annotations.append(ann_text)

        return annotations

    def _get_visibility(self, node: tree_sitter.Node, source: bytes) -> str | None:
        """Get visibility modifier (public, private, protected)."""
        modifiers = self.find_child_by_type(node, "modifiers")
        if modifiers:
            for child in modifiers.children:
                text = self.get_node_text(child, source)
                if text in ("public", "private", "protected"):
                    return text
        return None

    def _has_modifier(
        self, node: tree_sitter.Node, source: bytes, modifier: str
    ) -> bool:
        """Check if a node has a specific modifier."""
        modifiers = self.find_child_by_type(node, "modifiers")
        if modifiers:
            for child in modifiers.children:
                if self.get_node_text(child, source) == modifier:
                    return True
        return False

    def _get_javadoc(self, node: tree_sitter.Node, source: bytes) -> str | None:
        """Extract Javadoc comment if present before the node."""
        # Look at previous siblings or parent's previous children
        # Tree-sitter typically puts comments as separate nodes
        # This is a simplified approach
        return None  # TODO: Implement Javadoc extraction if needed

    def _extract_parameters(
        self, node: tree_sitter.Node, source: bytes
    ) -> list[dict[str, Any]]:
        """Extract method parameters."""
        params: list[dict[str, Any]] = []

        params_node = self.find_child_by_field(node, "parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "formal_parameter":
                type_node = self.find_child_by_field(child, "type")
                name_node = self.find_child_by_field(child, "name")

                type_str = self.get_node_text(type_node, source) if type_node else None
                name = self.get_node_text(name_node, source) if name_node else ""

                # Check for varargs (...)
                dimensions = self.find_child_by_field(child, "dimensions")
                if dimensions:
                    type_str = f"{type_str}[]" if type_str else "[]"

                params.append({
                    "name": name,
                    "annotation": type_str,
                    "has_default": False,
                })
            elif child.type == "spread_parameter":
                # Varargs: Type... name
                type_node = self.find_child_by_field(child, "type")
                name_node = self.find_child_by_field(child, "name")

                type_str = self.get_node_text(type_node, source) if type_node else None
                name = self.get_node_text(name_node, source) if name_node else ""

                params.append({
                    "name": f"...{name}",
                    "annotation": type_str,
                    "has_default": False,
                })

        return params


# Register this extractor
register_extractor("java", JavaExtractor)
