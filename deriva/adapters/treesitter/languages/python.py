"""Python language extractor using tree-sitter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tree_sitter

from ..models import ExtractedImport, ExtractedMethod, ExtractedType
from .base import LanguageExtractor
from . import register_extractor


class PythonExtractor(LanguageExtractor):
    """Extractor for Python source code using tree-sitter-python."""

    @property
    def language_name(self) -> str:
        return "python"

    def get_language(self) -> Any:
        """Return the tree-sitter Python language."""
        import tree_sitter_python
        return tree_sitter_python.language()

    def extract_types(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedType]:
        """Extract type definitions (classes, top-level functions, type aliases)."""
        types: list[ExtractedType] = []
        root = tree.root_node

        # Extract classes
        for node in self.walk_tree(root, {"class_definition"}):
            types.append(self._extract_class(node, source))

        # Extract top-level functions (not methods inside classes)
        for node in root.children:
            if node.type == "function_definition":
                types.append(self._extract_function_as_type(node, source))
            elif node.type == "decorated_definition":
                inner = self._get_decorated_inner(node)
                if inner and inner.type == "function_definition":
                    types.append(self._extract_function_as_type(node, source, inner))

        # Extract type aliases (Python 3.12+ type statement or TypeAlias annotation)
        for node in self.walk_tree(root, {"type_alias_statement"}):
            types.append(self._extract_type_alias(node, source))

        return types

    def extract_methods(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedMethod]:
        """Extract methods and functions."""
        methods: list[ExtractedMethod] = []
        root = tree.root_node

        # Extract methods from classes
        for class_node in self.walk_tree(root, {"class_definition"}):
            class_name = self._get_class_name(class_node, source)
            body = self.find_child_by_field(class_node, "body")
            if body:
                for item in body.children:
                    if item.type == "function_definition":
                        methods.append(self._extract_method(item, source, class_name))
                    elif item.type == "decorated_definition":
                        inner = self._get_decorated_inner(item)
                        if inner and inner.type == "function_definition":
                            methods.append(
                                self._extract_method(item, source, class_name, inner)
                            )

        # Extract top-level functions
        for node in root.children:
            if node.type == "function_definition":
                methods.append(self._extract_method(node, source, None))
            elif node.type == "decorated_definition":
                inner = self._get_decorated_inner(node)
                if inner and inner.type == "function_definition":
                    methods.append(self._extract_method(node, source, None, inner))

        return methods

    def extract_imports(
        self, tree: tree_sitter.Tree, source: bytes
    ) -> list[ExtractedImport]:
        """Extract import statements."""
        imports: list[ExtractedImport] = []
        root = tree.root_node

        for node in root.children:
            if node.type == "import_statement":
                imports.extend(self._extract_import(node, source))
            elif node.type == "import_from_statement":
                imports.append(self._extract_from_import(node, source))

        return imports

    # =========================================================================
    # Private extraction helpers
    # =========================================================================

    def _extract_class(
        self, node: tree_sitter.Node, source: bytes
    ) -> ExtractedType:
        """Extract a class definition."""
        name = self._get_class_name(node, source)
        bases = self._get_bases(node, source)
        decorators = self._get_decorators(node, source)
        docstring = self._get_docstring(node, source)

        return ExtractedType(
            name=name,
            kind="class",
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=docstring,
            bases=bases,
            decorators=decorators,
            is_async=False,
        )

    def _extract_function_as_type(
        self,
        node: tree_sitter.Node,
        source: bytes,
        inner_func: tree_sitter.Node | None = None,
    ) -> ExtractedType:
        """Extract a top-level function as a type definition."""
        func_node = inner_func or node
        decorators = self._get_decorators(node, source) if node.type == "decorated_definition" else []

        name_node = self.find_child_by_field(func_node, "name")
        name = self.get_node_text(name_node, source) if name_node else ""

        is_async = func_node.type == "function_definition" and any(
            child.type == "async" for child in func_node.children
        )
        # Check parent for async
        if not is_async:
            for child in func_node.children:
                if self.get_node_text(child, source) == "async":
                    is_async = True
                    break

        docstring = self._get_docstring(func_node, source)

        return ExtractedType(
            name=name,
            kind="function",
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=docstring,
            bases=[],
            decorators=decorators,
            is_async=is_async,
        )

    def _extract_type_alias(
        self, node: tree_sitter.Node, source: bytes
    ) -> ExtractedType:
        """Extract a type alias (type X = ...)."""
        name_node = self.find_child_by_field(node, "name")
        name = self.get_node_text(name_node, source) if name_node else ""

        return ExtractedType(
            name=name,
            kind="type_alias",
            line_start=self.get_line_start(node),
            line_end=self.get_line_end(node),
            docstring=None,
            bases=[],
            decorators=[],
            is_async=False,
        )

    def _extract_method(
        self,
        node: tree_sitter.Node,
        source: bytes,
        class_name: str | None,
        inner_func: tree_sitter.Node | None = None,
    ) -> ExtractedMethod:
        """Extract a method or function definition."""
        func_node = inner_func or node
        decorators = self._get_decorators(node, source) if node.type == "decorated_definition" else []
        if not decorators and node.type == "function_definition":
            # Check if parent is decorated_definition
            decorators = []

        name_node = self.find_child_by_field(func_node, "name")
        name = self.get_node_text(name_node, source) if name_node else ""

        # Check for async
        is_async = False
        node_text = self.get_node_text(func_node, source)
        if node_text.strip().startswith("async "):
            is_async = True

        # Determine method characteristics from decorators
        decorator_names = [d.split("(")[0] for d in decorators]
        is_static = "staticmethod" in decorator_names
        is_classmethod = "classmethod" in decorator_names
        is_property = "property" in decorator_names

        # Extract parameters
        parameters = self._extract_parameters(func_node, source)

        # Extract return annotation
        return_type = self.find_child_by_field(func_node, "return_type")
        return_annotation = self.get_node_text(return_type, source) if return_type else None
        if return_annotation and return_annotation.startswith("->"):
            return_annotation = return_annotation[2:].strip()

        docstring = self._get_docstring(func_node, source)

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
            is_classmethod=is_classmethod,
            is_property=is_property,
        )

    def _extract_import(
        self, node: tree_sitter.Node, source: bytes
    ) -> list[ExtractedImport]:
        """Extract regular import statement (import x, y, z)."""
        imports: list[ExtractedImport] = []

        for child in node.children:
            if child.type == "dotted_name":
                module = self.get_node_text(child, source)
                imports.append(
                    ExtractedImport(
                        module=module,
                        names=[],
                        alias=None,
                        line=self.get_line_start(node),
                        is_from_import=False,
                    )
                )
            elif child.type == "aliased_import":
                name_node = self.find_child_by_type(child, "dotted_name")
                alias_node = self.find_child_by_field(child, "alias")
                module = self.get_node_text(name_node, source) if name_node else ""
                alias = self.get_node_text(alias_node, source) if alias_node else None
                imports.append(
                    ExtractedImport(
                        module=module,
                        names=[],
                        alias=alias,
                        line=self.get_line_start(node),
                        is_from_import=False,
                    )
                )

        return imports

    def _extract_from_import(
        self, node: tree_sitter.Node, source: bytes
    ) -> ExtractedImport:
        """Extract from ... import statement."""
        module_node = self.find_child_by_field(node, "module_name")
        module = self.get_node_text(module_node, source) if module_node else ""

        # Handle relative imports
        prefix = ""
        for child in node.children:
            if child.type == "relative_import":
                for subchild in child.children:
                    if subchild.type == "import_prefix":
                        prefix = self.get_node_text(subchild, source)
                    elif subchild.type == "dotted_name":
                        module = self.get_node_text(subchild, source)
                break

        if prefix:
            module = prefix + module

        # Extract imported names
        names: list[str] = []
        for child in node.children:
            if child.type == "import_from_specifier" or child.type == "dotted_name":
                # Skip the module name
                if child == module_node:
                    continue
                name = self.get_node_text(child, source)
                if name and name not in ("from", "import", ","):
                    names.append(name)
            elif child.type == "wildcard_import":
                names.append("*")

        return ExtractedImport(
            module=module,
            names=names,
            alias=None,
            line=self.get_line_start(node),
            is_from_import=True,
        )

    # =========================================================================
    # Utility helpers
    # =========================================================================

    def _get_class_name(self, node: tree_sitter.Node, source: bytes) -> str:
        """Get the name of a class."""
        name_node = self.find_child_by_field(node, "name")
        return self.get_node_text(name_node, source) if name_node else ""

    def _get_bases(self, node: tree_sitter.Node, source: bytes) -> list[str]:
        """Get base classes from class definition."""
        bases: list[str] = []
        superclass_node = self.find_child_by_field(node, "superclasses")
        if superclass_node:
            # argument_list contains the base classes
            for child in superclass_node.children:
                if child.type in ("identifier", "attribute"):
                    bases.append(self.get_node_text(child, source))
                elif child.type == "argument":
                    # Handle keyword arguments like metaclass=...
                    pass
        return bases

    def _get_decorators(self, node: tree_sitter.Node, source: bytes) -> list[str]:
        """Get decorators from a decorated definition or function/class."""
        decorators: list[str] = []

        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "decorator":
                    dec_text = self.get_node_text(child, source)
                    # Remove the @ prefix
                    if dec_text.startswith("@"):
                        dec_text = dec_text[1:]
                    decorators.append(dec_text)
        return decorators

    def _get_docstring(self, node: tree_sitter.Node, source: bytes) -> str | None:
        """Extract docstring from a class or function."""
        body = self.find_child_by_field(node, "body")
        if not body:
            return None

        # First statement in body might be a docstring
        for child in body.children:
            if child.type == "expression_statement":
                for subchild in child.children:
                    if subchild.type == "string":
                        text = self.get_node_text(subchild, source)
                        # Remove quotes
                        if text.startswith('"""') and text.endswith('"""'):
                            return text[3:-3].strip()
                        elif text.startswith("'''") and text.endswith("'''"):
                            return text[3:-3].strip()
                        elif text.startswith('"') and text.endswith('"'):
                            return text[1:-1].strip()
                        elif text.startswith("'") and text.endswith("'"):
                            return text[1:-1].strip()
                break  # Only check first statement

        return None

    def _get_decorated_inner(
        self, node: tree_sitter.Node
    ) -> tree_sitter.Node | None:
        """Get the inner definition from a decorated_definition."""
        if node.type != "decorated_definition":
            return None
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child
        return None

    def _extract_parameters(
        self, func_node: tree_sitter.Node, source: bytes
    ) -> list[dict[str, Any]]:
        """Extract function parameters with type annotations."""
        params: list[dict[str, Any]] = []

        params_node = self.find_child_by_field(func_node, "parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "identifier":
                # Simple parameter without annotation
                params.append({
                    "name": self.get_node_text(child, source),
                    "annotation": None,
                    "has_default": False,
                })
            elif child.type == "typed_parameter":
                name_node = self.find_child_by_type(child, "identifier")
                type_node = self.find_child_by_field(child, "type")
                name = self.get_node_text(name_node, source) if name_node else ""
                annotation = self.get_node_text(type_node, source) if type_node else None
                params.append({
                    "name": name,
                    "annotation": annotation,
                    "has_default": False,
                })
            elif child.type == "default_parameter":
                name_node = self.find_child_by_type(child, "identifier")
                name = self.get_node_text(name_node, source) if name_node else ""
                params.append({
                    "name": name,
                    "annotation": None,
                    "has_default": True,
                })
            elif child.type == "typed_default_parameter":
                name_node = self.find_child_by_type(child, "identifier")
                type_node = self.find_child_by_field(child, "type")
                name = self.get_node_text(name_node, source) if name_node else ""
                annotation = self.get_node_text(type_node, source) if type_node else None
                params.append({
                    "name": name,
                    "annotation": annotation,
                    "has_default": True,
                })
            elif child.type == "list_splat_pattern":
                # *args
                name_node = self.find_child_by_type(child, "identifier")
                name = self.get_node_text(name_node, source) if name_node else ""
                params.append({
                    "name": f"*{name}",
                    "annotation": None,
                    "has_default": False,
                })
            elif child.type == "dictionary_splat_pattern":
                # **kwargs
                name_node = self.find_child_by_type(child, "identifier")
                name = self.get_node_text(name_node, source) if name_node else ""
                params.append({
                    "name": f"**{name}",
                    "annotation": None,
                    "has_default": False,
                })

        return params


# Register this extractor
register_extractor("python", PythonExtractor)
