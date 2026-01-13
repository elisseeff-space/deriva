"""Data models for tree-sitter extraction results.

These models are language-agnostic and used across all supported languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedType:
    """Represents an extracted type definition (class, interface, function, etc.)."""

    name: str
    kind: str  # 'class', 'interface', 'enum', 'struct', 'function', 'type_alias', 'package'
    line_start: int
    line_end: int
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    visibility: str | None = None  # 'public', 'private', 'protected', etc.


@dataclass
class ExtractedMethod:
    """Represents an extracted method/function."""

    name: str
    class_name: str | None  # None for top-level functions
    line_start: int
    line_end: int
    docstring: str | None = None
    parameters: list[dict[str, Any]] = field(default_factory=list)
    return_annotation: str | None = None
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    visibility: str | None = None  # 'public', 'private', 'protected', etc.


@dataclass
class ExtractedImport:
    """Represents an extracted import statement."""

    module: str
    names: list[str]  # What is imported (empty for 'import module')
    alias: str | None = None
    line: int = 0
    is_from_import: bool = False


__all__ = [
    "ExtractedType",
    "ExtractedMethod",
    "ExtractedImport",
]
