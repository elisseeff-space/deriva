"""Tree-sitter adapter - Multi-language code analysis using tree-sitter.

Provides deterministic extraction of types, methods, and imports from
Python, JavaScript, Java, C#, and Perl source files.
"""

from __future__ import annotations

from .manager import TreeSitterManager
from .models import (
    ExtractedImport,
    ExtractedMethod,
    ExtractedType,
)

__all__ = [
    "TreeSitterManager",
    "ExtractedType",
    "ExtractedMethod",
    "ExtractedImport",
]
