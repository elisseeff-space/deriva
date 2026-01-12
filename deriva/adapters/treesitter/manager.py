"""Tree-sitter based extraction manager for multi-language code analysis.

Provides deterministic, precise extraction that complements LLM-based semantic extraction.
Extracts types, methods, and imports from source files using tree-sitter parsers.
"""

from __future__ import annotations

import os
from typing import Any

import tree_sitter

from .models import ExtractedImport, ExtractedMethod, ExtractedType
from .languages import get_extractor, supported_languages


class TreeSitterManager:
    """Unified manager for tree-sitter-based code extraction across all languages."""

    # Supported languages
    SUPPORTED_LANGUAGES = {"python", "javascript", "typescript", "java", "csharp"}

    # Extension to language mapping
    EXTENSION_MAP: dict[str, str] = {
        ".py": "python",
        ".pyw": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mts": "typescript",
        ".cts": "typescript",
        ".java": "java",
        ".cs": "csharp",
    }

    def __init__(self) -> None:
        """Initialize the manager."""
        self._parsers: dict[str, tree_sitter.Parser] = {}

    def extract_types(
        self, source: str, file_path: str | None = None, language: str | None = None
    ) -> list[ExtractedType]:
        """Extract type definitions from source code.

        Args:
            source: Source code as string
            file_path: Optional file path (used for language detection)
            language: Optional language override

        Returns:
            List of ExtractedType objects
        """
        lang = self._resolve_language(file_path, language)
        if not lang:
            return []

        extractor = get_extractor(lang)
        if not extractor:
            return []

        tree = self._parse(source, lang, extractor)
        if not tree:
            return []

        return extractor.extract_types(tree, source.encode("utf-8"))

    def extract_methods(
        self, source: str, file_path: str | None = None, language: str | None = None
    ) -> list[ExtractedMethod]:
        """Extract method and function definitions from source code.

        Args:
            source: Source code as string
            file_path: Optional file path (used for language detection)
            language: Optional language override

        Returns:
            List of ExtractedMethod objects
        """
        lang = self._resolve_language(file_path, language)
        if not lang:
            return []

        extractor = get_extractor(lang)
        if not extractor:
            return []

        tree = self._parse(source, lang, extractor)
        if not tree:
            return []

        return extractor.extract_methods(tree, source.encode("utf-8"))

    def extract_imports(
        self, source: str, file_path: str | None = None, language: str | None = None
    ) -> list[ExtractedImport]:
        """Extract import statements from source code.

        Args:
            source: Source code as string
            file_path: Optional file path (used for language detection)
            language: Optional language override

        Returns:
            List of ExtractedImport objects
        """
        lang = self._resolve_language(file_path, language)
        if not lang:
            return []

        extractor = get_extractor(lang)
        if not extractor:
            return []

        tree = self._parse(source, lang, extractor)
        if not tree:
            return []

        return extractor.extract_imports(tree, source.encode("utf-8"))

    def extract_all(
        self, source: str, file_path: str | None = None, language: str | None = None
    ) -> dict[str, Any]:
        """Extract all elements from source code.

        Args:
            source: Source code as string
            file_path: Optional file path (used for language detection)
            language: Optional language override

        Returns:
            Dictionary with 'types', 'methods', 'imports' keys
        """
        lang = self._resolve_language(file_path, language)
        if not lang:
            return {"types": [], "methods": [], "imports": []}

        extractor = get_extractor(lang)
        if not extractor:
            return {"types": [], "methods": [], "imports": []}

        tree = self._parse(source, lang, extractor)
        if not tree:
            return {"types": [], "methods": [], "imports": []}

        source_bytes = source.encode("utf-8")
        return {
            "types": extractor.extract_types(tree, source_bytes),
            "methods": extractor.extract_methods(tree, source_bytes),
            "imports": extractor.extract_imports(tree, source_bytes),
        }

    @classmethod
    def detect_language(cls, file_path: str) -> str | None:
        """Detect language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None if not detected
        """
        ext = os.path.splitext(file_path)[1].lower()
        return cls.EXTENSION_MAP.get(ext)

    @classmethod
    def supports_language(cls, language: str) -> bool:
        """Check if a language is supported.

        Args:
            language: Language name

        Returns:
            True if the language is supported
        """
        return language.lower() in cls.SUPPORTED_LANGUAGES

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of supported languages.

        Returns:
            List of supported language names
        """
        return supported_languages()

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _resolve_language(
        self, file_path: str | None, language: str | None
    ) -> str | None:
        """Resolve the language to use for extraction.

        Args:
            file_path: Optional file path
            language: Optional explicit language

        Returns:
            Language name or None
        """
        if language:
            lang = language.lower()
            # Handle typescript -> javascript mapping for now
            # (we use JS grammar for TS until dedicated TS extractor exists)
            if lang == "typescript":
                lang = "javascript"
            return lang if lang in self.SUPPORTED_LANGUAGES else None

        if file_path:
            detected = self.detect_language(file_path)
            if detected == "typescript":
                detected = "javascript"
            return detected

        return None

    def _get_parser(self, language: str, extractor: Any) -> tree_sitter.Parser:
        """Get or create a parser for the given language.

        Args:
            language: Language name
            extractor: Language extractor (to get the grammar)

        Returns:
            Configured tree-sitter parser
        """
        if language not in self._parsers:
            raw_language = extractor.get_language()
            # Wrap in Language object (tree-sitter 0.24+ API)
            ts_language = tree_sitter.Language(raw_language)
            parser = tree_sitter.Parser(ts_language)
            self._parsers[language] = parser

        return self._parsers[language]

    def _parse(
        self, source: str, language: str, extractor: Any
    ) -> tree_sitter.Tree | None:
        """Parse source code into a tree-sitter tree.

        Args:
            source: Source code as string
            language: Language name
            extractor: Language extractor

        Returns:
            Parsed tree or None on failure
        """
        try:
            parser = self._get_parser(language, extractor)
            return parser.parse(source.encode("utf-8"))
        except Exception:
            return None


__all__ = ["TreeSitterManager"]
