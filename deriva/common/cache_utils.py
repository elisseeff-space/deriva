"""
Common caching utilities for Deriva.

Provides a base class for two-tier (memory + disk) caching and utilities
for generating cache keys. Used by LLM cache, graph cache, and other
caching implementations.

Usage:
    from deriva.common.cache_utils import BaseDiskCache, hash_inputs

    class MyCache(BaseDiskCache):
        def generate_key(self, *args) -> str:
            return hash_inputs(*args)
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from deriva.common.exceptions import CacheError


def hash_inputs(*args: Any, separator: str = "|") -> str:
    """
    Generate SHA256 hash from arbitrary inputs.

    Args:
        *args: Values to hash (will be converted to strings)
        separator: Separator between values (default: "|")

    Returns:
        SHA256 hex digest

    Example:
        >>> hash_inputs("prompt", "gpt-4", {"key": "value"})
        'a1b2c3...'  # 64-char hex string
    """
    parts = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, dict):
            # Sort dict keys for consistent hashing
            parts.append(json.dumps(arg, sort_keys=True, default=str))
        elif isinstance(arg, (list, tuple)):
            parts.append(json.dumps(arg, sort_keys=True, default=str))
        else:
            parts.append(str(arg))

    combined = separator.join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def dict_to_hashable(d: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """
    Convert a dict to a hashable tuple representation.

    Recursively converts nested dicts and lists to tuples.
    Useful for using dicts as cache keys with @lru_cache.

    Args:
        d: Dictionary to convert

    Returns:
        Nested tuple representation that can be hashed

    Example:
        >>> dict_to_hashable({"a": 1, "b": {"c": 2}})
        (('a', 1), ('b', (('c', 2),)))
    """
    items: list[tuple[str, Any]] = []
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            items.append((k, dict_to_hashable(v)))
        elif isinstance(v, list):
            # Convert list items recursively
            list_items: list[Any] = []
            for item in v:
                if isinstance(item, dict):
                    list_items.append(dict_to_hashable(item))
                else:
                    list_items.append(item)
            items.append((k, tuple(list_items)))
        else:
            items.append((k, v))
    return tuple(items)


@lru_cache(maxsize=128)
def _hash_dict_tuple(dict_tuple: tuple[tuple[str, Any], ...]) -> str:
    """
    Generate JSON string from a dict tuple (cached for performance).

    This is an internal function used to avoid repeated JSON serialization
    of the same dict structures.

    Args:
        dict_tuple: Tuple from dict_to_hashable()

    Returns:
        JSON string representation
    """
    # Convert back to dict for JSON serialization
    def tuple_to_dict(t: tuple) -> dict | list | Any:
        if isinstance(t, tuple) and len(t) > 0:
            # Check if it's a key-value tuple (dict item)
            if isinstance(t[0], tuple) and len(t[0]) == 2:
                return {k: tuple_to_dict(v) for k, v in t}
            # Check if it's a single key-value pair
            if len(t) == 2 and isinstance(t[0], str):
                return t  # Return as-is, handled by parent
        return t

    result = dict(dict_tuple)
    return json.dumps(result, sort_keys=True)


class BaseDiskCache:
    """
    Base class for two-tier (memory + disk) caching with JSON persistence.

    Provides a generic caching interface that stores entries in both memory
    (for fast access) and on disk (for persistence across runs).

    Subclasses should implement domain-specific key generation.

    Attributes:
        cache_dir: Path to the directory storing cache files
        _memory_cache: In-memory cache dictionary

    Example:
        class MyCache(BaseDiskCache):
            def __init__(self):
                super().__init__("./my_cache")

            def get_or_compute(self, key: str, compute_fn) -> Any:
                cached = self.get(key)
                if cached is not None:
                    return cached["data"]
                result = compute_fn()
                self.set(key, {"data": result})
                return result
    """

    def __init__(self, cache_dir: str | Path):
        """
        Initialize cache with specified directory.

        Args:
            cache_dir: Directory to store cache files (created if not exists)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def get_from_memory(self, cache_key: str) -> dict[str, Any] | None:
        """
        Retrieve cached data from in-memory cache.

        Args:
            cache_key: The cache key

        Returns:
            Cached data dict or None if not found
        """
        return self._memory_cache.get(cache_key)

    def get_from_disk(self, cache_key: str) -> dict[str, Any] | None:
        """
        Retrieve cached data from disk.

        Args:
            cache_key: The cache key

        Returns:
            Cached data dict or None if not found

        Raises:
            CacheError: If cache file is corrupted
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise CacheError(f"Corrupted cache file: {cache_file}") from e
        except Exception as e:
            raise CacheError(f"Error reading cache file: {e}") from e

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """
        Retrieve cached data, checking memory first, then disk.

        Args:
            cache_key: The cache key

        Returns:
            Cached data dict or None if not found
        """
        # Check memory cache first
        cached = self.get_from_memory(cache_key)
        if cached is not None:
            return cached

        # Check disk cache
        cached = self.get_from_disk(cache_key)
        if cached is not None:
            # Populate memory cache for faster future access
            self._memory_cache[cache_key] = cached

        return cached

    def set(self, cache_key: str, data: dict[str, Any]) -> None:
        """
        Store data in both memory and disk cache.

        Args:
            cache_key: The cache key
            data: Dictionary to cache (must be JSON-serializable)

        Raises:
            CacheError: If unable to write to disk
        """
        # Store in memory
        self._memory_cache[cache_key] = data

        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            raise CacheError(f"Error writing cache file: {e}") from e

    def invalidate(self, cache_key: str) -> None:
        """
        Remove entry from both memory and disk cache.

        Args:
            cache_key: The cache key to invalidate
        """
        # Remove from memory
        self._memory_cache.pop(cache_key, None)

        # Remove from disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception as e:
                raise CacheError(f"Error deleting cache file: {e}") from e

    def clear_memory(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()

    def clear_disk(self) -> None:
        """
        Clear all cache files from disk.

        Raises:
            CacheError: If unable to delete cache files
        """
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            raise CacheError(f"Error clearing disk cache: {e}") from e

    def clear_all(self) -> None:
        """Clear both memory and disk caches."""
        self.clear_memory()
        self.clear_disk()

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with:
                - memory_entries: Number of entries in memory
                - disk_entries: Number of files on disk
                - disk_size_bytes: Total size of cache files
                - disk_size_mb: Total size in megabytes
                - cache_dir: Path to cache directory
        """
        disk_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in disk_files)

        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(disk_files),
            "disk_size_bytes": total_size,
            "disk_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def keys(self) -> list[str]:
        """
        Get all cache keys (from disk).

        Returns:
            List of cache keys
        """
        return [f.stem for f in self.cache_dir.glob("*.json")]


__all__ = [
    "BaseDiskCache",
    "hash_inputs",
    "dict_to_hashable",
]
