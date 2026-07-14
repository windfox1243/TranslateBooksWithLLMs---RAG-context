"""Shared forwarding support for incremental database decomposition."""

from __future__ import annotations

from typing import Any, Iterable


class DatabaseRepository:
    """Expose only an owned subset of methods from the compatibility facade."""

    prefixes: tuple[str, ...] = ()
    methods: frozenset[str] = frozenset()

    def __init__(self, database: Any):
        self.database = database

    def __getattr__(self, name: str):
        if name in self.methods or name.startswith(self.prefixes):
            value = getattr(self.database, name)
            if callable(value):
                return value
        raise AttributeError(f"{type(self).__name__} does not own {name!r}")

    def available_methods(self) -> Iterable[str]:
        for name in dir(self.database):
            if name in self.methods or name.startswith(self.prefixes):
                if callable(getattr(self.database, name, None)):
                    yield name
