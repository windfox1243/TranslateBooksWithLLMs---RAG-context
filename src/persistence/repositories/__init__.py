"""Narrow persistence interfaces backed by the legacy Database facade."""

from .context import ContextRepository
from .editor import EditorRepository
from .jobs import JobRepository
from .narrator import NarratorRepository

__all__ = [
    "ContextRepository",
    "EditorRepository",
    "JobRepository",
    "NarratorRepository",
]
