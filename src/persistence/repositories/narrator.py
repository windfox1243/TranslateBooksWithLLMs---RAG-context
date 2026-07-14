"""Narrator-policy persistence boundary."""

from .base import DatabaseRepository


class NarratorRepository(DatabaseRepository):
    prefixes = ("narrator_",)
    methods = frozenset({"quarantine_narrator_voice_after"})
