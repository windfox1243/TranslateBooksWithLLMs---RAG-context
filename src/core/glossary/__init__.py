"""
Glossary module for consistent translation of recurring terms.

Provides:
- models: Glossary, GlossaryTerm, GlossaryConfig dataclasses
- store: SQLite CRUD operations
- filter: chunk-aware glossary filtering (Latin word-boundary, CJK substring)
- injector: build the glossary block injected into the system prompt
"""
from src.core.glossary.models import (
    BulkReplaceResult,
    Glossary,
    GlossaryConfig,
    GlossaryTerm,
)
from src.core.glossary.filter import filter_glossary
from src.core.glossary.injector import build_glossary_block
from src.core.glossary.store import GlossaryStore
from src.core.glossary.ner import parse_ner_response, suggest_terms

__all__ = [
    "BulkReplaceResult",
    "Glossary",
    "GlossaryTerm",
    "GlossaryConfig",
    "GlossaryStore",
    "filter_glossary",
    "build_glossary_block",
    "parse_ner_response",
    "suggest_terms",
]
