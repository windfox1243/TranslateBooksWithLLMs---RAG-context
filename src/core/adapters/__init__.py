"""
Adapter pattern implementation for generic file translation.

This module provides a unified interface for translating different file formats
(TXT, SRT, EPUB, PDF) using the adapter pattern. Each format implements the
FormatAdapter interface to handle format-specific operations.

"""

from .epub_adapter import EpubAdapter

# Error handling system (Phase 5)
from .exceptions import FileFormatError, TranslationError, UnsupportedFormatError
from .format_adapter import FormatAdapter
from .generic_translator import GenericTranslator

# Refine-only entry point
from .refine_file import refine_file
from .srt_adapter import SrtAdapter

# Unified translation entry point (Phase 6)
from .translate_file import (
    build_translated_output,
    get_file_type_from_path,
    translate_file,
)
from .translation_unit import TranslationUnit
from .txt_adapter import TxtAdapter

__all__ = [
    # Core adapter components
    'TranslationUnit',
    'FormatAdapter',
    'GenericTranslator',
    'TxtAdapter',
    'SrtAdapter',
    'EpubAdapter',

    # Unified translation entry point (Phase 6)
    'translate_file',
    'get_file_type_from_path',
    'build_translated_output',

    # Refine-only entry point
    'refine_file',

    # Error handling - Exceptions
    'TranslationError',
    'FileFormatError',
    'UnsupportedFormatError',




]
