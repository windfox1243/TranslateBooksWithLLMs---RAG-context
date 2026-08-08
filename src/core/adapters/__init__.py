"""
Adapter pattern implementation for generic file translation.

This module provides a unified interface for translating different file formats
(TXT, SRT, EPUB, PDF) using the adapter pattern. Each format implements the
FormatAdapter interface to handle format-specific operations.

New in Phase 5: Comprehensive error handling system with:
- Custom exception hierarchy
- Retry logic with exponential backoff
- Error recovery strategies
- Comprehensive logging
"""

from .epub_adapter import EpubAdapter
from .error_handler import ErrorHandler, with_error_handling
from .error_logger import ErrorLogger, ErrorLoggerContext, ErrorRecord, ErrorSeverity
from .error_recovery import (
    ContentSplitter,
    ErrorRecoveryManager,
    GracefulDegradation,
    RecoveryResult,
)

# Error handling system (Phase 5)
from .exceptions import (
    AdapterError,
    AdapterInitializationError,
    AdapterPreparationError,
    AdapterReconstructionError,
    CheckpointError,
    ContextOverflowError,
    FileFormatError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    RepetitionLoopError,
    RetryExhaustedError,
    TranslationError,
    TranslationUnitError,
    UnitTranslationError,
    UnsupportedFormatError,
)
from .format_adapter import FormatAdapter
from .generic_translator import GenericTranslator

# Refine-only entry point
from .refine_file import refine_file
from .retry_manager import RetryConfig, RetryManager, RetryStrategy, with_retry
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
    'AdapterError',
    'AdapterInitializationError',
    'AdapterPreparationError',
    'AdapterReconstructionError',
    'TranslationUnitError',
    'UnitTranslationError',
    'LLMError',
    'ContextOverflowError',
    'RepetitionLoopError',
    'LLMConnectionError',
    'LLMRateLimitError',
    'CheckpointError',
    'FileFormatError',
    'UnsupportedFormatError',
    'RetryExhaustedError',

    # Error handling - Retry
    'RetryManager',
    'RetryConfig',
    'RetryStrategy',
    'with_retry',

    # Error handling - Recovery
    'ErrorRecoveryManager',
    'RecoveryResult',
    'ContentSplitter',
    'GracefulDegradation',

    # Error handling - Logging
    'ErrorLogger',
    'ErrorLoggerContext',
    'ErrorSeverity',
    'ErrorRecord',

    # Error handling - Unified handler
    'ErrorHandler',
    'with_error_handling',
]
