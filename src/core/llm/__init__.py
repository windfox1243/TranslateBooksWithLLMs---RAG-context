"""
LLM Provider System

This package provides a modular system for interacting with various LLM providers.

Public API:
    - Exceptions: ContextOverflowError, RepetitionLoopError
    - Base classes: LLMProvider, LLMResponse
    - Thinking system: ThinkingBehavior, get_thinking_behavior_sync, get_model_warning_message, detect_repetition_loop
    - Utilities: ContextDetector, TranslationExtractor
    - Providers: OllamaProvider, OpenAICompatibleProvider, OpenRouterProvider, GeminiProvider
    - Factory: create_llm_provider

Example usage:
    >>> from src.core.llm import create_llm_provider, OllamaProvider
    >>> provider = create_llm_provider("ollama", model="llama3")
    >>> response = await provider.generate("Translate: Hello")
"""

# Base classes
from .base import LLMGenerationOptions, LLMProvider, LLMResponse

# Exceptions
from .exceptions import ContextOverflowError, RateLimitError, RepetitionLoopError

# Factory
from .factory import create_llm_provider
from .providers.deepseek import DeepSeekProvider
from .providers.gemini import GeminiProvider
from .providers.mistral import MistralProvider

# Providers
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.openrouter import OpenRouterProvider
from .providers.poe import PoeProvider

# Thinking system
from .thinking.behavior import (
    ThinkingBehavior,
    get_model_warning_message,
    get_thinking_behavior_sync,
)
from .thinking.cache import ThinkingCache, get_thinking_cache
from .thinking.detection import detect_repetition_loop

# Utilities
from .utils.context_detection import ContextDetector
from .utils.extraction import TranslationExtractor

__all__ = [
    # Exceptions
    'ContextOverflowError',
    'RepetitionLoopError',
    'RateLimitError',

    # Base
    'LLMProvider',
    'LLMResponse',
    'LLMGenerationOptions',

    # Thinking
    'ThinkingBehavior',
    'ThinkingCache',
    'get_thinking_cache',
    'get_thinking_behavior_sync',
    'get_model_warning_message',
    'detect_repetition_loop',

    # Utilities
    'ContextDetector',
    'TranslationExtractor',

    # Providers
    'OllamaProvider',
    'OpenAICompatibleProvider',
    'OpenRouterProvider',
    'GeminiProvider',
    'MistralProvider',
    'DeepSeekProvider',
    'PoeProvider',

    # Factory
    'create_llm_provider',
]
