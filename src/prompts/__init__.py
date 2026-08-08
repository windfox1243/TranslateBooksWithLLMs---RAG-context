"""
Prompts module for TranslateBookWithLLM
"""
from src.prompts.prompts import (
    PromptPair,
    generate_subtitle_block_prompt,
    generate_translation_prompt,
)

__all__ = [
    "PromptPair",
    "generate_translation_prompt",
    "generate_subtitle_block_prompt",
]
