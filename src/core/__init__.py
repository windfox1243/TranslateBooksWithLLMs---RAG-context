"""
Core translation modules
"""
from .epub import translate_epub_file
from .text_processor import split_text_into_chunks
from .translator import generate_translation_request

__all__ = [
    'split_text_into_chunks',
    'generate_translation_request',
    'translate_epub_file'
]
