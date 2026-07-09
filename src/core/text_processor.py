"""
Text processing module for chunking and context management
"""
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import TranslationConfig


# Legacy line-based chunking functions removed
# All text chunking now uses token-based approach via split_text_into_chunks()


def split_text_into_chunks(
    text: str,
    config: Optional['TranslationConfig'] = None,
    max_tokens_per_chunk: Optional[int] = None,
    soft_limit_ratio: Optional[float] = None,
    chapter_mode: bool = False,
) -> List[Dict[str, str]]:
    """
    Split text into chunks with context preservation using token-based chunking.

    Args:
        text: Input text to split
        config: TranslationConfig object (optional, for default values)
        max_tokens_per_chunk: Override for max tokens per chunk
        soft_limit_ratio: Override for soft limit ratio
        chapter_mode: Keep detected chapters as independent semantic ranges

    Returns:
        List of chunk dictionaries with context_before, main_content, context_after
    """
    from src.config import MAX_TOKENS_PER_CHUNK, SOFT_LIMIT_RATIO

    # Determine settings from config or defaults
    if config is not None:
        _max_tokens = max_tokens_per_chunk if max_tokens_per_chunk is not None else config.max_tokens_per_chunk
        _soft_limit = soft_limit_ratio if soft_limit_ratio is not None else config.soft_limit_ratio
    else:
        _max_tokens = max_tokens_per_chunk if max_tokens_per_chunk is not None else MAX_TOKENS_PER_CHUNK
        _soft_limit = soft_limit_ratio if soft_limit_ratio is not None else SOFT_LIMIT_RATIO

    # Token-based chunking
    from src.core.chunking.token_chunker import TokenChunker
    chunker = TokenChunker(
        max_tokens=_max_tokens,
        soft_limit_ratio=_soft_limit
    )
    if not chapter_mode:
        return chunker.chunk_text(text)

    from src.core.chunking.chapter_detector import find_chapter_ranges

    paragraphs = chunker.split_into_paragraphs(text)
    chapter_ranges = find_chapter_ranges(paragraphs)
    chunks: List[Dict[str, str]] = []

    for chapter_index, chapter_range in enumerate(chapter_ranges):
        chapter_text = "\n\n".join(
            paragraphs[chapter_range.start:chapter_range.end]
        )
        chapter_chunks = chunker.chunk_text(
            chapter_text,
            glue_decorative_separators=True,
        )
        for chunk_index, chunk in enumerate(chapter_chunks):
            chunk.update({
                "chapter_index": chapter_index,
                "chapter_title": chapter_range.title,
                "chunk_in_chapter": chunk_index,
                "chunks_in_chapter": len(chapter_chunks),
            })
        chunks.extend(chapter_chunks)

    return chunks
