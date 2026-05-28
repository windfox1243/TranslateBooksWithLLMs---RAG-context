"""
Plain-text translation pipeline used by Plain Text Mode.

Skips placeholder preservation and HTML chunking entirely. Paragraphs are
joined, chunked by token count, translated with has_placeholders=False, then
re-split on the paragraph separator.

Used by the EPUB and DOCX adapters when prompt_options['plain_text_mode'] is True.
"""
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.text_processor import split_text_into_chunks
from src.core.translator import generate_translation_request
from src.core.post_processor import clean_translated_text
from src.core.epub.translation_metrics import TranslationMetrics


PARAGRAPH_SEPARATOR = "\n\n"
_RESPLIT_REGEX = re.compile(r"\n{2,}")


def _split_translated_back_to_paragraphs(translated_text: str) -> List[str]:
    """Split a translated blob into paragraphs (tolerates 2+ newlines)."""
    return [p.strip() for p in _RESPLIT_REGEX.split(translated_text) if p.strip()]


def _reconcile_paragraph_counts(
    translated_paragraphs: List[str],
    expected_count: int,
) -> List[str]:
    """
    Best-effort alignment when the LLM merged or split paragraphs.

    - translated == expected: return as-is
    - translated < expected: pad with empty strings
    - translated > expected: merge surplus into the last slot
    """
    got = len(translated_paragraphs)
    if got == expected_count:
        return translated_paragraphs
    if got < expected_count:
        return translated_paragraphs + [""] * (expected_count - got)
    head = translated_paragraphs[:expected_count - 1]
    tail = " ".join(translated_paragraphs[expected_count - 1:])
    return head + [tail]


async def translate_paragraphs_plain(
    paragraphs: List[str],
    source_language: str,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    context_manager: Optional[Any] = None,
    check_interruption_callback: Optional[Callable] = None,
    prompt_options: Optional[Dict] = None,
) -> Tuple[List[str], TranslationMetrics, bool]:
    """
    Translate a list of plain-text paragraphs without placeholder preservation.

    Args:
        paragraphs: source paragraphs (one string per block)
        source_language, target_language: language names
        model_name, llm_client: LLM config
        max_tokens_per_chunk: chunking budget
        log_callback, stats_callback: callbacks (stats_callback receives
            file-local stats via TranslationMetrics.to_dict(); callers that
            aggregate across files are responsible for adding their global
            offset to completed_chunks).
        context_manager: AdaptiveContextManager (Ollama)
        check_interruption_callback: returns True to abort
        prompt_options: prompt customization (text_cleanup, glossary, etc.)

    Returns:
        (translated_paragraphs, stats, was_interrupted)
    """
    stats = TranslationMetrics()

    source = list(paragraphs)
    if not source or all(not (p or "").strip() for p in source):
        if stats_callback:
            stats_callback(stats.to_dict())
        return source, stats, False

    full_text = PARAGRAPH_SEPARATOR.join(source)

    chunks = split_text_into_chunks(
        text=full_text,
        max_tokens_per_chunk=max_tokens_per_chunk,
    )

    stats.total_chunks = len(chunks)
    if stats_callback:
        stats_callback(stats.to_dict())

    translated_parts: List[str] = []
    previous_translation_context = ""

    for i, chunk in enumerate(chunks):
        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback(
                    "plain_text_translation_interrupted",
                    f"⏸️ Plain-text translation interrupted at chunk {i + 1}/{len(chunks)}"
                )
            for remaining in chunks[i:]:
                translated_parts.append(remaining.get('main_content', ''))
            return _finalize(translated_parts, source), stats, True

        main_content = chunk.get('main_content', '')
        if not main_content.strip():
            translated_parts.append(main_content)
            stats.successful_first_try += 1
            stats.record_processed()
            if stats_callback:
                stats_callback(stats.to_dict())
            continue

        translated = await generate_translation_request(
            main_content=main_content,
            context_before=chunk.get('context_before', ''),
            context_after=chunk.get('context_after', ''),
            previous_translation_context=previous_translation_context,
            source_language=source_language,
            target_language=target_language,
            model=model_name,
            llm_client=llm_client,
            log_callback=log_callback,
            has_placeholders=False,
            prompt_options=prompt_options,
            context_manager=context_manager,
            placeholder_format=None,
        )

        if translated is None:
            if log_callback:
                log_callback(
                    "plain_text_chunk_failed",
                    f"Chunk {i + 1}/{len(chunks)} failed - keeping original text"
                )
            translated_parts.append(main_content)
            stats.failed_chunks += 1
        else:
            translated = clean_translated_text(translated)
            translated_parts.append(translated)
            stats.successful_first_try += 1
            words = translated.split()
            previous_translation_context = (
                " ".join(words[-25:]) if len(words) > 25 else translated
            )

        stats.record_processed()
        if stats_callback:
            stats_callback(stats.to_dict())

    return _finalize(translated_parts, source), stats, False


def _finalize(translated_parts: List[str], source_paragraphs: List[str]) -> List[str]:
    """Reassemble translated chunks into a paragraph list aligned with the source count."""
    joined = PARAGRAPH_SEPARATOR.join(translated_parts)
    parts = _split_translated_back_to_paragraphs(joined)
    return _reconcile_paragraph_counts(parts, len(source_paragraphs))
