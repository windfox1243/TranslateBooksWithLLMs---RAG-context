"""
Plain-text translation pipeline used by Plain Text Mode.

Skips placeholder preservation and HTML chunking entirely. Paragraphs are
grouped into token-budgeted segments that remember which source paragraph
indices they cover, translated with has_placeholders=False, then written back
to those exact indices. Empty source paragraphs (image-only blocks) are never
sent to the LLM and keep their slot; a paragraph larger than the token budget
is split into sentence pieces that all collapse back into its single slot
(issue #203: count-only realignment shifted every paragraph after an empty or
oversized block).

Used by the EPUB and DOCX adapters when prompt_options['plain_text_mode'] is True.
"""
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.chunking.token_chunker import TokenChunker
from src.core.translator import generate_translation_request
from src.core.post_processor import clean_translated_text
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.common.parallel import iter_ordered_concurrent
from src.core.llm.exceptions import RateLimitError


PARAGRAPH_SEPARATOR = "\n\n"
_RESPLIT_REGEX = re.compile(r"\n{2,}")
_MARKUP_TAG_REGEX = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*?)?/?>")


def strip_hallucinated_markup(translated: str, source: str) -> str:
    """Remove HTML-like tags the model invented in Plain Text Mode.

    Plain Text Mode never sends markup to the LLM, so a tag in the output is
    model noise (e.g. small models wrap ordinals or footnote numbers in
    <sup>...</sup>). Only the tags are dropped; their inner text is kept.
    Chunks whose source legitimately contains '<' (code samples inside <pre>
    blocks) are left untouched to avoid damaging real content.
    """
    if "<" not in translated or "<" in source:
        return translated
    return _MARKUP_TAG_REGEX.sub("", translated)


def _split_translated_back_to_paragraphs(translated_text: str) -> List[str]:
    """Split a translated blob into paragraphs (tolerates 2+ newlines)."""
    return [p.strip() for p in _RESPLIT_REGEX.split(translated_text) if p.strip()]


def _reconcile_paragraph_counts(
    translated_paragraphs: List[str],
    expected_count: int,
) -> List[str]:
    """
    Best-effort alignment when the LLM merged or split paragraphs inside one
    segment. The blast radius is the segment, never the whole document.

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


def build_plain_segments(
    paragraphs: List[str],
    max_tokens_per_chunk: int,
    paragraph_kinds: Optional[List[str]] = None,
    chapter_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Group source paragraphs into translation segments that track their indices.

    Each segment is {'indices': [int, ...], 'text': str, 'partial': bool}:
    - whole-paragraph segments cover consecutive non-empty paragraphs joined
      with PARAGRAPH_SEPARATOR ('partial' False, one index per paragraph);
    - an oversized paragraph yields several sentence-piece segments that share
      the same single index ('partial' True).

    Empty/whitespace-only paragraphs are skipped here and restored by index at
    reassembly time.
    """
    from src.core.chunking.chapter_detector import ChapterRange, find_chapter_ranges

    chunker = TokenChunker(max_tokens=max_tokens_per_chunk)
    sep_tokens = chunker.count_tokens(PARAGRAPH_SEPARATOR)
    chapter_ranges = (
        find_chapter_ranges(paragraphs, paragraph_kinds)
        if chapter_mode
        else []
    )
    if not chapter_ranges:
        chapter_ranges = [ChapterRange(0, len(paragraphs))]

    chapter_by_index: Dict[int, Tuple[int, str]] = {}
    chapter_starts = set()
    for chapter_index, chapter_range in enumerate(chapter_ranges):
        chapter_starts.add(chapter_range.start)
        for paragraph_index in range(chapter_range.start, chapter_range.end):
            chapter_by_index[paragraph_index] = (
                chapter_index,
                chapter_range.title,
            )

    segments: List[Dict[str, Any]] = []
    cur_indices: List[int] = []
    cur_texts: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur_indices, cur_texts, cur_tokens
        if cur_indices:
            chapter_index, chapter_title = chapter_by_index.get(
                cur_indices[0], (0, "")
            )
            segments.append({
                'indices': cur_indices,
                'text': PARAGRAPH_SEPARATOR.join(cur_texts),
                'partial': False,
                'chapter_index': chapter_index,
                'chapter_title': chapter_title,
            })
            cur_indices, cur_texts, cur_tokens = [], [], 0

    for idx, paragraph in enumerate(paragraphs):
        if chapter_mode and idx in chapter_starts and cur_indices:
            flush()

        text = paragraph or ""
        if not text.strip():
            continue

        tokens = chunker.count_tokens(text)

        if tokens > chunker.max_tokens:
            flush()
            sentences = chunker.split_paragraph_into_sentences(text)
            if len(sentences) > 1:
                pieces = chunker._chunk_units(sentences, separator=" ")
            else:
                pieces = [text]
            chapter_index, chapter_title = chapter_by_index.get(idx, (0, ""))
            for piece in pieces:
                segments.append({
                    'indices': [idx],
                    'text': piece,
                    'partial': True,
                    'chapter_index': chapter_index,
                    'chapter_title': chapter_title,
                })
            continue

        potential = cur_tokens + tokens + (sep_tokens if cur_indices else 0)
        if cur_indices and potential > chunker.max_tokens:
            flush()
        cur_indices.append(idx)
        cur_texts.append(text)
        cur_tokens = cur_tokens + tokens + (sep_tokens if len(cur_indices) > 1 else 0)

    flush()
    return segments


def _reassemble(
    segments: List[Dict[str, Any]],
    translated_parts: List[str],
    source_paragraphs: List[str],
) -> List[str]:
    """
    Write each segment's translation back to the source indices it covers.

    Empty source slots keep their original (empty) value; pieces of an
    oversized paragraph are concatenated in order into its single slot.
    """
    out: List[Optional[str]] = [None] * len(source_paragraphs)
    partial_pieces: Dict[int, List[str]] = {}

    for segment, translated in zip(segments, translated_parts):
        text = translated or ""
        if segment['partial']:
            partial_pieces.setdefault(segment['indices'][0], []).append(text.strip())
        else:
            parts = _split_translated_back_to_paragraphs(text)
            parts = _reconcile_paragraph_counts(parts, len(segment['indices']))
            for k, idx in enumerate(segment['indices']):
                out[idx] = parts[k]

    for idx, pieces in partial_pieces.items():
        out[idx] = " ".join(p for p in pieces if p)

    return [
        slot if slot is not None else source_paragraphs[i]
        for i, slot in enumerate(out)
    ]


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
    parallel_workers: int = 1,
    checkpoint_manager: Optional[Any] = None,
    translation_id: Optional[str] = None,
    global_chunk_offset: int = 0,
    paragraph_kinds: Optional[List[str]] = None,
    continuation_base_id: Optional[str] = None,
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
        parallel_workers: number of chunks translated concurrently (already
            resolved against the provider by the caller). When 1, behavior is
            identical to the legacy sequential loop, including previous-chunk
            context chaining; > 1 drops that chaining.
        checkpoint_manager, translation_id: optional persistence for context
            snapshots in EPUB/DOCX Plain Text Mode.
        global_chunk_offset: index offset used when an EPUB has several files.

    Returns:
        (translated_paragraphs, stats, was_interrupted)
    """
    stats = TranslationMetrics()

    source = list(paragraphs)
    if not source or all(not (p or "").strip() for p in source):
        if stats_callback:
            stats_callback(stats.to_dict())
        return source, stats, False

    chapter_mode = bool((prompt_options or {}).get('chapter_mode'))
    segments = build_plain_segments(
        source,
        max_tokens_per_chunk,
        paragraph_kinds=paragraph_kinds,
        chapter_mode=chapter_mode,
    )

    if chapter_mode and log_callback:
        chapter_count = len({
            segment.get('chapter_index', 0) for segment in segments
        })
        log_callback(
            "chapter_mode_ready",
            f"Chapter-aware mode prepared {chapter_count} chapter(s) as "
            f"{len(segments)} translation unit(s) with "
            f"{max_tokens_per_chunk} source tokens per unit.",
        )
    elif log_callback:
        log_callback(
            "plain_text_chunks_created",
            f"Plain Text Mode created {len(segments)} translation unit(s) "
            f"with {max_tokens_per_chunk} source tokens per unit.",
        )

    # Chunk dicts mirror split_text_into_chunks() output; context comes from
    # the neighboring segments.
    chunks: List[Dict[str, str]] = []
    for i, segment in enumerate(segments):
        same_previous_chapter = (
            i > 0
            and segments[i - 1].get('chapter_index')
            == segment.get('chapter_index')
        )
        same_next_chapter = (
            i < len(segments) - 1
            and segments[i + 1].get('chapter_index')
            == segment.get('chapter_index')
        )
        if i > 0 and (not chapter_mode or same_previous_chapter):
            context_before = segments[i - 1]['text'].split(PARAGRAPH_SEPARATOR)[-1]
        else:
            context_before = ""
        if i < len(segments) - 1 and (not chapter_mode or same_next_chapter):
            context_after = segments[i + 1]['text'].split(PARAGRAPH_SEPARATOR)[0]
        else:
            context_after = ""
        chunks.append({
            'context_before': context_before,
            'main_content': segment['text'],
            'context_after': context_after,
            'chapter_index': segment.get('chapter_index', 0),
            'chapter_title': segment.get('chapter_title', ''),
        })

    stats.total_chunks = len(chunks)
    if stats_callback:
        stats_callback(stats.to_dict())

    novel_context_file = prompt_options.get('novel_context_file') if prompt_options else None
    auto_update_context = prompt_options.get('auto_update_context', False) if prompt_options else False

    if prompt_options is None:
        prompt_options = {}

    context_session = None
    checkpoint_context_data_by_index: Dict[int, Dict] = {}
    continuation_reused_indices = set()
    continuation_context_seed = None
    if continuation_base_id and checkpoint_manager and translation_id:
        previous_checkpoint = checkpoint_manager.load_checkpoint(
            continuation_base_id
        )
        previous_chunks = (
            previous_checkpoint.get('chunks', [])
            if previous_checkpoint
            else []
        )
        if previous_chunks:
            from src.core.continuation import (
                latest_context_seed,
                seed_matching_prefix,
            )
            prefix = seed_matching_prefix(
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                previous_chunks=previous_chunks,
                new_source_units=[
                    chunk.get('main_content', '') for chunk in chunks
                ],
                total_units=global_chunk_offset + len(chunks),
                offset=global_chunk_offset,
                log_callback=log_callback,
                label="unit",
            )
            continuation_reused_indices = set(range(prefix))
            continuation_context_seed = latest_context_seed(previous_chunks)

    if checkpoint_manager and translation_id and hasattr(checkpoint_manager, "db"):
        for row in checkpoint_manager.db.get_chunks(translation_id) or []:
            row_index = row.get("chunk_index")
            row_data = row.get("chunk_data") or {}
            if (
                isinstance(row_index, int)
                and global_chunk_offset
                <= row_index
                < global_chunk_offset + len(chunks)
                and row.get("status") in ("completed", "partial", "failed")
                and row_data.get("context_snapshot")
            ):
                checkpoint_context_data_by_index[
                    row_index - global_chunk_offset
                ] = dict(row_data)

    if novel_context_file or auto_update_context:
        from src.config import NOVEL_CONTEXTS_DIR
        from src.utils.novel_context import (
            open_novel_context_session,
            should_update_novel_context_for_index,
        )
        try:
            resume_snapshot = None
            resume_dialogue_state = None
            resume_dialogue_scene_key = None
            used_continuation_context_seed = False
            if (
                checkpoint_manager
                and translation_id
                and hasattr(checkpoint_manager, "db")
                and global_chunk_offset > 0
            ):
                previous_rows = [
                    row
                    for row in (
                        checkpoint_manager.db.get_chunks(translation_id) or []
                    )
                    if row.get("status") in ("completed", "partial", "failed")
                    and row.get("chunk_index", -1) < global_chunk_offset
                    and (row.get("chunk_data") or {}).get("context_snapshot")
                ]
                if previous_rows:
                    previous_row = max(
                        previous_rows,
                        key=lambda row: row.get("chunk_index", -1),
                    )
                    previous_data = previous_row.get("chunk_data") or {}
                    resume_snapshot = previous_data.get("context_snapshot")
                    resume_dialogue_state = (
                        (
                            previous_data.get("dialogue_attribution") or {}
                        ).get("state_after")
                    )
                    resume_dialogue_scene_key = (
                        previous_data.get("dialogue_attribution") or {}
                    ).get("scene_key")
            if not resume_snapshot and continuation_context_seed:
                resume_snapshot = continuation_context_seed.get(
                    "context_snapshot"
                )
                resume_dialogue_state = continuation_context_seed.get(
                    "dialogue_state"
                )
                resume_dialogue_scene_key = continuation_context_seed.get(
                    "dialogue_scene_key"
                )
                used_continuation_context_seed = True
            context_session = open_novel_context_session(
                prompt_options=prompt_options,
                novel_contexts_dir=NOVEL_CONTEXTS_DIR,
                input_filename=prompt_options.get('input_filename', ''),
                fallback_name="plaintext",
                resume_snapshot=resume_snapshot,
                resume_dialogue_state=resume_dialogue_state,
                resume_dialogue_scene_key=resume_dialogue_scene_key,
                log_callback=log_callback,
            )
            if (
                used_continuation_context_seed
                and context_session
                and log_callback
            ):
                log_callback(
                    "continuation_context_seed",
                    "Add New Content: continuing context from previous "
                    "job chunk "
                    f"{continuation_context_seed.get('chunk_index')} "
                    "snapshot.",
                )
        except Exception as e:
            if log_callback:
                log_callback("novel_context_error", f"Error loading novel context '{novel_context_file}': {str(e)}")

    workers = max(1, int(parallel_workers))
    if auto_update_context and context_session:
        if workers > 1 and log_callback:
            log_callback(
                "novel_context_workers_override",
                "Warning: Auto-updating novel context requires sequential translation. Forcing parallel workers to 1.",
            )
        workers = 1
    sequential = workers == 1

    # Index-addressed results so out-of-order completion still reassembles in
    # source order.
    translated_parts: List[Optional[str]] = [None] * len(chunks)
    if continuation_reused_indices and checkpoint_manager and translation_id:
        seeded_rows = {
            row.get("chunk_index"): row
            for row in checkpoint_manager.db.get_chunks(translation_id) or []
        }
        for local_index in continuation_reused_indices:
            seeded = seeded_rows.get(global_chunk_offset + local_index) or {}
            translated_parts[local_index] = seeded.get("translated_text")
            if seeded.get("chunk_data"):
                checkpoint_context_data_by_index[local_index] = dict(
                    seeded.get("chunk_data") or {}
                )
        stats.processed_chunks = len(continuation_reused_indices)
        if stats_callback:
            stats_callback(stats.to_dict())
    previous_translation_context = ""
    failed_indices = set()
    reused_context_data_by_index: Dict[int, Dict] = {}

    async def _translate_chunk(i, analyze_context=True):
        """Translate one chunk. Reads previous_translation_context only in
        sequential mode (parallel runs have no stable previous chunk)."""
        if log_callback:
            log_callback("unit_start", f"Translating unit {i+1}/{len(chunks)}")
        main_content = chunks[i].get('main_content', '')
        if not main_content.strip():
            return ('empty', main_content)

        if (
            analyze_context
            and auto_update_context
            and context_session
            and should_update_novel_context_for_index(i, prompt_options)
        ):
            reused_context_data_by_index.pop(i, None)
            if log_callback:
                log_callback(
                    "novel_context_updating",
                    f"Analyzing source context for chunk {i+1} before translation...",
                )
            try:
                change_logs = await context_session.analyze_source(
                    llm_client=llm_client,
                    model_name=model_name,
                    source_chunk=main_content,
                    source_language=source_language,
                    target_language=target_language,
                    chunk_index=i + 1,
                    total_chunks=len(chunks),
                    scene_key=chunks[i].get("chapter_index"),
                )
                if log_callback:
                    log_callback(
                        "novel_context_updated",
                        f"Novel context prepared for chunk {i+1}.",
                    )
                    for change_log in change_logs:
                        log_callback("novel_context_log", change_log)
                    log_callback(
                        "novel_context_state",
                        "Context updated",
                        {
                            "type": "novel_context_state",
                            "content": context_session.content,
                            "filename": context_session.path.name,
                        },
                    )
            except Exception as e:
                if log_callback:
                    log_callback(
                        "novel_context_update_failed",
                        f"Failed to prepare novel context: {str(e)}",
                    )
        elif i in checkpoint_context_data_by_index:
            reused_context_data_by_index[i] = dict(
                checkpoint_context_data_by_index[i]
            )

        translated = await generate_translation_request(
            main_content=main_content,
            context_before=chunks[i].get('context_before', ''),
            context_after=chunks[i].get('context_after', ''),
            previous_translation_context=(
                previous_translation_context
                if (
                    sequential
                    and (
                        not chapter_mode
                        or i == 0
                        or chunks[i - 1].get('chapter_index')
                        == chunks[i].get('chapter_index')
                    )
                )
                else ""
            ),
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
        return ('done', translated)

    def _fill_remaining_with_source():
        for j in range(len(chunks)):
            if translated_parts[j] is None:
                translated_parts[j] = chunks[j].get('main_content', '')

    def _save_chunk_checkpoint(i, chunk_succeeded):
        if not (
            checkpoint_manager
            and translation_id
            and hasattr(checkpoint_manager, 'db')
        ):
            return
        chunk_data = dict(reused_context_data_by_index.get(i) or {})
        if not chunk_data and context_session:
            chunk_data['context_snapshot'] = context_session.snapshot()
            chunk_data['dialogue_attribution'] = (
                context_session.dialogue_attribution
            )
        if chunk_data.get('context_snapshot'):
            checkpoint_context_data_by_index[i] = dict(chunk_data)
        checkpoint_manager.db.save_chunk(
            translation_id=translation_id,
            chunk_index=global_chunk_offset + i,
            original_text=chunks[i].get('main_content', ''),
            translated_text=translated_parts[i],
            chunk_data=chunk_data,
            status='completed' if chunk_succeeded else 'partial',
        )

    pending = [
        index for index in range(len(chunks))
        if index not in continuation_reused_indices
    ]
    rate_limit_error = None
    processed = 0

    # Continuous concurrency with in-order delivery (see iter_ordered_concurrent).
    async for i, result in iter_ordered_concurrent(
        pending, workers, _translate_chunk, check_interruption_callback
    ):
        main_content = chunks[i].get('main_content', '')
        chunk_succeeded = False

        if isinstance(result, RateLimitError):
            rate_limit_error = result
            break

        if isinstance(result, Exception):
            if log_callback:
                log_callback(
                    "plain_text_chunk_failed",
                    f"Chunk {i + 1}/{len(chunks)} failed ({result}) - keeping original text"
                )
            translated_parts[i] = main_content
            failed_indices.add(i)
            stats.failed_chunks = len(failed_indices)
        else:
            kind, value = result
            if kind == 'empty':
                translated_parts[i] = value
                stats.successful_first_try += 1
                chunk_succeeded = True
            elif value is None:
                if log_callback:
                    log_callback(
                        "plain_text_chunk_failed",
                        f"Chunk {i + 1}/{len(chunks)} failed - keeping original text"
                    )
                translated_parts[i] = main_content
                failed_indices.add(i)
                stats.failed_chunks = len(failed_indices)
            else:
                cleaned = clean_translated_text(value)
                cleaned = strip_hallucinated_markup(
                    cleaned, chunks[i].get('main_content', ''))
                translated_parts[i] = cleaned
                stats.successful_first_try += 1
                chunk_succeeded = True
                if sequential:
                    words = cleaned.split()
                    previous_translation_context = (
                        " ".join(words[-25:]) if len(words) > 25 else cleaned
                    )

        _save_chunk_checkpoint(i, chunk_succeeded)

        stats.record_processed()
        if stats_callback:
            stats_callback(stats.to_dict())
        processed += 1

    if rate_limit_error is not None:
        # Keep source text for everything not yet translated, then propagate to
        # trigger the caller's pause/resume handling.
        _fill_remaining_with_source()
        raise rate_limit_error

    # Interruption: the scheduler stopped launching new chunks; keep source text
    # for the uncommitted tail and report the interruption.
    if processed < len(chunks) and check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback(
                "plain_text_translation_interrupted",
                f"⏸️ Plain-text translation interrupted at chunk {processed + 1}/{len(chunks)}"
            )
        _fill_remaining_with_source()
        safe_parts = [p if p is not None else "" for p in translated_parts]
        return _reassemble(segments, safe_parts, source), stats, True

    if failed_indices:
        retry_targets = sorted(failed_indices)
        if log_callback:
            log_callback(
                "failed_chunk_retry_start",
                f"Retrying {len(retry_targets)} failed plain-text chunk(s) before final output...",
            )
        for i in retry_targets:
            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback(
                        "plain_text_translation_interrupted",
                        f"⏸️ Plain-text translation interrupted before retrying chunk {i + 1}/{len(chunks)}",
                    )
                _fill_remaining_with_source()
                safe_parts = [p if p is not None else "" for p in translated_parts]
                return _reassemble(segments, safe_parts, source), stats, True

            chunk_succeeded = False
            main_content = chunks[i].get('main_content', '')
            try:
                retry_result = await _translate_chunk(i, analyze_context=False)
            except RateLimitError:
                _fill_remaining_with_source()
                raise
            except Exception as exc:
                if log_callback:
                    log_callback(
                        "plain_text_chunk_failed",
                        f"Retry failed for chunk {i + 1}/{len(chunks)} ({exc}) - keeping original text",
                    )
                translated_parts[i] = main_content
                _save_chunk_checkpoint(i, False)
                continue

            kind, value = retry_result
            if kind == 'empty':
                translated_parts[i] = value
                chunk_succeeded = True
                failed_indices.discard(i)
                stats.failed_chunks = len(failed_indices)
            elif value is not None:
                cleaned = clean_translated_text(value)
                cleaned = strip_hallucinated_markup(
                    cleaned, chunks[i].get('main_content', ''))
                translated_parts[i] = cleaned
                chunk_succeeded = True
                failed_indices.discard(i)
                stats.failed_chunks = len(failed_indices)
                if log_callback:
                    log_callback(
                        "failed_chunk_retry_success",
                        f"Failed plain-text chunk {i + 1}/{len(chunks)} translated successfully on retry.",
                    )
            else:
                translated_parts[i] = main_content

            _save_chunk_checkpoint(i, chunk_succeeded)
            if stats_callback:
                stats_callback(stats.to_dict())

    # Any None left (shouldn't happen) falls back to empty string.
    safe_parts = [p if p is not None else "" for p in translated_parts]
    return _reassemble(segments, safe_parts, source), stats, False
