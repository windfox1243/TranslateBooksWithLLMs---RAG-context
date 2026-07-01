"""EPUB refine-only mode.

Walks each XHTML content file of an already-translated EPUB, refines its
body in place, and repackages. Interruption stops cleanly between chunks
but does not persist partial state (no resume support in v1).
"""

import os
import tempfile
from typing import Optional, Callable, Dict, Any, List, Tuple
from lxml import etree

from src.config import (
    DEFAULT_MODEL, API_ENDPOINT, MAX_TOKENS_PER_CHUNK, THINKING_MODELS,
    ADAPTIVE_CONTEXT_INITIAL_THINKING,
)
from src.core.epub.translator import (
    _extract_epub, _parse_epub_manifest, _create_llm_client,
    _create_context_manager, _repackage_epub,
)
from src.core.epub.xhtml_translator import (
    _setup_translation, _preserve_tags, _create_chunks,
    _replace_body, _escape_stray_angle_brackets, _refine_epub_chunks,
)
from src.core.epub.container import TranslationContainer
from src.core.context_optimizer import INITIAL_CONTEXT_SIZE
from .client_setup import build_refine_client

_REFINE_AFTER_SPINE_UNIT_TOKEN_BUDGET = 10_000_000


def _refine_after_uses_spine_units(prompt_options: Optional[Dict]) -> bool:
    """Keep EPUB refine-after aligned to spine/content-file boundaries."""
    options = prompt_options or {}
    return bool(options.get("_refine_after") and options.get("chapter_mode"))


def _refine_chunking_options(
    prompt_options: Optional[Dict],
    max_tokens_per_chunk: int,
) -> Tuple[int, bool]:
    if _refine_after_uses_spine_units(prompt_options):
        return (
            max(max_tokens_per_chunk, _REFINE_AFTER_SPINE_UNIT_TOKEN_BUDGET),
            False,
        )
    return max_tokens_per_chunk, bool((prompt_options or {}).get("chapter_mode"))


def _refine_chunking_note(prompt_options: Optional[Dict]) -> Optional[str]:
    if _refine_after_uses_spine_units(prompt_options):
        return "EPUB spine-file refinement unit(s)"
    return None


def _globalize_chunk_text(
    chunk: Dict,
    placeholder_format: Tuple[str, str],
) -> str:
    """Convert a chunk's placeholders from local to global indices.

    `_refine_epub_chunks` expects globally-numbered placeholders because it
    re-localizes internally before sending to the LLM. In translate-mode the
    chunks already carry global indices; in refine-only mode they come fresh
    from HtmlChunker with local indices, so we re-globalize here.
    """
    text = chunk['text']
    global_indices = chunk.get('global_indices', [])
    if not global_indices:
        return text

    prefix, suffix = placeholder_format
    for local_idx, global_idx in enumerate(global_indices):
        local_ph = f"{prefix}{local_idx}{suffix}"
        text = text.replace(local_ph, f"__TEMP_GLOBAL_{global_idx}__")
    for global_idx in global_indices:
        text = text.replace(f"__TEMP_GLOBAL_{global_idx}__",
                            f"{prefix}{global_idx}{suffix}")
    return text


async def _refine_one_xhtml(
    doc_root: etree._Element,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable],
    context_manager: Optional[Any],
    prompt_options: Optional[Dict],
    check_interruption_callback: Optional[Callable],
    container: Optional[TranslationContainer] = None,
    context_tracker: Optional[Any] = None,
    stats_callback: Optional[Callable] = None,
) -> bool:
    """Refine a single parsed XHTML document in place."""
    body_html, body_element, tag_preserver = _setup_translation(
        doc_root, log_callback, container
    )
    if not body_html or body_element is None:
        if log_callback:
            log_callback("no_body", "No <body> element found")
        return False

    text_with_placeholders, global_tag_map, placeholder_format = _preserve_tags(
        body_html, tag_preserver, log_callback, protect_technical=True
    )
    chunk_budget, chunk_chapter_mode = _refine_chunking_options(
        prompt_options,
        max_tokens_per_chunk,
    )

    chunks = _create_chunks(
        text_with_placeholders, global_tag_map, chunk_budget,
        log_callback, container,
        chapter_mode=chunk_chapter_mode,
        chunking_note=_refine_chunking_note(prompt_options),
    )

    if not chunks:
        if log_callback:
            log_callback("no_chunks", "No translatable chunks in this XHTML, skipping")
        return True

    draft_globalized = [_globalize_chunk_text(c, placeholder_format) for c in chunks]

    refined_chunks = await _refine_epub_chunks(
        translated_chunks=draft_globalized,
        chunks=chunks,
        target_language=target_language,
        model_name=model_name,
        llm_client=llm_client,
        context_manager=context_manager,
        placeholder_format=placeholder_format,
        log_callback=log_callback,
        prompt_options=prompt_options,
        context_tracker=context_tracker,
        check_interruption_callback=check_interruption_callback,
        stats_callback=stats_callback,
    )

    if check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback("refine_interrupted",
                         "Refinement interrupted before reconstruction")
        return False

    full_text = ''.join(refined_chunks)
    full_text = _escape_stray_angle_brackets(full_text)
    final_html = tag_preserver.restore_tags(full_text, global_tag_map)

    return _replace_body(body_element, final_html, log_callback)


async def refine_epub_file(
    input_filepath: str,
    output_filepath: str,
    target_language: str,
    model_name: str = DEFAULT_MODEL,
    cli_api_endpoint: str = API_ENDPOINT,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    llm_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    prompt_options: Optional[Dict] = None,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    checkpoint_manager: Optional[Any] = None,
    translation_id: Optional[str] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated EPUB."""
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input EPUB file '{input_filepath}' not found."
        if log_callback:
            log_callback("epub_input_file_not_found", err_msg)
        return False

    llm_client, context_manager = build_refine_client(
        model_name=model_name,
        llm_provider=llm_provider,
        cli_api_endpoint=cli_api_endpoint,
        auto_adjust_context=auto_adjust_context,
        context_window=context_window,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        log_callback=log_callback,
    )
    if llm_client is None:
        return False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_epub(input_filepath, temp_dir, log_callback)
            manifest_data = _parse_epub_manifest(temp_dir, log_callback)

            content_files: List[str] = manifest_data['content_files']
            opf_dir: str = manifest_data['opf_dir']

            total_files = len(content_files)
            if log_callback:
                log_callback("epub_refine_start",
                             f"✨ Starting EPUB refine pass over {total_files} content files...")
                if _refine_after_uses_spine_units(prompt_options):
                    log_callback(
                        "epub_refine_spine_units",
                        "Chapter-aware refine-after: using EPUB spine files "
                        "as refinement units to preserve translation alignment.",
                    )

            db_chunks = []
            if checkpoint_manager and translation_id:
                db_chunks = checkpoint_manager.db.get_chunks(translation_id) or []

            # Pre-calculate the global refine chunk count so persisted snapshots
            # map consistently across all XHTML files.
            total_refine_chunks = 0
            refinement_units: List[str] = []
            refine_chunk_counts: Dict[str, int] = {}
            pre_container = TranslationContainer()
            for href in content_files:
                file_path = os.path.join(opf_dir, href)
                if not os.path.exists(file_path):
                    continue
                try:
                    parser = etree.XMLParser(recover=True, remove_blank_text=False)
                    tree = etree.parse(file_path, parser)
                    doc_root = tree.getroot()
                    body_html, body_element, tag_preserver = _setup_translation(
                        doc_root, None, pre_container
                    )
                    if body_html and body_element is not None:
                        (
                            text_with_placeholders,
                            global_tag_map,
                            placeholder_format,
                        ) = _preserve_tags(
                            body_html, tag_preserver, None, protect_technical=True
                        )
                        chunk_budget, chunk_chapter_mode = _refine_chunking_options(
                            prompt_options,
                            max_tokens_per_chunk,
                        )
                        chunks = _create_chunks(
                            text_with_placeholders, global_tag_map, chunk_budget,
                            None, pre_container,
                            chapter_mode=chunk_chapter_mode,
                        )
                        total_refine_chunks += len(chunks)
                        refine_chunk_counts[href] = len(chunks)
                        refinement_units.extend(
                            _globalize_chunk_text(chunk, placeholder_format)
                            for chunk in chunks
                        )
                except Exception:
                    pass

            progress_total = total_refine_chunks or total_files
            if stats_callback:
                stats_callback({
                    'total_chunks': progress_total,
                    'completed_chunks': 0,
                    'failed_chunks': 0,
                })

            from src.utils.novel_context import (
                RefinementContextTracker,
                map_dialogue_attributions_for_refinement,
                map_context_snapshots_for_refinement,
            )
            historical_contexts = map_context_snapshots_for_refinement(
                total_refine_chunks,
                db_chunks,
                (prompt_options or {}).get('novel_context', ''),
                refinement_units=refinement_units,
            )
            historical_dialogue_attributions = (
                map_dialogue_attributions_for_refinement(
                    total_refine_chunks,
                    db_chunks,
                )
            )
            context_tracker = RefinementContextTracker(
                prompt_options=prompt_options or {},
                historical_contexts=historical_contexts,
                historical_dialogue_attributions=(
                    historical_dialogue_attributions
                ),
                log_callback=log_callback,
            )

            completed, failed, interrupted = 0, 0, False
            completed_refine_chunks = 0
            for idx, href in enumerate(content_files):
                if check_interruption_callback and check_interruption_callback():
                    if log_callback:
                        log_callback("epub_refine_interrupted",
                                     f"Refinement interrupted at file {idx + 1}/{total_files}")
                    interrupted = True
                    break

                file_path = os.path.join(opf_dir, href)
                if not os.path.exists(file_path):
                    if log_callback:
                        log_callback("epub_refine_missing",
                                     f"⚠️ Content file missing in EPUB: {href}, skipping")
                    failed += 1
                    continue

                if log_callback:
                    log_callback("epub_refine_file",
                                 f"📄 Refining file {idx + 1}/{total_files}: {href}")

                def _file_stats_callback(local_stats, *, base=completed_refine_chunks):
                    if not stats_callback or not total_refine_chunks:
                        return
                    try:
                        local_completed = int(
                            (local_stats or {}).get('completed_chunks', 0)
                        )
                    except (TypeError, ValueError):
                        local_completed = 0
                    stats_callback({
                        'total_chunks': total_refine_chunks,
                        'completed_chunks': min(
                            total_refine_chunks,
                            base + max(0, local_completed),
                        ),
                        'failed_chunks': failed,
                    })

                try:
                    parser = etree.XMLParser(recover=True, remove_blank_text=False)
                    tree = etree.parse(file_path, parser)
                    doc_root = tree.getroot()
                except Exception as e:
                    if log_callback:
                        log_callback("epub_refine_parse_error",
                                     f"⚠️ Could not parse {href}: {e}")
                    failed += 1
                    continue

                ok = await _refine_one_xhtml(
                    doc_root=doc_root,
                    target_language=target_language,
                    model_name=model_name,
                    llm_client=llm_client,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    log_callback=log_callback,
                    context_manager=context_manager,
                    prompt_options=prompt_options,
                    check_interruption_callback=check_interruption_callback,
                    context_tracker=context_tracker,
                    stats_callback=_file_stats_callback,
                )

                if ok:
                    completed_refine_chunks += refine_chunk_counts.get(href, 0)
                    try:
                        tree.write(file_path, xml_declaration=True,
                                   encoding='utf-8', method='xml')
                    except Exception as e:
                        if log_callback:
                            log_callback("epub_refine_write_error",
                                         f"⚠️ Could not write refined XHTML {href}: {e}")
                        failed += 1
                        continue
                    completed += 1
                else:
                    failed += 1
                    completed_refine_chunks += refine_chunk_counts.get(href, 0)

                if stats_callback:
                    if total_refine_chunks:
                        stats_callback({
                            'total_chunks': total_refine_chunks,
                            'completed_chunks': min(
                                total_refine_chunks,
                                completed_refine_chunks,
                            ),
                            'failed_chunks': failed,
                        })
                    else:
                        stats_callback({
                            'total_chunks': total_files,
                            'completed_chunks': completed + failed,
                            'failed_chunks': failed,
                        })

            from src.utils.file_utils import get_partial_output_path
            final_output = (
                get_partial_output_path(output_filepath) if interrupted
                else output_filepath
            )
            _repackage_epub(temp_dir=temp_dir,
                            output_filepath=final_output,
                            log_callback=log_callback)

            if log_callback:
                log_callback("epub_refine_done",
                             f"✅ EPUB refine complete: {completed} files refined, "
                             f"{failed} failed, output: {final_output}")
            return not interrupted and failed == 0
    finally:
        if llm_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass
