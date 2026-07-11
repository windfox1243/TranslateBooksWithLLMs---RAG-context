"""DOCX refine-only mode.

Converts an already-translated DOCX to HTML, refines, then writes a new
DOCX. Reuses the EPUB tag-preservation + chunking machinery. No resume
support in v1.
"""

import os
import tempfile
from typing import Optional, Callable, Dict, Any

from src.config import DEFAULT_MODEL, API_ENDPOINT, MAX_TOKENS_PER_CHUNK
from src.core.epub.xhtml_translator import (
    _create_chunks,
    _escape_stray_angle_brackets,
    _refine_epub_chunks,
)
from src.core.epub.container import TranslationContainer
from src.core.docx.converter import DocxHtmlConverter
from .client_setup import build_refine_client
from .epub_refiner import _globalize_chunk_text


async def refine_docx_file(
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
    refinement_original_path: Optional[str] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated DOCX file."""
    from src.utils.relationship_sync import (
        attach_relationship_context_to_prompt_options,
    )

    prompt_options = attach_relationship_context_to_prompt_options(
        prompt_options,
        translation_id=translation_id or "",
        db=getattr(checkpoint_manager, "db", None) if checkpoint_manager else None,
        target_language=target_language,
        log_callback=log_callback,
    )
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input DOCX file '{input_filepath}' not found."
        if log_callback:
            log_callback("docx_input_file_not_found", err_msg)
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
        if log_callback:
            log_callback("docx_refine_start", "✨ Starting DOCX refine pass...")

        converter = DocxHtmlConverter()
        container = TranslationContainer()
        tag_preserver = container.tag_preserver

        html_content, metadata = converter.to_html(input_filepath)
        if log_callback:
            log_callback("docx_html_extracted",
                         f"Extracted {len(html_content)} chars HTML from DOCX")

        text_with_placeholders, tag_map = tag_preserver.preserve_tags(html_content)
        placeholder_format = (
            tag_preserver.placeholder_format.prefix,
            tag_preserver.placeholder_format.suffix,
        )
        if log_callback:
            log_callback("docx_tags_preserved",
                         f"Preserved {len(tag_map)} tag groups")

        chunks = _create_chunks(
            text_with_placeholders,
            tag_map,
            max_tokens_per_chunk,
            log_callback,
            container,
            chapter_mode=bool((prompt_options or {}).get("chapter_mode")),
        )
        if not chunks:
            if log_callback:
                log_callback("docx_no_chunks", "Nothing to refine in this DOCX")
            return True

        if log_callback:
            log_callback("docx_chunks_created", f"Created {len(chunks)} chunks")

        if stats_callback:
            stats_callback({'total_chunks': len(chunks), 'completed_chunks': 0, 'failed_chunks': 0})

        db_chunks = []
        if checkpoint_manager and translation_id:
            db_chunks = checkpoint_manager.db.get_chunks(translation_id) or []
        checkpoint_sources = [
            str(row.get("original_text") or "")
            for row in sorted(
                db_chunks,
                key=lambda item: item.get("chunk_index", -1),
            )
            if row.get("status") == "completed"
        ]
        if checkpoint_sources:
            prompt_options["_refinement_source_queue"] = checkpoint_sources
            prompt_options["_refinement_source_cursor"] = 0
        elif refinement_original_path:
            try:
                source_html, _source_metadata = converter.to_html(
                    refinement_original_path
                )
                source_container = TranslationContainer()
                source_preserver = source_container.tag_preserver
                source_text, source_map = source_preserver.preserve_tags(source_html)
                source_format = (
                    source_preserver.placeholder_format.prefix,
                    source_preserver.placeholder_format.suffix,
                )
                source_chunks = _create_chunks(
                    source_text,
                    source_map,
                    max_tokens_per_chunk,
                    None,
                    source_container,
                    chapter_mode=bool((prompt_options or {}).get("chapter_mode")),
                )
                if len(source_chunks) != len(chunks):
                    raise ValueError("paired DOCX structural unit alignment differs")
                prompt_options["_refinement_source_queue"] = [
                    _globalize_chunk_text(chunk, source_format)
                    for chunk in source_chunks
                ]
                prompt_options["_refinement_source_cursor"] = 0
                if log_callback:
                    log_callback(
                        "refinement_source_aligned",
                        f"Aligned {len(source_chunks)} paired DOCX source unit(s).",
                    )
            except (OSError, ValueError) as exc:
                if log_callback:
                    log_callback(
                        "refinement_source_alignment_failed",
                        f"Could not align paired DOCX source: {exc}",
                    )
                return False

        draft_globalized = [
            _globalize_chunk_text(chunk, placeholder_format)
            for chunk in chunks
        ]

        from src.utils.novel_context import (
            RefinementContextTracker,
            map_dialogue_attributions_for_refinement,
            map_context_snapshots_for_refinement,
        )
        historical_contexts = map_context_snapshots_for_refinement(
            len(chunks),
            db_chunks,
            (prompt_options or {}).get('novel_context', ''),
            refinement_units=draft_globalized,
        )
        historical_dialogue_attributions = (
            map_dialogue_attributions_for_refinement(
                len(chunks),
                db_chunks,
            )
        )
        context_tracker = RefinementContextTracker(
            prompt_options=prompt_options or {},
            historical_contexts=historical_contexts,
            historical_dialogue_attributions=historical_dialogue_attributions,
            log_callback=log_callback,
        )

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
            stats_callback=stats_callback,
            context_tracker=context_tracker,
            check_interruption_callback=check_interruption_callback,
        )

        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback("docx_refine_interrupted",
                             "Refinement interrupted before reconstruction")
            return False

        full_text = ''.join(refined_chunks)
        full_text = _escape_stray_angle_brackets(full_text)
        final_html = tag_preserver.restore_tags(full_text, tag_map)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
        try:
            converter.from_html(final_html, metadata, tmp_path)
            with open(tmp_path, 'rb') as src, open(output_filepath, 'wb') as dst:
                dst.write(src.read())
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if log_callback:
            log_callback("docx_refine_done",
                         f"✅ DOCX refine complete, saved: {output_filepath}")
        return True
    finally:
        if llm_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass
