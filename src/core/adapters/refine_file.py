"""Unified refine-only entry point.

Mirrors translate_file() but skips the translation phase: the input file
is assumed to already be in the target language, and only a refinement
pass is applied.
"""

import os
import uuid
from typing import Optional, Callable, Dict, Any

from .exceptions import UnsupportedFormatError
from src.utils.file_detector import detect_file_type, detect_file_type_by_content


async def refine_file(
    input_filepath: str,
    output_filepath: str,
    target_language: str,
    model_name: str,
    llm_provider: str,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    refinement_original_path: Optional[str] = None,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    resume_from_index: int = 0,
    llm_api_endpoint: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: Optional[int] = None,
    auto_adjust_context: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    **additional_config,
) -> bool:
    """Run a refinement-only pass on an already-translated file.

    `target_language` names the language the file is already in: refinement
    is monolingual and does not translate.

    Raises UnsupportedFormatError when the file format cannot be refined.
    """
    if prompt_options is None:
        prompt_options = {}
    else:
        prompt_options = dict(prompt_options)

    from src.core.llm.runtime import build_runtime_spec, create_runtime_client

    credentials = {
        "gemini_api_key": gemini_api_key,
        "openai_api_key": openai_api_key,
        "openrouter_api_key": openrouter_api_key,
        "mistral_api_key": mistral_api_key,
        "deepseek_api_key": deepseek_api_key,
        "poe_api_key": poe_api_key,
        "nim_api_key": nim_api_key,
    }
    editor_provider_value = str(prompt_options.get("editor_provider") or llm_provider)
    editor_model_value = str(prompt_options.get("editor_model") or model_name)
    editor_endpoint = prompt_options.get("editor_api_endpoint")
    if not editor_endpoint and editor_provider_value.casefold() == llm_provider.casefold():
        editor_endpoint = llm_api_endpoint
    editor_spec = build_runtime_spec(
        editor_provider_value,
        editor_model_value,
        api_endpoint=editor_endpoint,
        credentials=credentials,
    )
    editor_client = create_runtime_client(
        editor_spec, context_window=context_window, log_callback=log_callback,
    )
    editor_provider = editor_spec.provider
    editor_model = editor_spec.model
    prompt_options["_editor_llm_client"] = editor_client
    prompt_options.update({
        "editor_provider_resolved": editor_provider,
        "editor_model_resolved": editor_model,
        "llm_provider": llm_provider,
        "model": model_name,
        "translation_id": translation_id,
        "editor_phase": "refinement",
        "jobs_db_path": getattr(
            getattr(checkpoint_manager, "db", None), "db_path", None,
        ),
        "_checkpoint_db": getattr(checkpoint_manager, "db", None),
    })

    # Load novel context if a file is specified
    novel_context_file = prompt_options.get('novel_context_file')
    if novel_context_file:
        from src.config import NOVEL_CONTEXTS_DIR
        from src.utils.novel_context import (
            build_novel_context,
            extract_global_lore,
            load_novel_context,
            resolve_novel_context_path,
        )
        try:
            # For refinement, we always reset the context to the global lore (removing any end-of-run
            # dynamic relationship state), because using the final dynamic state from the end of the book
            # would spoil the relationships for earlier chunks during the refinement pass.
            # The local chunk-specific dynamic context is dynamically resolved and injected per-chunk.
            novel_context_path = resolve_novel_context_path(novel_context_file, NOVEL_CONTEXTS_DIR)
            current_context_content = load_novel_context(novel_context_path.name, novel_context_path.parent)
            global_lore_only = extract_global_lore(current_context_content)
            
            prompt_options['novel_context'] = build_novel_context(
                global_lore_only,
                "",
            )
            if log_callback:
                log_callback("novel_context_state", "Context loaded for refinement (global lore; historical state resolved per unit)", {
                    "type": "novel_context_state", 
                    "content_omitted": True,
                    "content_size": len(prompt_options['novel_context']),
                    "filename": novel_context_path.name,
                    "phase": "refinement",
                    "ephemeral": True,
                })
        except Exception as e:
            if log_callback:
                log_callback("novel_context_error", f"Error loading novel context '{novel_context_file}': {str(e)}")

    # Resolve max_tokens_per_chunk lazily so a reload_config() between calls is
    # honoured for subsequent runs (the .env value can change at runtime via
    # the /api/settings endpoint).
    if max_tokens_per_chunk is None:
        from src.config import MAX_TOKENS_PER_CHUNK as _DEFAULT_MAX_TOKENS
        max_tokens_per_chunk = _DEFAULT_MAX_TOKENS

    _, ext = os.path.splitext(input_filepath.lower())
    try:
        detected_type = detect_file_type(input_filepath)
    except ValueError:
        detected_type = detect_file_type_by_content(input_filepath)
        if detected_type is None:
            raise UnsupportedFormatError(
                f"Cannot determine file type for: {ext}. "
                f"The file does not appear to be a supported format."
            )

    if log_callback and detected_type != ext.lstrip('.'):
        log_callback("file_type_detected",
                     f"📄 File with extension '{ext}' detected as '{detected_type.upper()}' format")
    prompt_options["file_type"] = detected_type

    refinement_pass_id = ""
    if checkpoint_manager is not None and translation_id:
        checkpoint = checkpoint_manager.load_checkpoint(translation_id) or {}
        expected_units = len([
            row for row in checkpoint.get("chunks", [])
            if row.get("status") in {"completed", "partial"}
            and row.get("translated_text") is not None
        ])
        source_mode = "checkpoint" if expected_units else "monolingual"
        alignment_mode = "exact" if expected_units else "unmapped"
        refinement_pass_id = f"ref_{uuid.uuid4().hex}"
        if checkpoint_manager.db.create_refinement_pass(
            refinement_pass_id,
            translation_id,
            context_revision=int((checkpoint.get("job") or {}).get("config", {}).get("context_revision", 0) or 0),
            source_mode=source_mode,
            alignment_mode=alignment_mode,
            expected_units=expected_units,
        ):
            prompt_options["_refinement_pass_id"] = refinement_pass_id
            prompt_options["_refinement_expected_units"] = expected_units

    async def _managed_refinement(coroutine):
        try:
            result = await coroutine
        except Exception as exc:
            if refinement_pass_id:
                checkpoint_manager.db.finish_refinement_pass(
                    refinement_pass_id, successful=False, error=type(exc).__name__,
                )
            raise
        if refinement_pass_id:
            checkpoint_manager.db.finish_refinement_pass(
                refinement_pass_id, successful=bool(result),
            )
        return result

    if detected_type == 'txt':
        from src.core.refine.txt_refiner import refine_txt_file
        return await _managed_refinement(refine_txt_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
            log_callback=log_callback,
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            max_tokens_per_chunk=max_tokens_per_chunk,
            soft_limit_ratio=additional_config.get('soft_limit_ratio'),
            prompt_options=prompt_options,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            refinement_original_path=refinement_original_path,
        ))

    if detected_type == 'epub':
        from src.core.refine.epub_refiner import refine_epub_file
        return await _managed_refinement(refine_epub_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
            log_callback=log_callback,
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            max_tokens_per_chunk=max_tokens_per_chunk,
            prompt_options=prompt_options,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            refinement_original_path=refinement_original_path,
        ))

    if detected_type == 'docx':
        from src.core.refine.docx_refiner import refine_docx_file
        return await _managed_refinement(refine_docx_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
            log_callback=log_callback,
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            max_tokens_per_chunk=max_tokens_per_chunk,
            prompt_options=prompt_options,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            refinement_original_path=refinement_original_path,
        ))

    if detected_type == 'srt':
        from src.core.refine.srt_refiner import refine_srt_file
        return await _managed_refinement(refine_srt_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
            log_callback=log_callback,
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            prompt_options=prompt_options,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            refinement_original_path=refinement_original_path,
        ))

    supported = ', '.join(['txt', 'epub', 'srt', 'docx'])
    raise UnsupportedFormatError(
        f"Unsupported file format for refine-only: {detected_type}. "
        f"Supported formats: {supported}"
    )
