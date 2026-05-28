"""SRT refine-only mode.

Runs the existing refine_subtitle_translations helper on each subtitle of
an already-translated SRT and writes a polished file. Timestamps and
subtitle indices are preserved verbatim.
"""

import os
import aiofiles
from typing import Optional, Callable, Dict, Any

from src.config import DEFAULT_MODEL, API_ENDPOINT
from src.core.llm_client import create_llm_client
from src.core.srt_processor import SRTProcessor
from src.core.subtitle_translator import refine_subtitle_translations


async def refine_srt_file(
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
    prompt_options: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated SRT file."""
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input SRT file '{input_filepath}' not found."
        if log_callback:
            log_callback("srt_file_not_found", err_msg)
        return False

    try:
        async with aiofiles.open(input_filepath, 'r', encoding='utf-8') as f:
            srt_content = await f.read()
    except Exception as e:
        if log_callback:
            log_callback("srt_read_error",
                         f"ERROR: Reading SRT file '{input_filepath}': {e}")
        return False

    srt_processor = SRTProcessor()
    if not srt_processor.validate_srt(srt_content):
        if log_callback:
            log_callback("srt_invalid_format", "Invalid SRT file format")
        return False

    subtitles = srt_processor.parse_srt(srt_content)
    if not subtitles:
        if log_callback:
            log_callback("srt_no_subtitles", "No subtitles found in file")
        return False

    if log_callback:
        log_callback("srt_refine_start",
                     f"✨ Refining {len(subtitles)} subtitles in {target_language}...")

    if stats_callback:
        stats_callback({
            'total_chunks': len(subtitles),
            'completed_chunks': 0,
            'failed_chunks': 0,
        })

    translations: Dict[int, str] = {}
    for sub in subtitles:
        try:
            idx = int(sub['number']) - 1
        except (KeyError, ValueError):
            continue
        translations[idx] = sub.get('text', '')

    llm_client = create_llm_client(
        llm_provider, gemini_api_key, cli_api_endpoint, model_name,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        log_callback=log_callback,
    )

    try:
        refined = await refine_subtitle_translations(
            translations=translations,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            log_callback=log_callback,
            prompt_options=prompt_options,
            post_processing_instructions=(
                prompt_options.get('refinement_instructions', '')
                if prompt_options else ''
            ),
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
        )
    finally:
        if llm_client:
            try:
                await llm_client.close()
            except Exception:
                pass

    if check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback("srt_refine_interrupted",
                         "Refinement interrupted before save")
        return False

    refined_subs = srt_processor.update_translated_subtitles(subtitles, refined)
    refined_srt = srt_processor.reconstruct_srt(refined_subs)

    try:
        async with aiofiles.open(output_filepath, 'w', encoding='utf-8') as f:
            await f.write(refined_srt)
        if log_callback:
            log_callback("srt_refine_done",
                         f"✅ Refined SRT saved: {output_filepath}")
        return True
    except Exception as e:
        if log_callback:
            log_callback("srt_save_error",
                         f"ERROR: Saving SRT file '{output_filepath}': {e}")
        return False
