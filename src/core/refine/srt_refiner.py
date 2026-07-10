"""SRT refine-only mode.

Runs the existing refine_subtitle_translations helper on each subtitle of
an already-translated SRT and writes a polished file. Timestamps and
subtitle indices are preserved verbatim.
"""

import os
import aiofiles
from typing import Optional, Callable, Dict, Any

from src.config import (
    DEFAULT_MODEL,
    API_ENDPOINT,
    SRT_LINES_PER_BLOCK,
)

# Disable the char cap when grouping: block sizing is purely fixed-count
# (every block holds exactly SRT_LINES_PER_BLOCK subtitles).
_NO_CHAR_CAP = 10 ** 12
from src.core.llm_client import create_llm_client
from src.core.srt_processor import SRTProcessor
from src.core.subtitle_translator import refine_subtitle_translations


def _blocks_from_translation_checkpoint(subtitles, db_chunks):
    """Rebuild the exact phase-1 subtitle block boundaries when available."""
    checkpoint_blocks = []
    covered_positions = []
    for row in sorted(db_chunks or [], key=lambda item: item.get("chunk_index", -1)):
        if row.get("status") != "completed":
            continue
        metadata = row.get("chunk_data") or {}
        positions = metadata.get("block_subtitles")
        if positions is None:
            positions = metadata.get("block_indices")
        if not isinstance(positions, list):
            return []
        try:
            positions = [int(position) for position in positions]
        except (TypeError, ValueError):
            return []
        if not positions or any(
            position < 0 or position >= len(subtitles)
            for position in positions
        ):
            return []
        checkpoint_blocks.append([subtitles[position] for position in positions])
        covered_positions.extend(positions)

    if (
        checkpoint_blocks
        and sorted(covered_positions) == list(range(len(subtitles)))
        and len(set(covered_positions)) == len(subtitles)
    ):
        return checkpoint_blocks
    return []


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
    checkpoint_manager: Optional[Any] = None,
    translation_id: Optional[str] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated SRT file."""
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

    # Key translations by list position, NOT by the cue number printed in
    # the file: update_translated_subtitles applies by position, and real
    # files have gaps, restarts or 0-based numbering (issue #205).
    translations: Dict[int, str] = {
        idx: sub.get('text', '') for idx, sub in enumerate(subtitles)
    }
    subtitle_positions = {id(sub): idx for idx, sub in enumerate(subtitles)}

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

    # Fixed-count grouping for refine (no char cap): every block sent to
    # the LLM has the same shape, which keeps [N] marker accounting
    # predictable across the whole file.
    refine_blocks = srt_processor.group_subtitles_for_translation(
        subtitles, SRT_LINES_PER_BLOCK, _NO_CHAR_CAP
    )

    db_chunks = []
    if checkpoint_manager and translation_id:
        db_chunks = checkpoint_manager.db.get_chunks(translation_id) or []

    # Phase-1 checkpoints persist the exact subtitle indices in each block.
    # Reuse them for refine-after so context snapshots and marker groups remain
    # one-to-one even if the configured block size changes later.
    checkpoint_blocks = _blocks_from_translation_checkpoint(
        subtitles,
        db_chunks,
    )
    if checkpoint_blocks:
        refine_blocks = checkpoint_blocks
        if log_callback:
            log_callback(
                "refine_chunk_alignment_exact",
                f"Reusing {len(refine_blocks)} subtitle blocks for exact refinement/context alignment.",
            )
        
    from src.utils.novel_context import (
        RefinementContextTracker,
        map_dialogue_attributions_for_refinement,
        map_context_snapshots_for_refinement,
    )
    historical_contexts = map_context_snapshots_for_refinement(
        len(refine_blocks),
        db_chunks,
        (prompt_options or {}).get('novel_context', ''),
        refinement_units=[
            "\n".join(subtitle.get("text", "") for subtitle in block)
            for block in refine_blocks
        ],
    )
    historical_dialogue_attributions = (
        map_dialogue_attributions_for_refinement(
            len(refine_blocks),
            db_chunks,
        )
    )
    context_tracker = RefinementContextTracker(
        prompt_options=prompt_options or {},
        historical_contexts=historical_contexts,
        historical_dialogue_attributions=historical_dialogue_attributions,
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
            subtitle_blocks=refine_blocks,
            subtitle_positions=subtitle_positions,
            context_tracker=context_tracker,
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
