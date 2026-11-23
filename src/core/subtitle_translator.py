"""
Subtitle-specific translation module
"""
from typing import Any, Dict, List, Optional, Tuple

from src.config import SRT_LINES_PER_BLOCK
from src.prompts.prompts import generate_subtitle_refinement_block_prompt

# Sentinel large enough to disable the char cap when grouping subtitles:
# block sizing is now purely fixed-count (SRT_LINES_PER_BLOCK) for both
# translate and refine, so the legacy char cap is unused.
_NO_CHAR_CAP = 10 ** 12

from .epub import TagPreserver
from .translator import _build_chunk_glossary_block


async def refine_subtitle_translations(
    translations: Dict[int, str],
    target_language: str,
    model_name: str,
    llm_client,
    log_callback=None,
    prompt_options=None,
    post_processing_instructions="",
    stats_callback=None,
    check_interruption_callback=None,
    subtitle_blocks: Optional[List[List[Dict[str, str]]]] = None,
    subtitle_positions: Optional[Dict[int, int]] = None,
    dynamic_contexts: Optional[List[str]] = None,
    context_tracker: Optional[Any] = None,
) -> Dict[int, str]:
    """
    Refine subtitle translations using a second LLM pass.

    Mirrors the block-based translate pass: subtitles are grouped into blocks
    and refined together in a single LLM call per block, with per-subtitle
    fallback when the block response cannot be parsed.

    Args:
        translations: Dict mapping subtitle index to translated text
        target_language: Target language
        model_name: LLM model name
        llm_client: LLM client instance
        log_callback: Optional logging callback
        prompt_options: Optional prompt options dict
        post_processing_instructions: Additional refinement instructions
        stats_callback: Optional callback to report per-subtitle progress
        check_interruption_callback: Optional callback to abort the pass early
        subtitle_blocks: Optional list of subtitle blocks (each block is a
            list of subtitle dicts with a 'number' field). When provided,
            the refinement mirrors the translate-pass block structure.
            When None, blocks are derived from the translations dict using
            the configured SRT block size and char cap.
        subtitle_positions: Optional identity map (id(subtitle dict) ->
            position in the parsed subtitle list). When provided, blocks
            are resolved to the same index space the caller keyed
            `translations` with. Without it the cue number printed in the
            file is used as a fallback, which is only safe when numbering
            is exactly 1..N (issue #205).
        dynamic_contexts: Optional list of dynamic contexts per block for historical relationship mapping

    Returns:
        Dict mapping subtitle index to refined text
    """
    from src.core.srt_processor import SRTProcessor

    total_subtitles = len(translations)
    refined_translations: Dict[int, str] = {}
    completed_count = 0
    failed_count = 0
    # Transient per-job state (e.g. glossary cap warning dedupe) — never persisted.
    runtime_state: dict = {}

    if log_callback:
        log_callback("srt_refinement_info", f"Refining {total_subtitles} subtitles...")

    # Build the list of global-index groups we will refine together.
    # Each group is a list of global subtitle indices, ordered.
    if subtitle_blocks:
        index_groups: List[List[int]] = []
        for block in subtitle_blocks:
            group: List[int] = []
            for subtitle in block:
                if subtitle_positions is not None:
                    g_idx = subtitle_positions.get(id(subtitle))
                    if g_idx is None:
                        continue
                else:
                    # Fallback: printed cue number. Only correct for files
                    # numbered exactly 1..N — callers should pass
                    # subtitle_positions instead (issue #205).
                    try:
                        g_idx = int(subtitle['number']) - 1
                    except (KeyError, ValueError, TypeError):
                        continue
                if g_idx in translations and translations[g_idx].strip():
                    group.append(g_idx)
            if group:
                index_groups.append(group)
    else:
        # Re-group from the translations dict when no block structure is
        # supplied (e.g. refine-only path that didn't pass blocks through).
        # Fixed-count only — no char cap.
        sorted_indices = sorted(translations.keys())
        index_groups = []
        current: List[int] = []
        for g_idx in sorted_indices:
            text = translations[g_idx]
            if not text or not text.strip():
                continue
            if len(current) >= SRT_LINES_PER_BLOCK:
                index_groups.append(current)
                current = []
            current.append(g_idx)
        if current:
            index_groups.append(current)

    total_blocks = len(index_groups)
    srt_processor = SRTProcessor()
    previous_refined_block = ""
    if context_tracker is None:
        from src.utils.novel_context import RefinementContextTracker
        context_tracker = RefinementContextTracker(
            prompt_options=prompt_options or {},
            historical_contexts=dynamic_contexts or [],
            log_callback=log_callback,
        )

    # Preserve empty subtitles untouched so the output stays complete.
    # Count them as completed: they require no refinement, so leaving them
    # out of completed_count would prevent progress from ever reaching 100%.
    for g_idx, text in translations.items():
        if not text or not text.strip():
            refined_translations[g_idx] = text
            completed_count += 1

    max_block_attempts = 2  # initial attempt + 1 retry with reinforced reminder

    for block_idx, group in enumerate(index_groups):
        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback(
                    "srt_refinement_interrupted",
                    f"Refinement interrupted at block {block_idx + 1}/{total_blocks}"
                )
            # Carry over any not-yet-refined subtitles unchanged.
            for remaining_group in index_groups[block_idx:]:
                for g_idx in remaining_group:
                    refined_translations.setdefault(g_idx, translations[g_idx])
            break

        if log_callback:
            log_callback(
                "srt_refinement_block_start",
                f"🪄 Refining subtitle block {block_idx + 1}/{total_blocks}...",
            )

        # Build local-index tuples and the local->global mapping.
        local_subtitle_tuples: List[Tuple[int, str]] = []
        local_to_global: Dict[int, int] = {}
        for local_idx, g_idx in enumerate(group):
            local_subtitle_tuples.append((local_idx, translations[g_idx]))
            local_to_global[local_idx] = g_idx

        block_text_for_glossary = "\n".join(text for _, text in local_subtitle_tuples)
        # The block holds translations: match the glossary on its target forms.
        # (Ported from upstream c63963a3.)
        glossary_block = _build_chunk_glossary_block(
            block_text_for_glossary, prompt_options, log_callback=log_callback,
            runtime_state=runtime_state, match_target_forms=True,
        )

        block_refined: Dict[int, str] = {}
        expected_local_indices = list(range(len(local_subtitle_tuples)))

        context_content = await context_tracker.next_context(
            text=block_text_for_glossary,
            llm_client=llm_client,
            model_name=model_name,
            target_language=target_language,
            display_index=block_idx + 1,
            total_chunks=total_blocks,
        )

        # Inject historical/source-first context for this block.
        local_prompt_options = dict(prompt_options) if prompt_options else {}
        if context_content:
            local_prompt_options['novel_context'] = context_content
        dialogue_attribution = getattr(
            context_tracker,
            "current_dialogue_attribution",
            None,
        )
        if dialogue_attribution:
            local_prompt_options["dialogue_attribution"] = dialogue_attribution
        else:
            local_prompt_options.pop("dialogue_attribution", None)

        for attempt in range(max_block_attempts):
            if check_interruption_callback and check_interruption_callback():
                break

            # On retry, reinforce the reminder with the exact missing indices.
            extra_instructions = post_processing_instructions or ''
            if attempt > 0:
                missing_local = [li for li in expected_local_indices
                                 if local_to_global[li] not in block_refined]
                missing_str = ", ".join(f"[{li}]" for li in missing_local)
                extra_instructions = (
                    (extra_instructions + "\n\n" if extra_instructions else "")
                    + f"CRITICAL: Your previous response was incomplete. "
                    f"You MUST output ALL {len(local_subtitle_tuples)} indices "
                    f"[0] through [{len(local_subtitle_tuples) - 1}] in order, "
                    f"each followed by the refined subtitle. "
                    f"Missing indices last time: {missing_str}. Do NOT stop early."
                )

            try:
                if log_callback and attempt > 0:
                    log_callback("srt_refinement_retry",
                                 f"Block {block_idx + 1}: retry attempt {attempt} "
                                 f"({len(local_subtitle_tuples) - len(block_refined)} subtitles still missing)")

                from src.core.translator import run_chunk_reflection_pass

                draft_block = "\n".join(
                    f"[{local_idx}]{text}"
                    for local_idx, text in local_subtitle_tuples
                )
                source_queue = (prompt_options or {}).get("_refinement_source_queue") or []
                source_chunk = (
                    str(source_queue[block_idx])
                    if block_idx < len(source_queue)
                    else ""
                )
                local_prompt_options["editor_source_mode"] = (
                    "checkpoint" if source_chunk.strip() else "monolingual"
                )
                from src.core.context.unit_pipeline import prepare_unit_prompt_options
                local_prompt_options = prepare_unit_prompt_options(
                    local_prompt_options,
                    unit_index=block_idx,
                    phase="refinement",
                    file_type="srt",
                    source_text=source_chunk,
                    target_language=target_language,
                    dialogue_attribution=dialogue_attribution,
                    scene_key=str(block_idx),
                    nearby_source=(previous_refined_block,),
                )

                def validate_subtitle_repair(value):
                    parsed_value = srt_processor.extract_block_translations_with_remapping(
                        value,
                        local_to_global,
                    )
                    if sorted(parsed_value) != sorted(group):
                        return "adapter_subtitle_marker_sequence_changed"
                    if any(not str(parsed_value[index] or "").strip() for index in group):
                        return "adapter_subtitle_text_empty"
                    return None

                refined_block_text = await run_chunk_reflection_pass(
                    source_chunk=source_chunk,
                    draft_translation=draft_block,
                    target_language=target_language,
                    model_name=model_name,
                    llm_client=llm_client,
                    novel_context=local_prompt_options.get("novel_context", ""),
                    custom_instructions=extra_instructions,
                    glossary_block=glossary_block,
                    log_callback=log_callback,
                    prompt_options=local_prompt_options,
                    repair_validator=validate_subtitle_repair,
                )
                parsed = srt_processor.extract_block_translations_with_remapping(
                    refined_block_text,
                    local_to_global,
                )
                for g_idx, text in parsed.items():
                    if g_idx not in block_refined and text.strip():
                        block_refined[g_idx] = text

                # All subtitles recovered? stop retrying.
                if len(block_refined) == len(group):
                    break

            except Exception as e:
                # Re-raise RateLimitError to trigger auto-pause
                from src.core.llm.exceptions import RateLimitError
                if isinstance(e, RateLimitError):
                    raise
                if log_callback:
                    log_callback("srt_refinement_error",
                                 f"Block {block_idx + 1} attempt {attempt + 1}: {e}")

        # Apply refined where available, keep original draft otherwise.
        # No per-subtitle fallback: small models corrupt single-subtitle calls
        # with hallucinated content from the surrounding context.
        # Both branches bump completed_count: the subtitle is finalized in
        # the output either way, so progress reaches 100%.
        # NOTE: we do NOT bump failed_count for kept-original. In refine
        # semantics, the subtitle already has a valid translation — refine
        # just didn't polish it further. The file is complete; reporting
        # failed_chunks > 0 here would make the backend mark the job as
        # 'partial' and hide the download UI (handlers.py:588), which is
        # the right behavior for translate but wrong for refine.
        # Per-subtitle log_callback below preserves visibility of which
        # subs fell back to original.
        # Emit stats per-subtitle so the UI ticks 1-by-1 (e.g. 781, 782, ...,
        # 786) instead of jumping by the block size all at once.
        missing_in_block = 0
        for g_idx in group:
            if g_idx in block_refined:
                refined_translations[g_idx] = block_refined[g_idx]
            else:
                refined_translations[g_idx] = translations[g_idx]
                failed_count += 1
                missing_in_block += 1
                if log_callback:
                    log_callback("srt_refinement_fallback",
                                 f"Subtitle {g_idx + 1}: keeping original (block missed this index)")
            completed_count += 1
            if stats_callback:
                stats_callback({
                    'total_chunks': total_subtitles,
                    'completed_chunks': completed_count,
                    'failed_chunks': failed_count,
                })

        from src.core.refine.persistence import save_refinement_unit
        block_translations = {
            g_idx: refined_translations[g_idx] for g_idx in group
        }
        persisted_block = "\n".join(
            f"[{local_idx}]{block_translations[g_idx]}"
            for local_idx, g_idx in enumerate(group)
        )
        save_refinement_unit(
            prompt_options,
            unit_index=block_idx,
            source_text=(
                str(((prompt_options or {}).get("_refinement_source_queue") or [])[block_idx])
                if block_idx < len((prompt_options or {}).get("_refinement_source_queue") or [])
                else ""
            ),
            refined_text=persisted_block,
            status=("completed" if missing_in_block == 0 else "failed"),
            quality_status=("passed" if missing_in_block == 0 else "review_required"),
            metadata={
                "block_translations": block_translations,
                "block_indices": group,
                "structure": "srt",
            },
        )

        # Update previous_refined_block context (last up to 5 subtitles of this group).
        last_items = []
        for local_idx, g_idx in enumerate(group[-5:]):
            last_items.append(f"[{local_idx}]{refined_translations.get(g_idx, translations[g_idx])}")
        previous_refined_block = "\n".join(last_items)
        if log_callback:
            log_callback(
                "srt_refinement_block_complete",
                f"✅ Subtitle block {block_idx + 1}/{total_blocks} refinement complete.",
            )

    return refined_translations
