"""
Generic translator orchestrator using the adapter pattern.

This module provides a unified translation workflow that works with any file format
through the FormatAdapter interface.
"""

from typing import Callable, Optional, Dict, Any
from pathlib import Path

from .format_adapter import FormatAdapter
from src.core.llm_client import LLMClient
from src.utils.unified_logger import get_logger
from src.core.jobs import UnitTranslationOutcome

logger = get_logger(__name__)


class _ValidationFailed:
    """LLM content that failed adapter validation after all retry attempts.

    Carries the best-effort content so already-valid parts (e.g. the SRT
    cues whose markers did come back) survive in the output while the unit
    itself is recorded as failed.
    """

    def __init__(self, content: str):
        self.content = content


class GenericTranslator:
    """
    Generic orchestrator for translating files using format adapters.

    This is the single translation engine for TXT and SRT, shared by both the
    web API and the CLI (via src.core.adapters.translate_file). The legacy
    per-format functions it replaced (translate_chunks, the *_with_callbacks
    dispatchers) have been removed.

    It provides a unified workflow:
    1. Prepare file via adapter
    2. Get translation units
    3. Load checkpoint if exists
    4. Translate each unit with LLM
    5. Save each translated unit
    6. Reconstruct output file
    7. Clean up resources
    """

    def __init__(
        self,
        adapter: FormatAdapter,
        checkpoint_manager: Any,  # CheckpointManager
        translation_id: str
    ):
        """
        Initialize the generic translator.

        Args:
            adapter: Format-specific adapter (TxtAdapter, SrtAdapter, etc.)
            checkpoint_manager: Checkpoint manager for resume capability
            translation_id: Unique identifier for this translation job
        """
        self.adapter = adapter
        self.checkpoint_manager = checkpoint_manager
        self.translation_id = translation_id

    async def translate(
        self,
        source_language: str,
        target_language: str,
        model_name: str,
        llm_provider: str,
        log_callback: Optional[Callable] = None,
        stats_callback: Optional[Callable] = None,
        check_interruption_callback: Optional[Callable] = None,
        bilingual_output: bool = False,
        parallel_workers: int = 1,
        **llm_kwargs
    ) -> bool:
        """
        Execute the complete translation workflow.

        Args:
            source_language: Source language name
            target_language: Target language name
            model_name: LLM model identifier
            llm_provider: LLM provider name (ollama, gemini, openai, openrouter)
            log_callback: Optional callback for logging (receives type and message)
            stats_callback: Optional callback for statistics updates (receives dict with total_chunks, completed_chunks, failed_chunks)
            check_interruption_callback: Optional callback to check if translation should be interrupted
            bilingual_output: If True, output will contain both original and translated text
            parallel_workers: Number of chunks translated concurrently (resolved
                against the provider; local providers are forced back to 1). When
                1, behavior is identical to the legacy sequential loop, including
                cross-chunk translation context chaining.
            **llm_kwargs: Additional LLM configuration (endpoint, api_key, etc.)

        Returns:
            True if translation completed successfully, False otherwise
        """
        try:
            prompt_options = llm_kwargs.setdefault('prompt_options', {})
            if not isinstance(prompt_options, dict):
                prompt_options = {}
                llm_kwargs['prompt_options'] = prompt_options
            prompt_options.setdefault("use_relationship_reasoning", "project")

            # 1. Prepare file for translation
            if log_callback:
                log_callback("prepare_start", f"Preparing {self.adapter.format_name.upper()} file for translation")

            if not await self.adapter.prepare_for_translation():
                if log_callback:
                    log_callback("prepare_failed", "Failed to prepare file for translation")
                return False

            # 2. Get translation units
            units = self.adapter.get_translation_units()
            total_units = len(units)

            if total_units == 0:
                if log_callback:
                    log_callback("no_units", "No translation units found in file")
                return False

            if log_callback:
                log_callback("units_found", f"Found {total_units} translation units")
                chapter_mode = bool(
                    (self.adapter.config.get("prompt_options") or {}).get(
                        "chapter_mode"
                    )
                )
                if chapter_mode and self.adapter.format_name == "txt":
                    chapter_count = len({
                        unit.metadata.get("chapter_index")
                        for unit in units
                        if unit.metadata.get("chapter_index") is not None
                    })
                    log_callback(
                        "chapter_mode_ready",
                        f"Chapter-aware mode prepared {chapter_count} chapter(s) "
                        f"as {total_units} translation unit(s).",
                    )

            # Send initial stats with total_chunks
            if stats_callback:
                stats_callback({
                    'total_chunks': total_units,
                    'completed_chunks': 0,
                    'failed_chunks': 0
                })

            # 3. Check for checkpoint and resume
            restored_completed = set()
            restored_failed = set()
            failed_editor_drafts_by_index = {}
            completed_translations_by_index = {}
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.translation_id)
            continuation_base_id = self.adapter.config.get('continuation_base_id')
            continuation_context_seed = None
            if continuation_base_id and checkpoint_data:
                previous_checkpoint = self.checkpoint_manager.load_checkpoint(
                    continuation_base_id
                )
                previous_chunks = (
                    previous_checkpoint.get('chunks', [])
                    if previous_checkpoint
                    else []
                )
                if previous_chunks:
                    from src.core.continuation import seed_matching_prefix
                    seed_matching_prefix(
                        checkpoint_manager=self.checkpoint_manager,
                        translation_id=self.translation_id,
                        previous_chunks=previous_chunks,
                        new_source_units=[unit.content for unit in units],
                        total_units=total_units,
                        log_callback=log_callback,
                        label="unit",
                    )
                    checkpoint_data = self.checkpoint_manager.load_checkpoint(
                        self.translation_id
                    )
                from src.core.continuation import latest_context_seed
                continuation_context_seed = latest_context_seed(previous_chunks)

            if checkpoint_data:
                await self.adapter.resume_from_checkpoint(checkpoint_data)
                # Pending work is derived from per-chunk statuses, not from the
                # progress pointer: the pointer advances past failed units, so
                # resuming from it alone would skip them forever (issue #204).
                restored_completed = {
                    c['chunk_index'] for c in checkpoint_data.get('chunks', [])
                    if c.get('status') == 'completed'
                    and not (c.get('chunk_data') or {}).get('editor_retry_pending')
                    and 0 <= c.get('chunk_index', -1) < total_units
                }
                failed_editor_drafts_by_index = {
                    c['chunk_index']: c.get('translated_text')
                    for c in checkpoint_data.get('chunks', [])
                    if c.get('status') in {'failed', 'editor_retry'}
                    and c.get('translated_text')
                    and (c.get('chunk_data') or {}).get('editor_validation')
                    and 0 <= c.get('chunk_index', -1) < total_units
                }
                # Beta.45 stored usable editor-review drafts as failed chunks.
                # Reclassify them only on an explicit resume, and only after the
                # format adapter confirms the preserved content is structurally
                # valid. This avoids retranslating valid book content.
                for chunk in checkpoint_data.get('chunks', []):
                    index = chunk.get('chunk_index', -1)
                    draft = chunk.get('translated_text') or ''
                    data = dict(chunk.get('chunk_data') or {})
                    if not (
                        chunk.get('status') in {'failed', 'editor_retry'}
                        and draft
                        and data.get('editor_validation')
                        and 0 <= index < total_units
                        and self.adapter.validate_unit_translation(
                            units[index].unit_id, draft
                        ) is None
                    ):
                        continue
                    if await self.adapter.save_unit_translation(
                        units[index].unit_id, draft
                    ):
                        data['quality_status'] = 'review_required'
                        data.pop('execution_failure_class', None)
                        self.checkpoint_manager.save_checkpoint(
                            translation_id=self.translation_id,
                            chunk_index=index,
                            original_text=units[index].content,
                            translated_text=draft,
                            chunk_data=data,
                            total_chunks=total_units,
                            chunk_status='completed',
                        )
                        restored_completed.add(index)
                        failed_editor_drafts_by_index.pop(index, None)
                checkpoint_data = self.checkpoint_manager.load_checkpoint(
                    self.translation_id
                )
                restored_completed = {
                    c['chunk_index'] for c in checkpoint_data.get('chunks', [])
                    if c.get('status') == 'completed'
                    and not (c.get('chunk_data') or {}).get('editor_retry_pending')
                    and 0 <= c.get('chunk_index', -1) < total_units
                }
                remaining_failed = sum(
                    1 for c in checkpoint_data.get('chunks', [])
                    if c.get('status') != 'completed'
                    and 0 <= c.get('chunk_index', -1) < total_units
                )
                self.checkpoint_manager.db.update_job_progress(
                    self.translation_id,
                    completed_chunks=len(restored_completed),
                    failed_chunks=remaining_failed,
                )
                restored_failed = {
                    c['chunk_index'] for c in checkpoint_data.get('chunks', [])
                    if c.get('status') != 'completed'
                    and 0 <= c.get('chunk_index', -1) < total_units
                }
                completed_translations_by_index = {
                    c["chunk_index"]: c.get("translated_text") or ""
                    for c in checkpoint_data.get("chunks", [])
                    if c.get("status") in {"completed", "partial"}
                    and c.get("translated_text")
                    and 0 <= c.get("chunk_index", -1) < total_units
                }
                if log_callback:
                    log_callback("checkpoint_resumed",
                        f"Resuming: {len(restored_completed)}/{total_units} units already translated")
                # Update stats with resumed progress
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': len(restored_completed),
                        'failed_chunks': remaining_failed,
                        'review_required_chunks': sum(
                            1 for c in checkpoint_data.get('chunks', [])
                            if (c.get('chunk_data') or {}).get('quality_status')
                            == 'review_required'
                        ),
                    })
            else:
                # 4. Create new translation job
                prompt_options.setdefault("context_contract_version", 5)
                prompt_options.setdefault("use_relationship_llm_judge", "selective")
                prompt_options.setdefault("source_residue_validation", True)
                self.checkpoint_manager.start_job(
                    translation_id=self.translation_id,
                    file_type=self.adapter.format_name,
                    config={
                        'input_file_path': str(self.adapter.input_file_path),
                        'output_file_path': str(self.adapter.output_file_path),
                        'source_language': source_language,
                        'target_language': target_language,
                        'model': model_name,
                        'model_name': model_name,
                        'llm_provider': llm_provider,
                        'llm_api_endpoint': (
                            llm_kwargs.get('api_endpoint')
                            or llm_kwargs.get('endpoint')
                        ),
                        'request_timeout': llm_kwargs.get('timeout', 120),
                        'prompt_options': llm_kwargs.get('prompt_options', {}),
                        'parallel_workers': parallel_workers,
                        **self.adapter.config
                    },
                    input_file_path=str(self.adapter.input_file_path)
                )

            # 5. Create LLM client
            from src.core.translator import generate_translation_request
            from src.core.llm.runtime import build_draft_and_editor_clients

            prompt_options = llm_kwargs.get('prompt_options', {})
            (
                llm_client,
                editor_llm_client,
                _draft_spec,
                editor_spec,
            ) = build_draft_and_editor_clients(
                draft_provider=llm_provider,
                draft_model=model_name,
                draft_endpoint=(
                    llm_kwargs.get("api_endpoint") or llm_kwargs.get("endpoint")
                ),
                prompt_options=prompt_options,
                credentials=llm_kwargs,
                context_window=llm_kwargs.get("context_window"),
                log_callback=log_callback,
            )
            editor_provider = editor_spec.provider
            editor_model = editor_spec.model

            # 6. Translate each unit (sequentially, or with continuous concurrency)
            from src.config import resolve_parallel_workers, UNIT_VALIDATION_RETRIES
            from src.core.common.parallel import iter_ordered_concurrent
            from src.core.llm.exceptions import RateLimitError

            chapter_mode = bool(
                prompt_options.get("chapter_mode")
                and self.adapter.format_name == "txt"
            )
            novel_context_file = prompt_options.get('novel_context_file')
            novel_context_path = None
            auto_update_context = prompt_options.get('auto_update_context', False)

            from src.config import NOVEL_CONTEXTS_DIR
            from src.utils.novel_context import (
                open_novel_context_session,
                should_update_novel_context_for_index,
            )
            from src.utils.db_addressing import (
                apply_db_addressing_to_session,
                build_directed_addressing_prompt_context,
                sync_context_update_addressing_to_db,
                sync_markdown_addressing_to_db,
            )
            from src.utils.addressing_schema import context_contract_version
            from src.utils.relationship_sync import (
                apply_relationship_graph_to_session,
                build_relationship_prompt_context,
                judge_ambiguous_relationship_candidates,
                resolve_relationship_reasoning_mode,
                sync_context_update_relationships_to_db,
                sync_markdown_relationships_to_db,
            )
            relationship_mode = resolve_relationship_reasoning_mode(prompt_options)
            contract_version = context_contract_version(prompt_options)
            from src.utils.progress_logging import emit_progress_log

            emit_progress_log(
                log_callback,
                "relationship_reasoning_mode",
                f"Relationship reasoning mode: {relationship_mode}.",
                layer="relationship_reasoning",
                data={"mode": relationship_mode, "contract_version": "1.0"},
            )

            resume_snapshot = None
            resume_snapshot_index = None
            resume_dialogue_state = None
            resume_dialogue_scene_key = None
            used_continuation_context_seed = False
            analyzed_context_indices = set()
            checkpoint_context_data_by_index = {}
            if checkpoint_data:
                context_rows = []
                for checkpoint_chunk in checkpoint_data.get('chunks', []):
                    checkpoint_chunk_data = (
                        checkpoint_chunk.get('chunk_data') or {}
                    )
                    chunk_index = checkpoint_chunk.get('chunk_index')
                    if (
                        isinstance(chunk_index, int)
                        and checkpoint_chunk_data.get('context_snapshot')
                        and checkpoint_chunk.get('status') in (
                            'completed',
                            'partial',
                            'failed',
                        )
                    ):
                        analyzed_context_indices.add(chunk_index)
                        checkpoint_context_data_by_index[chunk_index] = (
                            dict(checkpoint_chunk_data)
                        )
                        context_rows.append(checkpoint_chunk)

                if context_rows:
                    resume_chunk = max(
                        context_rows,
                        key=lambda chunk: chunk.get('chunk_index', -1),
                    )
                    resume_snapshot_index = resume_chunk.get('chunk_index')
                    checkpoint_chunk_data = (
                        resume_chunk.get('chunk_data') or {}
                    )
                    resume_snapshot = checkpoint_chunk_data.get(
                        'context_snapshot'
                    )
                    resume_dialogue_state = (
                        (
                            checkpoint_chunk_data.get(
                                'dialogue_attribution'
                            ) or {}
                        ).get('state_after')
                    )
                    resume_dialogue_scene_key = (
                        checkpoint_chunk_data.get(
                            'dialogue_attribution'
                        ) or {}
                    ).get('scene_key')
                elif restored_completed:
                    resume_snapshot_index = max(restored_completed)
                    for checkpoint_chunk in checkpoint_data.get('chunks', []):
                        if checkpoint_chunk.get('chunk_index') == resume_snapshot_index:
                            checkpoint_chunk_data = (
                                checkpoint_chunk.get('chunk_data') or {}
                            )
                            resume_snapshot = checkpoint_chunk_data.get(
                                'context_snapshot'
                            )
                            resume_dialogue_state = (
                                (
                                    checkpoint_chunk_data.get(
                                        'dialogue_attribution'
                                    ) or {}
                                ).get('state_after')
                            )
                            resume_dialogue_scene_key = (
                                checkpoint_chunk_data.get(
                                    'dialogue_attribution'
                                ) or {}
                            ).get('scene_key')
                            break

            if not resume_snapshot and continuation_context_seed:
                resume_snapshot = continuation_context_seed.get(
                    'context_snapshot'
                )
                resume_snapshot_index = continuation_context_seed.get(
                    'chunk_index'
                )
                resume_dialogue_state = continuation_context_seed.get(
                    'dialogue_state'
                )
                resume_dialogue_scene_key = continuation_context_seed.get(
                    'dialogue_scene_key'
                )
                used_continuation_context_seed = True

            try:
                context_session = open_novel_context_session(
                    prompt_options=prompt_options,
                    novel_contexts_dir=NOVEL_CONTEXTS_DIR,
                    input_filename=str(getattr(self.adapter, 'input_file_path', '') or ''),
                    fallback_name="text",
                    resume_snapshot=resume_snapshot,
                    resume_dialogue_state=resume_dialogue_state,
                    resume_dialogue_scene_key=resume_dialogue_scene_key,
                    log_callback=log_callback,
                )
                if resume_snapshot and context_session and log_callback:
                    event_type = (
                        "continuation_context_seed"
                        if used_continuation_context_seed
                        else "novel_context_resume"
                    )
                    message = (
                        "Add New Content: continuing context from previous "
                        f"job chunk {resume_snapshot_index} snapshot."
                        if event_type == "continuation_context_seed"
                        else (
                            f"Restored context from chunk "
                            f"{resume_snapshot_index} snapshot."
                        )
                    )
                    log_callback(event_type, message)
                if (
                    context_session
                    and self.translation_id
                    and getattr(self.checkpoint_manager, "db", None)
                    and relationship_mode != "off"
                ):
                    sync_markdown_relationships_to_db(
                        translation_id=self.translation_id,
                        db=self.checkpoint_manager.db,
                        context_or_dynamic_state=context_session.content,
                        target_language=target_language,
                        chunk_index=resume_snapshot_index or 0,
                        trigger_source="job_context_load",
                        log_callback=log_callback,
                    )
                    apply_relationship_graph_to_session(
                        context_session,
                        self.translation_id,
                        self.checkpoint_manager.db,
                    )
                if (
                    context_session
                    and self.translation_id
                    and getattr(self.checkpoint_manager, "db", None)
                    and prompt_options.get("use_db_directed_addressing", True)
                ):
                    sync_markdown_addressing_to_db(
                        translation_id=self.translation_id,
                        db=self.checkpoint_manager.db,
                        context_or_dynamic_state=context_session.content,
                        target_language=target_language,
                        chunk_index=0,
                        trigger_source="job_context_load",
                        log_callback=log_callback,
                    )
                    apply_db_addressing_to_session(
                        context_session,
                        self.translation_id,
                        self.checkpoint_manager.db,
                    )
            except Exception as e:
                context_session = None
                if log_callback:
                    log_callback(
                        "novel_context_error",
                        f"Error loading novel context '{novel_context_file}': {str(e)}",
                    )

            if auto_update_context and context_session:
                if parallel_workers > 1 or resolve_parallel_workers(llm_provider, parallel_workers) > 1:
                    if log_callback:
                        log_callback("novel_context_workers_override", "Warning: Auto-updating novel context requires sequential translation. Forcing parallel workers to 1.")
                parallel_workers = 1

            workers = resolve_parallel_workers(llm_provider, parallel_workers)
            sequential = workers == 1
            max_validation_attempts = 1 + max(0, UNIT_VALIDATION_RETRIES)

            last_context = ""
            failed_count = len(restored_failed)
            completed_count = len(restored_completed)
            failed_indices = set(restored_failed)
            review_required_indices = {
                c['chunk_index'] for c in (checkpoint_data or {}).get('chunks', [])
                if (c.get('chunk_data') or {}).get('quality_status') == 'review_required'
                and 0 <= c.get('chunk_index', -1) < total_units
            }
            reused_context_data_by_index = {}
            pending_db_addressing_context_by_index = {}
            pending_relationship_context_by_index = {}

            if (
                sequential and self.translation_id
                and getattr(self.checkpoint_manager, "db", None)
                and (
                    prompt_options.get("reflection_mode")
                    or prompt_options.get("auto_update_context")
                )
            ):
                from src.utils.narrator_voice import bootstrap_narrator_voice
                await bootstrap_narrator_voice(
                    db=self.checkpoint_manager.db,
                    translation_id=self.translation_id,
                    chunks=self.checkpoint_manager.db.get_chunks(self.translation_id) or [],
                    target_language=target_language,
                    model_name=str(prompt_options.get("editor_model") or model_name),
                    llm_client=prompt_options.get("_editor_llm_client") or llm_client,
                    file_type=self.adapter.format_name,
                )

            async def _translate_unit(i, analyze_context=True):
                """Translate one unit. Reads last_context only in sequential mode
                (parallel runs have no stable 'previous translation').

                Results failing adapter validation (e.g. SRT [N] markers
                dropped by the LLM) are retried with a reinforced prompt up
                to max_validation_attempts; after exhaustion the best-effort
                content is returned wrapped in _ValidationFailed."""
                unit = units[i]
                if log_callback:
                    log_callback("unit_start",
                        f"Translating unit {i+1}/{total_units} ({unit.unit_id})")
                if i > 0 and sequential and self.translation_id and (
                    prompt_options.get("reflection_mode")
                    or prompt_options.get("auto_update_context")
                ):
                    from src.utils.narrator_voice import bootstrap_narrator_voice
                    await bootstrap_narrator_voice(
                        db=self.checkpoint_manager.db,
                        translation_id=self.translation_id,
                        chunks=self.checkpoint_manager.db.get_chunks(self.translation_id) or [],
                        target_language=target_language,
                        model_name=str(prompt_options.get("editor_model") or model_name),
                        llm_client=prompt_options.get("_editor_llm_client") or llm_client,
                        file_type=self.adapter.format_name,
                    )

                base_instructions = (prompt_options or {}).get('custom_instructions', '')
                attempt_options = prompt_options
                result = None

                should_analyze_context = (
                    analyze_context
                    and auto_update_context
                    and context_session
                    and should_update_novel_context_for_index(i, prompt_options)
                    and not (
                        i in analyzed_context_indices
                        and resume_snapshot_index is not None
                        and resume_snapshot_index >= i
                    )
                )
                if should_analyze_context:
                    if log_callback:
                        log_callback(
                            "novel_context_updating",
                            f"Analyzing source context for unit {i+1} before translation...",
                        )
                    try:
                        relationship_fallback_context = context_session.content
                        change_logs = await context_session.analyze_source(
                            llm_client=llm_client,
                            model_name=model_name,
                            source_chunk=unit.content,
                            source_language=source_language,
                            target_language=target_language,
                            chunk_index=i + 1,
                            total_chunks=total_units,
                            scene_key=unit.metadata.get("chapter_index"),
                        )
                        # Deterministic social evidence is committed before the
                        # draft request so an honorific or explicit hierarchy in
                        # this source unit can guide the translation of the same
                        # unit. LLM-derived candidates retain their existing
                        # transaction boundary below.
                        if self.translation_id and getattr(
                            self.checkpoint_manager, "db", None
                        ):
                            from src.core.context import commit_source_social_evidence

                            commit_source_social_evidence(
                                translation_id=self.translation_id,
                                db=self.checkpoint_manager.db,
                                global_lore=context_session.global_lore,
                                source_text=unit.content,
                                dialogue_attribution=(
                                    context_session.dialogue_attribution
                                ),
                                source_language=source_language,
                                target_language=target_language,
                                chunk_index=i,
                                log_callback=log_callback,
                            )
                        if log_callback:
                            log_callback(
                                "novel_context_updated",
                                f"Novel context prepared for unit {i+1}.",
                            )
                            for change_log in change_logs:
                                log_callback("novel_context_log", change_log)
                        if unit.metadata is None:
                            unit.metadata = {}
                        unit.metadata['dialogue_attribution'] = (
                            context_session.dialogue_attribution
                        )
                        unit.metadata['context_snapshot'] = (
                            context_session.snapshot()
                        )
                        analyzed_context_indices.add(i)
                        if (
                            self.translation_id
                            and getattr(self.checkpoint_manager, "db", None)
                            and prompt_options.get("use_db_directed_addressing", True)
                        ):
                            pending_db_addressing_context_by_index[i] = {
                                "content": context_session.content,
                                "dialogue_attribution": context_session.dialogue_attribution,
                                "candidates": list(
                                    context_session.addressing_candidates
                                ),
                                "parser_status": context_session.addressing_parse_status,
                                "source_text": unit.content,
                            }
                        if (
                            self.translation_id
                            and getattr(self.checkpoint_manager, "db", None)
                            and relationship_mode != "off"
                        ):
                            from src.core.context import relevant_character_names

                            locked_facts = [
                                edge for edge in self.checkpoint_manager.db.get_relationship_edges(
                                    self.translation_id,
                                    statuses=["accepted"],
                                )
                                if edge.get("is_locked")
                            ]
                            judged_candidates = await judge_ambiguous_relationship_candidates(
                                llm_client=llm_client,
                                candidates=context_session.relationship_candidates,
                                source_text=unit.content,
                                model_name=model_name,
                                enabled=(
                                    prompt_options.get("use_relationship_llm_judge")
                                    in {True, "selective", "always"}
                                ),
                                locked_facts=locked_facts,
                                log_callback=log_callback,
                            )
                            pending_relationship_context_by_index[i] = {
                                "content": context_session.content,
                                "source_text": unit.content,
                                "candidates": judged_candidates,
                                "parser_status": context_session.relationship_parse_status,
                                "active_character_names": relevant_character_names(
                                    global_lore=context_session.global_lore,
                                    source_text=unit.content,
                                    dialogue_attribution=context_session.dialogue_attribution,
                                    source_language=source_language,
                                ),
                            }
                            if relationship_mode == "project":
                                if prompt_options.get(
                                    "use_db_directed_addressing",
                                    True,
                                ):
                                    apply_db_addressing_to_session(
                                        context_session,
                                        self.translation_id,
                                        self.checkpoint_manager.db,
                                        fallback_context=relationship_fallback_context,
                                    )
                                apply_relationship_graph_to_session(
                                    context_session,
                                    self.translation_id,
                                    self.checkpoint_manager.db,
                                    fallback_context=relationship_fallback_context,
                                )
                                context_session.save()
                                unit.metadata['context_snapshot'] = (
                                    context_session.snapshot()
                                )
                        if log_callback and contract_version < 2:
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
                    if unit.metadata is None:
                        unit.metadata = {}
                    reused_context_data_by_index[i] = dict(
                        checkpoint_context_data_by_index[i]
                    )
                    unit.metadata.update(reused_context_data_by_index[i])
                elif analyze_context and auto_update_context and context_session:
                    context_session.remember_source(unit.content)
                elif (
                    not analyze_context
                    and unit.metadata
                    and unit.metadata.get('context_snapshot')
                ):
                    reused_context_data_by_index[i] = dict(unit.metadata)

                if novel_context_path and novel_context_path.is_file():
                    try:
                        disk_content = load_novel_context(novel_context_path.name, novel_context_path.parent)
                        disk_global_lore = extract_global_lore(disk_content)
                        if disk_global_lore:
                            prompt_options['novel_context'] = build_novel_context(
                                disk_global_lore,
                                context_session.dynamic_state if context_session else (current_dynamic_state or ""),
                            )
                    except Exception:
                        pass

                for attempt in range(max_validation_attempts):
                    same_previous_chapter = (
                        i > 0
                        and units[i - 1].metadata.get("chapter_index")
                        == unit.metadata.get("chapter_index")
                    )
                    unit_prompt_options = dict(attempt_options or {})
                    unit_prompt_options.setdefault("source_language", source_language)
                    unit_prompt_options.setdefault("target_language", target_language)
                    unit_prompt_options.update({
                        "_editor_llm_client": editor_llm_client,
                        "editor_provider_resolved": editor_provider,
                        "editor_model_resolved": editor_model,
                        "llm_provider": llm_provider,
                        "model": model_name,
                        "translation_id": self.translation_id,
                        "jobs_db_path": getattr(
                            getattr(self.checkpoint_manager, "db", None),
                            "db_path", None,
                        ),
                        "chunk_index": i,
                        "file_type": self.adapter.format_name,
                        "editor_phase": "translation",
                    })
                    if unit.metadata.get("dialogue_attribution"):
                        unit_prompt_options["dialogue_attribution"] = (
                            unit.metadata["dialogue_attribution"]
                        )
                    elif context_session and context_session.dialogue_attribution:
                        unit_prompt_options["dialogue_attribution"] = (
                            context_session.dialogue_attribution
                        )
                    if context_session:
                        from src.core.context import relevant_character_names
                        unit_prompt_options.setdefault(
                            "active_character_names",
                            relevant_character_names(
                                global_lore=context_session.global_lore,
                                source_text=unit.content,
                                dialogue_attribution=context_session.dialogue_attribution,
                                source_language=source_language,
                            ),
                        )
                    directed_context = build_directed_addressing_prompt_context(
                        translation_id=self.translation_id,
                        db=getattr(self.checkpoint_manager, "db", None),
                        target_language=target_language,
                        prompt_options=unit_prompt_options,
                        log_callback=log_callback,
                    )
                    if directed_context:
                        unit_prompt_options["directed_addressing_context"] = directed_context
                    relationship_context = build_relationship_prompt_context(
                        translation_id=self.translation_id,
                        db=getattr(self.checkpoint_manager, "db", None),
                        target_language=target_language,
                        prompt_options=unit_prompt_options,
                        reference_text="\n".join(
                            part for part in (
                                unit.context_before,
                                unit.content,
                                unit.context_after,
                            )
                            if part
                        ),
                        log_callback=log_callback,
                    )
                    if relationship_context:
                        unit_prompt_options["relationship_context"] = relationship_context
                    from src.utils.narrator_voice import build_narrator_voice_context
                    voice_db = getattr(self.checkpoint_manager, "db", None)
                    narrative_voice_context = build_narrator_voice_context(
                        self.translation_id, voice_db,
                        chunk_index=i,
                        target_language=target_language,
                    )
                    if narrative_voice_context:
                        unit_prompt_options["narrative_voice_context"] = narrative_voice_context
                    from src.core.context import build_unit_prompt_context
                    unit_prompt_options["prompt_context_bundle"] = (
                        build_unit_prompt_context(
                            addressing=directed_context,
                            relationships=relationship_context,
                            narrator=narrative_voice_context,
                            nearby_source=(unit.context_before, unit.context_after),
                        )
                    )
                    unit_prompt_options.update({
                        "_checkpoint_db": voice_db,
                        "chapter_index": getattr(unit, "chapter_index", None),
                        "scene_key": str(getattr(unit, "scene_key", "") or ""),
                    })
                    result = failed_editor_drafts_by_index.pop(i, None)
                    if result and log_callback:
                        log_callback(
                            "editor_draft_retry",
                            f"Retrying Senior Editor for preserved draft of unit {i+1}/{total_units}.",
                        )
                    if not result:
                        result = await generate_translation_request(
                            main_content=unit.content,
                            context_before=unit.context_before,
                            context_after=unit.context_after,
                            previous_translation_context=(
                                last_context
                                if (
                                    sequential
                                    and (
                                        not chapter_mode
                                        or i == 0
                                        or same_previous_chapter
                                    )
                                )
                                else ""
                            ),
                            source_language=source_language,
                            target_language=target_language,
                            model=model_name,
                            llm_client=llm_client,
                            log_callback=log_callback,
                            prompt_options=unit_prompt_options
                        )

                    # API failure / empty result: existing failure semantics.
                    if not result:
                        return result

                    # Optional 2-pass Senior Translation Editor reflection & repair pass
                    if result and (prompt_options or {}).get("reflection_mode"):
                        from src.core.translator import (
                            ReflectionValidationError,
                            run_chunk_reflection_pass,
                        )
                        try:
                            result = await run_chunk_reflection_pass(
                                source_chunk=unit.content,
                                draft_translation=result,
                                target_language=target_language,
                                model_name=model_name,
                                llm_client=llm_client,
                                novel_context=(prompt_options or {}).get("novel_context", ""),
                                custom_instructions=(prompt_options or {}).get("custom_instructions", ""),
                                glossary_block=(prompt_options or {}).get("glossary_block", ""),
                                log_callback=log_callback,
                                context_session=context_session,
                                prompt_options=unit_prompt_options,
                                repair_validator=lambda repaired, unit_id=unit.unit_id: (
                                    self.adapter.validate_unit_translation(
                                        unit_id, repaired
                                    )
                                ),
                            )
                        except ReflectionValidationError as exc:
                            result = UnitTranslationOutcome.review_required(
                                exc.draft_translation or result,
                                exc.diagnostics,
                            )

                    result_text = (
                        result.text if isinstance(result, UnitTranslationOutcome)
                        else result
                    )
                    feedback = self.adapter.validate_unit_translation(
                        unit.unit_id, result_text
                    )
                    if feedback is None:
                        return result

                    if log_callback:
                        log_callback("unit_validation_failed",
                            f"Unit {i+1}/{total_units}: {feedback} "
                            f"(attempt {attempt+1}/{max_validation_attempts})")

                    reinforced = (
                        f"CRITICAL: Your previous response was structurally "
                        f"incomplete ({feedback}). You MUST reproduce every "
                        f"[N] index marker from the input exactly once, in "
                        f"order, each followed by its translation. Do NOT "
                        f"merge, drop or renumber markers."
                    )
                    attempt_options = {
                        **(prompt_options or {}),
                        'custom_instructions': (
                            f"{base_instructions}\n\n{reinforced}"
                            if base_instructions else reinforced
                        ),
                    }

                if log_callback:
                    log_callback("unit_validation_exhausted",
                        f"Unit {i+1}/{total_units} still incomplete after "
                        f"{max_validation_attempts} attempts — keeping valid "
                        f"parts and marking the unit failed")
                return _ValidationFailed(result)

            async def _save_partial_and_pause(at_index):
                if log_callback:
                    log_callback("translation_interrupted",
                        f"Translation interrupted at unit {at_index+1}/{total_units}")
                # Try to save partial output for TXT/SRT (fast reconstruction).
                # For EPUB, partial output may not be valid, so we skip it.
                if self.adapter.format_name in ['txt', 'srt']:
                    try:
                        if log_callback:
                            log_callback("reconstruct_partial", "Saving partial output before interruption")
                        output_bytes = await self.adapter.reconstruct_output(bilingual=bilingual_output)
                        with open(self.adapter.output_file_path, 'wb') as f:
                            f.write(output_bytes)
                    except Exception as e:
                        if log_callback:
                            log_callback("reconstruct_partial_failed",
                                f"Could not save partial output: {str(e)}")
                self.checkpoint_manager.mark_paused(self.translation_id)

            def _record_failure(i, unit, error=None):
                nonlocal failed_count
                if log_callback:
                    log_callback("unit_failed", f"Failed to translate unit {i+1}/{total_units}")
                pending_db_addressing_context_by_index.pop(i, None)
                pending_relationship_context_by_index.pop(i, None)
                if i not in failed_indices:
                    failed_count += 1
                    failed_indices.add(i)
                diagnostics = getattr(error, "diagnostics", None)
                preserved_draft = getattr(error, "draft_translation", "")
                chunk_data = dict(unit.metadata or {})
                chunk_data["quality_status"] = "not_checked"
                chunk_data["execution_failure_class"] = (
                    "structure" if isinstance(error, _ValidationFailed)
                    else "translation"
                )
                if diagnostics:
                    chunk_data["editor_validation"] = diagnostics
                    failed_editor_drafts_by_index[i] = preserved_draft
                    if log_callback:
                        log_callback(
                            "editor_validation_diagnostics",
                            "Senior Editor validation reasons: "
                            + "; ".join(
                                str(item)
                                for item in diagnostics.get("final_reason_codes", [])[:3]
                            ),
                        )
                self.checkpoint_manager.save_checkpoint(
                    translation_id=self.translation_id,
                    chunk_index=i,
                    original_text=unit.content,
                    translated_text=preserved_draft or None,
                    chunk_data=chunk_data,
                    total_chunks=total_units,
                    failed_chunks=failed_count,
                    chunk_status='failed',
                )
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': completed_count,
                        'failed_chunks': failed_count
                    })

            def _commit_db_addressing(i):
                if not (
                    context_session
                    and self.translation_id
                    and getattr(self.checkpoint_manager, "db", None)
                    and prompt_options.get("use_db_directed_addressing", True)
                ):
                    return
                pending = pending_db_addressing_context_by_index.pop(i, None)
                if not pending:
                    return
                sync_context_update_addressing_to_db(
                    translation_id=self.translation_id,
                    db=self.checkpoint_manager.db,
                    updated_context_or_dynamic_state=pending["content"],
                    target_language=target_language,
                    chunk_index=i,
                    log_callback=log_callback,
                    dialogue_attribution=pending.get("dialogue_attribution"),
                    candidates=pending.get("candidates"),
                    source_text=pending.get("source_text", ""),
                    source_language=source_language,
                    contract_version=contract_version,
                )
                apply_db_addressing_to_session(
                    context_session,
                    self.translation_id,
                    self.checkpoint_manager.db,
                )

            def _commit_relationship_context(i):
                if not (
                    context_session
                    and self.translation_id
                    and getattr(self.checkpoint_manager, "db", None)
                    and relationship_mode != "off"
                ):
                    return
                pending_relationship = pending_relationship_context_by_index.pop(i, None)
                if not pending_relationship:
                    return
                sync_context_update_relationships_to_db(
                    translation_id=self.translation_id,
                    db=self.checkpoint_manager.db,
                    updated_context_or_dynamic_state=pending_relationship["content"],
                    source_text=pending_relationship["source_text"],
                    candidates=pending_relationship.get("candidates"),
                    parser_status=pending_relationship.get("parser_status", "absent"),
                    target_language=target_language,
                    chunk_index=i,
                    active_character_names=pending_relationship.get("active_character_names"),
                    log_callback=log_callback,
                )
                apply_relationship_graph_to_session(
                    context_session,
                    self.translation_id,
                    self.checkpoint_manager.db,
                )

            async def _sync_successful_translated_context(i, unit, translated_content):
                if not (auto_update_context and context_session):
                    return
                try:
                    await context_session.sync_translated_output(
                        translated_chunk=translated_content,
                        source_chunk=unit.content,
                        target_language=target_language,
                        source_language=source_language,
                        llm_client=llm_client,
                        model_name=model_name,
                        chunk_index=i + 1,
                        total_chunks=total_units,
                    )
                    judged_candidates = []
                    if (
                        self.translation_id
                        and getattr(self.checkpoint_manager, "db", None)
                        and relationship_mode != "off"
                    ):
                        locked_facts = [
                            edge for edge in self.checkpoint_manager.db.get_relationship_edges(
                                self.translation_id,
                                statuses=["accepted"],
                            )
                            if edge.get("is_locked")
                        ]
                        judged_candidates = await judge_ambiguous_relationship_candidates(
                            llm_client=llm_client,
                            candidates=context_session.relationship_candidates,
                            source_text=unit.content,
                            model_name=model_name,
                            enabled=(
                                prompt_options.get("use_relationship_llm_judge")
                                in {True, "selective", "always"}
                            ),
                            locked_facts=locked_facts,
                            log_callback=log_callback,
                        )
                    db = getattr(self.checkpoint_manager, "db", None)
                    if self.translation_id and db is not None:
                        with db.context_state_transaction():
                            _commit_db_addressing(i)
                            _commit_relationship_context(i)
                            if prompt_options.get(
                                "use_db_directed_addressing",
                                True,
                            ):
                                sync_context_update_addressing_to_db(
                                    translation_id=self.translation_id,
                                    db=db,
                                    updated_context_or_dynamic_state=context_session.content,
                                    target_language=target_language,
                                    chunk_index=i,
                                    log_callback=log_callback,
                                    dialogue_attribution=context_session.dialogue_attribution,
                                    candidates=context_session.addressing_candidates,
                                    source_text=unit.content,
                                    source_language=source_language,
                                    contract_version=contract_version,
                                )
                            if relationship_mode != "off":
                                from src.core.context import relevant_character_names

                                sync_context_update_relationships_to_db(
                                    translation_id=self.translation_id,
                                    db=db,
                                    updated_context_or_dynamic_state=context_session.content,
                                    source_text=unit.content,
                                    candidates=judged_candidates,
                                    parser_status=context_session.relationship_parse_status,
                                    target_language=target_language,
                                    chunk_index=i,
                                    active_character_names=relevant_character_names(
                                        global_lore=context_session.global_lore,
                                        source_text=unit.content,
                                        dialogue_attribution=context_session.dialogue_attribution,
                                        source_language=source_language,
                                    ),
                                    log_callback=log_callback,
                                )
                        if prompt_options.get("use_db_directed_addressing", True):
                            apply_db_addressing_to_session(
                                context_session,
                                self.translation_id,
                                db,
                            )
                        if relationship_mode != "off":
                            apply_relationship_graph_to_session(
                                context_session,
                                self.translation_id,
                                db,
                            )
                    context_session.save()
                    if log_callback and contract_version >= 2:
                        emit_progress_log(
                            log_callback,
                            "novel_context_state",
                            "Context committed",
                            layer="novel_context",
                            data={
                                "type": "novel_context_state",
                                "content": context_session.content,
                                "filename": context_session.path.name,
                            },
                        )
                except Exception as sync_err:
                    if log_callback:
                        log_callback(
                            "context_sync_failed",
                            f"Failed to sync translated unit into novel context: {sync_err}",
                        )

            def _commit_context(i, unit):
                if not context_session:
                    return
                if unit.metadata is None:
                    unit.metadata = {}
                if i in reused_context_data_by_index:
                    unit.metadata.update(reused_context_data_by_index[i])
                    return
                unit.metadata['dialogue_attribution'] = (
                    context_session.dialogue_attribution
                )
                unit.metadata['context_snapshot'] = context_session.snapshot()

            # Everything without a committed translation is (re)translated:
            # never-attempted units AND previously failed ones.
            pending = [i for i in range(total_units) if i not in restored_completed]
            rate_limit_error = None
            remaining = len(pending)
            # First not-yet-committed index; used for the pause log message.
            next_index = pending[0] if pending else total_units

            # Continuous concurrency: up to `workers` requests in flight at once,
            # results delivered strictly in index order so checkpoints stay
            # contiguous. should_interrupt stops launching new units; already
            # in-flight ones still complete and commit.
            async for i, result in iter_ordered_concurrent(
                pending, workers, _translate_unit, check_interruption_callback
            ):
                unit = units[i]

                if isinstance(result, RateLimitError):
                    # Stop before committing this unit so resume restarts at it.
                    rate_limit_error = result
                    break

                remaining -= 1

                if isinstance(result, Exception):
                    if log_callback:
                        log_callback("unit_error",
                            f"Error translating unit {i+1}/{total_units}: {str(result)}")
                    _record_failure(i, unit, result)
                    next_index = i + 1
                    continue

                if isinstance(result, _ValidationFailed):
                    # Keep whatever parsed (e.g. the cues whose markers came
                    # back) so the output stays best-effort, but record the
                    # unit as failed: the job ends 'partial' and the unit is
                    # fully retranslated on retry (issue #204 mechanics).
                    await self.adapter.save_unit_translation(
                        unit.unit_id, result.content
                    )
                    _record_failure(i, unit, result)
                    next_index = i + 1
                    continue

                translated_content = result
                outcome = (
                    result if isinstance(result, UnitTranslationOutcome)
                    else UnitTranslationOutcome.completed(
                        str(result),
                        quality_status=getattr(result, 'quality_status', 'passed'),
                        diagnostics=getattr(result, 'editor_validation', None),
                    ) if result else UnitTranslationOutcome.failed(
                        failure_class='content'
                    )
                )
                translated_content = outcome.text
                if translated_content:
                    save_success = await self.adapter.save_unit_translation(
                        unit.unit_id, translated_content
                    )
                    if not save_success:
                        if log_callback:
                            log_callback("save_failed",
                                f"Failed to save translation for unit {unit.unit_id}")
                        _record_failure(i, unit, result)
                        next_index = i + 1
                        continue

                    completed_count += 1
                    failed_indices.discard(i)
                    failed_count = len(failed_indices)
                    if outcome.quality_status == 'review_required':
                        review_required_indices.add(i)
                    else:
                        review_required_indices.discard(i)
                    if unit.metadata is None:
                        unit.metadata = {}
                    unit.metadata['quality_status'] = outcome.quality_status
                    unit.metadata.pop('execution_failure_class', None)
                    if outcome.diagnostics:
                        unit.metadata['editor_validation'] = outcome.diagnostics
                    await _sync_successful_translated_context(
                        i,
                        unit,
                        translated_content,
                    )
                    _commit_context(i, unit)

                    self.checkpoint_manager.save_checkpoint(
                        translation_id=self.translation_id,
                        chunk_index=i,
                        original_text=unit.content,
                        translated_text=translated_content,
                        chunk_data=unit.metadata,
                        total_chunks=total_units,
                        completed_chunks=completed_count,
                        failed_chunks=failed_count,
                    )
                    if stats_callback:
                        stats_callback({
                            'total_chunks': total_units,
                            'completed_chunks': completed_count,
                            'failed_chunks': failed_count,
                            'review_required_chunks': len(review_required_indices),
                        })

                    if sequential:
                        completed_translations_by_index[i] = translated_content
                        last_context = (
                            translated_content[-200:]
                            if len(translated_content) > 200
                            else translated_content
                        )

                    if log_callback:
                        log_callback("unit_complete",
                            f"Unit {i+1}/{total_units} translated successfully")
                else:
                    _record_failure(i, unit)

                next_index = i + 1

            if rate_limit_error is not None:
                # Re-raise to trigger auto-pause (handled by the caller).
                raise rate_limit_error

            # If the scheduler stopped early because interruption was requested,
            # the committed units are persisted; save partial output and pause.
            if (remaining > 0
                    and check_interruption_callback and check_interruption_callback()):
                await _save_partial_and_pause(next_index)
                return False

            if failed_indices:
                retry_targets = sorted(failed_indices)
                if log_callback:
                    log_callback(
                        "failed_unit_retry_start",
                        f"Retrying {len(retry_targets)} failed unit(s) before final output...",
                    )
                for i in retry_targets:
                    if check_interruption_callback and check_interruption_callback():
                        await _save_partial_and_pause(i)
                        return False

                    unit = units[i]
                    result = await _translate_unit(i, analyze_context=False)

                    if isinstance(result, RateLimitError):
                        raise result
                    if isinstance(result, Exception):
                        if log_callback:
                            log_callback(
                                "unit_error",
                                f"Retry failed for unit {i+1}/{total_units}: {str(result)}",
                            )
                        _record_failure(i, unit, result)
                        continue

                    if isinstance(result, _ValidationFailed):
                        await self.adapter.save_unit_translation(
                            unit.unit_id, result.content
                        )
                        _record_failure(i, unit, result)
                        continue

                    if not result:
                        _record_failure(i, unit, result)
                        continue
                    outcome = (
                        result if isinstance(result, UnitTranslationOutcome)
                        else UnitTranslationOutcome.completed(
                            str(result),
                            quality_status=getattr(result, 'quality_status', 'passed'),
                            diagnostics=getattr(result, 'editor_validation', None),
                        )
                    )
                    translated_content = outcome.text
                    if not translated_content:
                        _record_failure(i, unit)
                        continue

                    save_success = await self.adapter.save_unit_translation(
                        unit.unit_id, translated_content
                    )
                    if not save_success:
                        if log_callback:
                            log_callback(
                                "save_failed",
                                f"Failed to save retry translation for unit {unit.unit_id}",
                            )
                        _record_failure(i, unit)
                        continue

                    _commit_db_addressing(i)
                    _commit_relationship_context(i)
                    _commit_context(i, unit)
                    failed_indices.discard(i)
                    failed_count = len(failed_indices)
                    completed_count += 1
                    if outcome.quality_status == 'review_required':
                        review_required_indices.add(i)
                    else:
                        review_required_indices.discard(i)
                    if unit.metadata is None:
                        unit.metadata = {}
                    unit.metadata['quality_status'] = outcome.quality_status
                    unit.metadata.pop('execution_failure_class', None)
                    if outcome.diagnostics:
                        unit.metadata['editor_validation'] = outcome.diagnostics
                    self.checkpoint_manager.save_checkpoint(
                        translation_id=self.translation_id,
                        chunk_index=i,
                        original_text=unit.content,
                        translated_text=translated_content,
                        chunk_data=unit.metadata,
                        total_chunks=total_units,
                        completed_chunks=completed_count,
                        failed_chunks=failed_count,
                    )
                    if stats_callback:
                        stats_callback({
                            'total_chunks': total_units,
                            'completed_chunks': completed_count,
                            'failed_chunks': failed_count,
                            'review_required_chunks': len(review_required_indices),
                        })
                    if log_callback:
                        log_callback(
                            "failed_unit_retry_success",
                            f"Failed unit {i+1}/{total_units} translated successfully on retry.",
                        )

            # 7. Reconstruct output file
            if log_callback:
                log_callback("reconstruct_start", "Reconstructing output file")

            try:
                output_bytes = await self.adapter.reconstruct_output(bilingual=bilingual_output)

                # Save final file
                with open(self.adapter.output_file_path, 'wb') as f:
                    f.write(output_bytes)

                if log_callback:
                    log_callback("reconstruct_complete",
                        f"Output file written to {self.adapter.output_file_path}")

            except Exception as e:
                if log_callback:
                    log_callback("reconstruct_failed",
                        f"Failed to reconstruct output: {str(e)}")
                return False

            # 8. Cleanup
            await self.adapter.cleanup()

            # 9. Mark job as completed only when every unit is genuinely
            # translated; otherwise keep the checkpoint resumable so the
            # failed units can be retried (issue #204).
            if failed_count == 0 and completed_count == total_units:
                self.checkpoint_manager.mark_completed(self.translation_id)
                if log_callback:
                    log_callback("translation_complete",
                        f"Translation completed successfully: {total_units} units")
                return True
            else:
                self.checkpoint_manager.mark_partial(self.translation_id)
                if log_callback:
                    log_callback("translation_partial",
                        f"Translation completed with {failed_count} failures out of "
                        f"{total_units} units — checkpoint kept for retry")
                return False

        except Exception as e:
            # Re-raise RateLimitError to trigger auto-pause
            from src.core.llm.exceptions import RateLimitError
            if isinstance(e, RateLimitError):
                raise
            if log_callback:
                log_callback("translation_error", f"Translation error: {str(e)}")
            return False
        finally:
            # Ensure cleanup even on error
            try:
                await self.adapter.cleanup()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"GenericTranslator("
            f"id={self.translation_id}, "
            f"adapter={self.adapter})"
        )

def resync_context_snapshots_background(
    translation_id: str,
    start_chunk_index: int,
    initial_compressed_snapshot: str,
    socketio=None,
    was_active=False,
    auto_resume_callback=None,
    post_resync_callback=None,
    post_resync_message=None,
    global_only_resync=False,
):
    """Entry point for the background thread."""
    import asyncio
    
    # Try to use existing loop if we are in one, otherwise run new
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                _resync_context_snapshots_async(
                    translation_id,
                    start_chunk_index,
                    initial_compressed_snapshot,
                    socketio,
                    was_active,
                    auto_resume_callback,
                    post_resync_callback,
                    post_resync_message,
                    global_only_resync,
                ),
                loop,
            )
            return future.result()
    except RuntimeError:
        pass
        
    return asyncio.run(
        _resync_context_snapshots_async(
            translation_id,
            start_chunk_index,
            initial_compressed_snapshot,
            socketio,
            was_active,
            auto_resume_callback,
            post_resync_callback,
            post_resync_message,
            global_only_resync,
        )
    )

async def _resync_context_snapshots_async(
    translation_id: str,
    start_chunk_index: int,
    initial_compressed_snapshot: str,
    socketio=None,
    was_active=False,
    auto_resume_callback=None,
    post_resync_callback=None,
    post_resync_message=None,
    global_only_resync=False,
):
    """Forward-pass through chunks to re-evaluate the context using the LLM."""
    from src.api.translation_state import get_state_manager
    from src.utils.novel_context import (
        build_novel_context,
        character_alias_map,
        compress_dynamic_state,
        decode_context_snapshot,
        load_novel_context,
        normalize_novel_context_filename,
        resolve_novel_context_path,
        save_novel_context,
        update_novel_context_chunk,
        _bounded_source_memory,
    )
    from src.core.llm_client import LLMClient
    from src.config import NOVEL_CONTEXTS_DIR
    from datetime import datetime
    import copy
    import time
    from src.utils.unified_logger import get_logger
    from src.api.websocket import emit_update
    from src.utils.relationship_sync import (
        judge_ambiguous_relationship_candidates,
        resolve_relationship_reasoning_mode,
        sync_context_update_relationships_to_db,
        sync_markdown_relationships_to_db,
    )
    from src.utils.db_addressing import (
        sync_context_update_addressing_to_db,
        sync_markdown_addressing_to_db,
    )
    
    state_manager = get_state_manager()
    logger = get_logger("context_resync")
    
    def _load_resync_state():
        job = state_manager.checkpoint_manager.get_job(translation_id) or {}
        if not isinstance(job, dict):
            job = {}
        config = copy.deepcopy(job.get('config') or {})
        state = config.get('_context_resync')
        return config, dict(state) if isinstance(state, dict) else {}

    def _save_resync_state(updates):
        config, state = _load_resync_state()
        state.update(updates)
        state["updated_at"] = time.time()
        config['_context_resync'] = state
        state_manager.checkpoint_manager.update_job_config(
            translation_id,
            config,
        )
        return state

    def _pause_requested():
        _, state = _load_resync_state()
        return bool(state.get("pause_requested"))

    def append_and_emit(msg_str, resync_state=None):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{timestamp}] [context_resync] {msg_str}"
        logger.info(f"[context_resync] {msg_str}")
        if resync_state is None:
            _, resync_state = _load_resync_state()
        if not state_manager.exists(translation_id):
            state_manager.restore_job_from_checkpoint(translation_id)
        state_manager.append_log(translation_id, full_msg)
        if socketio:
            data = {
                "ui_step": "context_resync",
                "phase": "context",
            }
            if resync_state:
                data["resync_status"] = resync_state.get("status")
                data["resync_state"] = resync_state
            structured_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "general",
                "message": msg_str,
                "data": data,
            }
            emit_update(
                socketio,
                translation_id,
                {
                    "log": msg_str,
                    "log_entry": structured_entry,
                },
                state_manager,
            )

    def should_log_resync_progress(processed_count, total_count):
        return True

    def run_follow_up():
        callback = post_resync_callback or auto_resume_callback
        if not callback:
            return
        message = post_resync_message or "Auto-resuming active translation..."
        append_and_emit(f"▶️ {message}")
        callback()
    
    if global_only_resync:
        msg = (
            f"Starting global context propagation for {translation_id} from "
            f"saved snapshot at chunk {start_chunk_index + 1}; later chunks "
            "will keep their saved dynamic state."
        )
    else:
        msg = (
            f"Starting context resync for {translation_id} from saved snapshot "
            f"at chunk {start_chunk_index + 1}; later chunks will be replayed "
            "from that context."
        )
    append_and_emit(
        f"🔄 {msg}",
        _save_resync_state({
            "status": "running",
            "last_processed_chunk": start_chunk_index,
        }),
    )
    
    if was_active:
        msg_pause = "Waiting for active translation to pause before resyncing..."
        append_and_emit(
            f"⏸️ {msg_pause}",
            _save_resync_state({
                "status": "pause_requested",
                "last_processed_chunk": start_chunk_index,
            }),
        )
        
        import asyncio
        paused = False
        live_job = state_manager.get_translation(translation_id) or {}
        live_config = live_job.get("config") or {}
        try:
            pause_timeout = int(live_config.get("request_timeout", 120)) + 15
        except (TypeError, ValueError):
            pause_timeout = 135
        pause_timeout = max(60, min(pause_timeout, 600))
        for _ in range(pause_timeout):
            status_dict = state_manager.get_translation(translation_id)
            status = status_dict.get('status') if status_dict else None
            if not status and hasattr(state_manager.checkpoint_manager, 'get_job'):
                persisted_job = state_manager.checkpoint_manager.get_job(translation_id)
                status = persisted_job.get('status') if persisted_job else None
            if status in (
                'paused', 'interrupted', 'partial', 'completed', 'failed', 'error'
            ):
                paused = True
                break
            await asyncio.sleep(1)
        if not paused:
            err_msg = (
                "Active translation did not pause within "
                f"{pause_timeout} seconds; resync aborted to avoid racing "
                "with translation."
            )
            logger.error(err_msg)
            append_and_emit(f"❌ {err_msg}")
            return False
    
    checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint_data:
        append_and_emit("❌ Translation checkpoint is no longer available.")
        return False
        
    config = checkpoint_data.get('job', {}).get('config', {})
    chunks = checkpoint_data.get('chunks', [])
    
    # Re-sync is source-derived, so it may walk failed/partial chunks too.
    # Their translated output remains retryable; only their source facts feed
    # later context snapshots.
    completed_chunks = [
        c for c in chunks
        if c.get('status') in ('completed', 'partial', 'failed')
        and c.get('chunk_index') is not None
    ]
    completed_chunks.sort(key=lambda x: x['chunk_index'])
    
    chunks_to_process = [c for c in completed_chunks if c['chunk_index'] > start_chunk_index]
    _save_resync_state({
        "status": "running",
        "start_chunk_index": start_chunk_index,
        "last_processed_chunk": start_chunk_index,
        "total_chunks": len(completed_chunks),
    })
    if chunks_to_process:
        if global_only_resync:
            append_and_emit(
                "Using the edited global lore as the base; propagating it to "
                f"{len(chunks_to_process)} later saved snapshot(s), "
                f"{chunks_to_process[0]['chunk_index'] + 1}-"
                f"{chunks_to_process[-1]['chunk_index'] + 1}."
            )
        else:
            append_and_emit(
                "Using the selected context snapshot as the base; replaying "
                f"{len(chunks_to_process)} later chunk(s), "
                f"{chunks_to_process[0]['chunk_index'] + 1}-"
                f"{chunks_to_process[-1]['chunk_index'] + 1}."
            )
    
    novel_context_file = config.get('prompt_options', {}).get('novel_context_file')
    path = None
    fallback_context = ""
    if novel_context_file:
        try:
            novel_context_file = normalize_novel_context_filename(novel_context_file)
            path = resolve_novel_context_path(novel_context_file, NOVEL_CONTEXTS_DIR)
            fallback_context = load_novel_context(path.name, path.parent)
        except Exception as e:
            err_msg = f"Failed to load global lore: {e}"
            logger.error(err_msg)
            append_and_emit(f"❌ {err_msg}")
            return False

    current_full_context, global_lore, current_dynamic_text = decode_context_snapshot(
        initial_compressed_snapshot,
        fallback_context,
    )
    relationship_mode = resolve_relationship_reasoning_mode(
        config.get('prompt_options') or {}
    )
    db = state_manager.checkpoint_manager.db
    run_id = f"{translation_id}:{start_chunk_index}:{int(time.time() * 1000)}"
    if not db.create_context_resync_run(
        run_id,
        translation_id,
        start_chunk_index,
        initial_compressed_snapshot,
    ):
        append_and_emit("❌ Could not create atomic context resync staging state.")
        return False

    def activate_staged_timeline(final_context):
        staged = db.get_context_resync_stage(run_id)
        options = config.get('prompt_options') or {}
        contract_version = int(options.get("context_contract_version", 1) or 1)
        with db.context_state_transaction():
            if not global_only_resync:
                db.quarantine_narrator_voice_after(
                    translation_id, start_chunk_index,
                )
                for rule in db.get_addressing_rules(translation_id):
                    if not rule.get("is_locked") and int(rule.get("last_chunk_index") or 0) >= start_chunk_index:
                        db.delete_addressing_rule(
                            translation_id,
                            str(rule.get("speaker_name") or ""),
                            str(rule.get("addressee_name") or ""),
                        )
                for edge in db.get_relationship_edges(translation_id):
                    if not edge.get("is_locked") and int(edge.get("last_chunk_index") or 0) >= start_chunk_index:
                        db.set_relationship_edge_status(
                            translation_id,
                            edge["id"],
                            "superseded",
                        )
            if relationship_mode != "off":
                sync_markdown_relationships_to_db(
                    translation_id=translation_id,
                    db=db,
                    context_or_dynamic_state=current_full_context,
                    target_language=config.get('target_language') or "",
                    chunk_index=start_chunk_index,
                    trigger_source="manual_context",
                )
            if options.get("use_db_directed_addressing", True):
                sync_markdown_addressing_to_db(
                    translation_id=translation_id,
                    db=db,
                    context_or_dynamic_state=current_full_context,
                    target_language=config.get('target_language') or "",
                    chunk_index=start_chunk_index,
                    trigger_source="manual_context",
                )
            for item in staged:
                db.save_chunk(
                    translation_id=translation_id,
                    chunk_index=item["chunk_index"],
                    original_text=item.get("original_text"),
                    translated_text=item.get("translated_text"),
                    chunk_data=item.get("chunk_data") or {},
                    status=item.get("status") or "completed",
                )
                if item.get("status") != "completed" or global_only_resync:
                    continue
                snapshot_context, _snapshot_global, _snapshot_dynamic = decode_context_snapshot(
                    (item.get("chunk_data") or {}).get("context_snapshot"),
                    final_context,
                )
                dialogue = (item.get("chunk_data") or {}).get("dialogue_attribution") or {}
                if relationship_mode != "off":
                    sync_context_update_relationships_to_db(
                        translation_id=translation_id,
                        db=db,
                        updated_context_or_dynamic_state=snapshot_context,
                        source_text=item.get("original_text") or "",
                        candidates=item.get("relationship_candidates") or [],
                        parser_status=item.get("parser_status") or "absent",
                        target_language=config.get('target_language') or "",
                        chunk_index=item["chunk_index"],
                        active_character_names=list(
                            (dialogue.get("state_after") or {}).values()
                        ),
                    )
                if options.get("use_db_directed_addressing", True):
                    sync_context_update_addressing_to_db(
                        translation_id=translation_id,
                        db=db,
                        updated_context_or_dynamic_state=snapshot_context,
                        target_language=config.get('target_language') or "",
                        chunk_index=item["chunk_index"],
                        dialogue_attribution=dialogue,
                        candidates=item.get("addressing_candidates") or [],
                        source_text=item.get("original_text") or "",
                        source_language=config.get('source_language') or "",
                        contract_version=contract_version,
                    )
        db.finish_context_resync_run(
            run_id,
            "completed",
            final_context=final_context,
        )
    initial_chunk = next(
        (
            chunk for chunk in completed_chunks
            if chunk.get('chunk_index') == start_chunk_index
        ),
        None,
    )
    current_dialogue_state = (
        (
            ((initial_chunk or {}).get('chunk_data') or {}).get(
                'dialogue_attribution'
            ) or {}
        ).get('state_after')
        or {}
    )
    from src.utils.dialogue_attribution import canonicalize_dialogue_state
    current_dialogue_state = canonicalize_dialogue_state(
        current_dialogue_state,
        character_alias_map(global_lore),
    )
    current_dialogue_scene_key = (
        (
            ((initial_chunk or {}).get('chunk_data') or {}).get(
                'dialogue_attribution'
            ) or {}
        ).get('scene_key')
    )

    if global_only_resync and chunks_to_process:
        missing_snapshot = next(
            (
                c for c in chunks_to_process
                if not ((c.get('chunk_data') or {}).get('context_snapshot'))
            ),
            None,
        )
        if missing_snapshot:
            append_and_emit(
                "Saved dynamic state is missing for a later chunk; falling "
                "back to LLM replay for timeline safety."
            )
        else:
            latest_full_context = current_full_context
            total_to_process = len(chunks_to_process)
            for processed_count, chunk in enumerate(chunks_to_process, start=1):
                idx = chunk['chunk_index']
                if _pause_requested():
                    paused_state = _save_resync_state({
                        "status": "paused",
                        "pause_requested": False,
                        "last_processed_chunk": max(start_chunk_index, idx - 1),
                    })
                    append_and_emit(
                        "⏸️ Context resync paused. Resume it after changing model settings.",
                        paused_state,
                    )
                    return False

                try:
                    latest_cp = state_manager.checkpoint_manager.load_checkpoint(
                        translation_id
                    )
                    if not latest_cp:
                        append_and_emit(
                            "❌ Translation checkpoint disappeared during resync."
                        )
                        return False

                    snapshot_saved = False
                    for c in latest_cp['chunks']:
                        if c.get('chunk_index') != idx:
                            continue
                        if c.get('chunk_data') is None:
                            c['chunk_data'] = {}
                        existing_snapshot = c['chunk_data'].get(
                            'context_snapshot'
                        )
                        _, _, existing_dynamic_text = decode_context_snapshot(
                            existing_snapshot,
                            fallback_context,
                        )
                        latest_full_context = build_novel_context(
                            global_lore,
                            existing_dynamic_text,
                        )
                        c['chunk_data']['context_snapshot'] = (
                            compress_dynamic_state(latest_full_context)
                        )
                        db.stage_context_resync_chunk(
                            run_id,
                            translation_id,
                            {
                                "chunk_index": idx,
                                "original_text": c.get('original_text'),
                                "translated_text": c.get('translated_text'),
                                "chunk_data": c.get('chunk_data'),
                                "status": c.get('status') or 'completed',
                            },
                        )
                        snapshot_saved = True
                        progress_state = _save_resync_state({
                            "status": "running",
                            "last_processed_chunk": idx,
                        })
                        if should_log_resync_progress(
                            processed_count,
                            total_to_process,
                        ):
                            append_and_emit(
                                "🔄 Global context propagation progress: "
                                f"{processed_count}/{total_to_process} "
                                f"saved snapshot(s) updated "
                                f"(latest chunk {idx + 1}).",
                                progress_state,
                            )
                        break
                    if not snapshot_saved:
                        append_and_emit(f"❌ Chunk {idx} disappeared during resync.")
                        return False
                except Exception as e:
                    err_msg = f"Global context propagation failed at chunk {idx}: {e}"
                    logger.error(err_msg)
                    _save_resync_state({
                        "status": "failed",
                        "pause_requested": False,
                        "last_processed_chunk": max(start_chunk_index, idx - 1),
                        "error": str(e),
                    })
                    append_and_emit(f"❌ {err_msg}")
                    return False

                if _pause_requested():
                    paused_state = _save_resync_state({
                        "status": "paused",
                        "pause_requested": False,
                        "last_processed_chunk": idx,
                    })
                    append_and_emit(
                        "⏸️ Context resync paused. Resume it after changing model settings.",
                        paused_state,
                    )
                    return False

            activate_staged_timeline(latest_full_context)
            if path:
                try:
                    save_novel_context(path.name, path.parent, latest_full_context)
                    append_and_emit(
                        f"✅ Context file '{path.name}' updated successfully."
                    )
                except Exception as e:
                    err_msg = f"Failed to update context file: {e}"
                    logger.error(err_msg)
                    _save_resync_state({"status": "failed", "error": err_msg})
                    append_and_emit(f"❌ {err_msg}")
                    return False

            msg_end = "Global context propagation completed."
            logger.info(msg_end)
            completed_state = _save_resync_state({
                "status": "completed",
                "pause_requested": False,
                "last_processed_chunk": chunks_to_process[-1]['chunk_index'],
            })
            append_and_emit(f"✅ {msg_end}", completed_state)
            run_follow_up()
            return True

    if not chunks_to_process:
        latest_completed = [c['chunk_index'] for c in completed_chunks]
        if not latest_completed or start_chunk_index != max(latest_completed):
            append_and_emit("❌ The selected context snapshot is not available.")
            return False
        activate_staged_timeline(current_full_context)
        if path:
            try:
                save_novel_context(path.name, path.parent, current_full_context)
                append_and_emit(f"✅ Context file '{path.name}' updated successfully.")
            except Exception as e:
                err_msg = f"Failed to update context file '{path.name}': {e}"
                logger.error(err_msg)
                _save_resync_state({"status": "failed", "error": err_msg})
                append_and_emit(f"❌ {err_msg}")
                return False
        completed_state = _save_resync_state({
            "status": "completed",
            "pause_requested": False,
            "last_processed_chunk": start_chunk_index,
        })
        append_and_emit("✅ Background context resync completed.", completed_state)
        run_follow_up()
        return True

    llm_provider = config.get('llm_provider', 'ollama')
    model_name = config.get('model') or config.get('model_name')
    provider_key = config.get(f'{llm_provider}_api_key')
    live_job = state_manager.get_translation(translation_id) or {}
    live_config = live_job.get('config') or {}
    if not provider_key:
        provider_key = live_config.get(f'{llm_provider}_api_key')
    if not provider_key:
        import os
        import src.config as runtime_config

        env_var = f"{llm_provider.upper()}_API_KEY"
        provider_key = (
            os.getenv(env_var)
            or getattr(runtime_config, env_var, '')
        )

    if llm_provider not in ('ollama', 'openai') and not provider_key:
        err_msg = (
            f"Cannot re-sync context with '{llm_provider}': the live API key "
            "is unavailable."
        )
        logger.error(err_msg)
        append_and_emit(f"❌ {err_msg}")
        return False

    llm_kwargs = {
        'api_endpoint': config.get('llm_api_endpoint'),
        'api_key': provider_key,
        'timeout': config.get('request_timeout', 120),
    }
    try:
        llm_client = LLMClient(provider_type=llm_provider, model=model_name, **llm_kwargs)
    except Exception as e:
        err_msg = f"Failed to init LLM client for resync: {e}"
        logger.error(err_msg)
        append_and_emit(f"❌ {err_msg}")
        return False
    
    source_memory_chunks = [
        c.get('original_text') or ''
        for c in completed_chunks
        if c.get('chunk_index', -1) <= start_chunk_index
    ]
    new_full_context = current_full_context
    for chunk in chunks_to_process:
        idx = chunk['chunk_index']
        source_text = chunk.get('original_text') or ''
        if _pause_requested():
            paused_state = _save_resync_state({
                "status": "paused",
                "pause_requested": False,
                "last_processed_chunk": max(start_chunk_index, idx - 1),
            })
            append_and_emit(
                "⏸️ Context resync paused. Resume it after changing model settings.",
                paused_state,
            )
            return False
        if (chunk.get('status') or 'completed') != 'completed':
            db.stage_context_resync_chunk(
                run_id,
                translation_id,
                {
                    "chunk_index": idx,
                    "original_text": chunk.get('original_text'),
                    "translated_text": chunk.get('translated_text'),
                    "chunk_data": chunk.get('chunk_data') or {},
                    "status": chunk.get('status') or 'failed',
                },
            )
            append_and_emit(
                f"Skipped context state from failed chunk {idx + 1}; it will be reconsidered after a successful retry."
            )
            continue
        
        try:
            msg_resync = f"Resyncing chunk {idx + 1} from the saved context timeline..."
            append_and_emit(f"🔄 {msg_resync}")
            from src.utils.dialogue_attribution import (
                canonicalize_dialogue_state,
                detect_dialogue_turns,
                dialogue_attribution_stats,
                empty_dialogue_attribution,
            )

            dialogue_sink = {}
            relationship_sink = {}
            addressing_sink = {}
            dialogue_turns = detect_dialogue_turns(source_text)
            chunk_data = chunk.get('chunk_data') or {}
            scene_key = chunk_data.get('chapter_index')
            if scene_key is None:
                scene_key = (chunk_data.get('dialogue_attribution') or {}).get('scene_key')
            normalized_scene_key = (
                str(scene_key) if scene_key is not None else None
            )
            if (
                normalized_scene_key is not None
                and current_dialogue_scene_key is not None
                and normalized_scene_key != current_dialogue_scene_key
            ):
                current_dialogue_state = {}
            if normalized_scene_key is not None:
                current_dialogue_scene_key = normalized_scene_key
            global_lore, current_dynamic_text, change_logs = await update_novel_context_chunk(
                llm_client=llm_client,
                model_name=model_name,
                current_global_lore=global_lore,
                current_dynamic_state=current_dynamic_text,
                source_chunk=source_text,
                translated_chunk=chunk.get('translated_text'),
                source_language=config.get('source_language'),
                target_language=config.get('target_language'),
                chunk_index=idx + 1,
                total_chunks=len(chunks),
                source_context=_bounded_source_memory(source_memory_chunks),
                dialogue_turns=dialogue_turns,
                current_dialogue_state=current_dialogue_state,
                dialogue_attribution_sink=dialogue_sink,
                relationship_candidate_sink=relationship_sink,
                addressing_candidate_sink=addressing_sink,
                selective_context_view=(
                    (config.get('prompt_options') or {}).get(
                        "novel_context_selective_update",
                        True,
                    )
                ),
                context_view_max_tokens=(
                    (config.get('prompt_options') or {}).get(
                        "novel_context_update_prompt_max_tokens",
                    )
                ),
                custom_instructions=str((config.get('prompt_options') or {}).get("custom_instructions") or ""),
                glossary_block=str((config.get('prompt_options') or {}).get("glossary_block") or ""),
            )
            if source_text.strip():
                source_memory_chunks.append(source_text)
                bounded_source_memory = _bounded_source_memory(source_memory_chunks)
                source_memory_chunks = (
                    bounded_source_memory.split("\n\n--- Previous source chunk ---\n\n")
                    if bounded_source_memory
                    else []
                )
            if dialogue_turns or dialogue_sink.get('state_after'):
                current_dialogue_state = dict(
                    canonicalize_dialogue_state(
                        dialogue_sink.get('state_after') or {},
                        character_alias_map(global_lore),
                    )
                )
            else:
                dialogue_sink = empty_dialogue_attribution(
                    current_dialogue_state
                )
            if normalized_scene_key is not None:
                dialogue_sink['scene_key'] = normalized_scene_key
            if dialogue_turns:
                dialogue_stats = dialogue_attribution_stats(dialogue_sink)
                append_and_emit(
                    "Dialogue context: "
                    f"{dialogue_stats['identified']} turns identified, "
                    f"{dialogue_stats['assigned']} assigned, "
                    f"{dialogue_stats['uncertain']} uncertain."
                )
            
            new_full_context = build_novel_context(global_lore, current_dynamic_text)
            new_compressed = compress_dynamic_state(new_full_context)
            
            if new_compressed:
                # Log any changes
                for change_log in change_logs:
                    append_and_emit(change_log)
                
                # Save to DB
                latest_cp = state_manager.checkpoint_manager.load_checkpoint(translation_id)
                if not latest_cp:
                    append_and_emit("❌ Translation checkpoint disappeared during resync.")
                    return False
                
                snapshot_saved = False
                for c in latest_cp['chunks']:
                    if c.get('chunk_index') == idx:
                        if c.get('chunk_data') is None:
                            c['chunk_data'] = {}
                        c['chunk_data']['context_snapshot'] = new_compressed
                        c['chunk_data']['dialogue_attribution'] = dialogue_sink
                        staged_relationships = relationship_sink.get("candidates") or []
                        if (
                            relationship_mode != "off"
                            and (c.get('status') or 'completed') == 'completed'
                        ):
                            locked_facts = [
                                edge for edge in db.get_relationship_edges(
                                    translation_id,
                                    statuses=["accepted"],
                                )
                                if edge.get("is_locked")
                            ]
                            judged = await judge_ambiguous_relationship_candidates(
                                llm_client=llm_client,
                                candidates=staged_relationships,
                                source_text=source_text,
                                model_name=model_name,
                                enabled=bool(
                                    (config.get('prompt_options') or {}).get(
                                        "use_relationship_llm_judge",
                                        False,
                                    )
                                ),
                                locked_facts=locked_facts,
                            )
                            staged_relationships = [item.to_dict() for item in judged]
                        db.stage_context_resync_chunk(
                            run_id,
                            translation_id,
                            {
                                "chunk_index": idx,
                                "original_text": c.get('original_text'),
                                "translated_text": c.get('translated_text'),
                                "chunk_data": c.get('chunk_data'),
                                "status": c.get('status') or 'completed',
                            },
                            relationship_candidates=staged_relationships,
                            addressing_candidates=addressing_sink.get("candidates") or [],
                            parser_status=relationship_sink.get("parse_status", "absent"),
                        )
                        snapshot_saved = True
                        _save_resync_state({
                            "status": "running",
                            "last_processed_chunk": idx,
                        })
                        break
                if not snapshot_saved:
                    append_and_emit(f"❌ Chunk {idx} disappeared during resync.")
                    return False
                
                if _pause_requested():
                    paused_state = _save_resync_state({
                        "status": "paused",
                        "pause_requested": False,
                        "last_processed_chunk": idx,
                    })
                    append_and_emit(
                        "⏸️ Context resync paused. Resume it after changing model settings.",
                        paused_state,
                    )
                    db.finish_context_resync_run(
                        run_id,
                        "paused",
                        final_context=new_full_context,
                    )
                    return False
                        
        except Exception as e:
            err_msg = f"Resync failed at chunk {idx}: {e}"
            logger.error(err_msg)
            _save_resync_state({
                "status": "failed",
                "pause_requested": False,
                "last_processed_chunk": max(start_chunk_index, idx - 1),
                "error": str(e),
            })
            append_and_emit(f"❌ {err_msg}")
            db.finish_context_resync_run(
                run_id,
                "failed",
                final_context=new_full_context,
                error=str(e),
            )
            return False
            
    msg_end = "Background context resync completed."
    activate_staged_timeline(new_full_context)
    if path:
        save_novel_context(path.name, path.parent, new_full_context)
    logger.info(msg_end)
    completed_state = _save_resync_state({
        "status": "completed",
        "pause_requested": False,
        "last_processed_chunk": (
            chunks_to_process[-1]['chunk_index']
            if chunks_to_process
            else start_chunk_index
        ),
    })
    append_and_emit(f"✅ {msg_end}", completed_state)
    
    if post_resync_callback or auto_resume_callback:
        logger.info("Triggering follow-up callback after resync")
        run_follow_up()
    return True
