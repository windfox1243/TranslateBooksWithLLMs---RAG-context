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
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.translation_id)

            if checkpoint_data:
                await self.adapter.resume_from_checkpoint(checkpoint_data)
                # Pending work is derived from per-chunk statuses, not from the
                # progress pointer: the pointer advances past failed units, so
                # resuming from it alone would skip them forever (issue #204).
                restored_completed = {
                    c['chunk_index'] for c in checkpoint_data.get('chunks', [])
                    if c.get('status') == 'completed'
                    and 0 <= c.get('chunk_index', -1) < total_units
                }
                if log_callback:
                    log_callback("checkpoint_resumed",
                        f"Resuming: {len(restored_completed)}/{total_units} units already translated")
                # Update stats with resumed progress
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': len(restored_completed),
                        'failed_chunks': 0
                    })
            else:
                # 4. Create new translation job
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
            from src.core.llm_client import LLMClient
            from src.core.translator import generate_translation_request

            llm_client = LLMClient(
                provider_type=llm_provider,
                model=model_name,
                **llm_kwargs
            )

            # 6. Translate each unit (sequentially, or with continuous concurrency)
            from src.config import resolve_parallel_workers, UNIT_VALIDATION_RETRIES
            from src.core.common.parallel import iter_ordered_concurrent
            from src.core.llm.exceptions import RateLimitError

            prompt_options = llm_kwargs.get('prompt_options', {})
            chapter_mode = bool(
                prompt_options.get("chapter_mode")
                and self.adapter.format_name == "txt"
            )
            novel_context_file = prompt_options.get('novel_context_file')
            auto_update_context = prompt_options.get('auto_update_context', False)

            from src.config import NOVEL_CONTEXTS_DIR
            from src.utils.novel_context import open_novel_context_session

            resume_snapshot = None
            resume_snapshot_index = None
            if restored_completed and checkpoint_data:
                resume_snapshot_index = max(restored_completed)
                for checkpoint_chunk in checkpoint_data.get('chunks', []):
                    if checkpoint_chunk.get('chunk_index') == resume_snapshot_index:
                        resume_snapshot = (
                            checkpoint_chunk.get('chunk_data') or {}
                        ).get('context_snapshot')
                        break

            try:
                context_session = open_novel_context_session(
                    prompt_options=prompt_options,
                    novel_contexts_dir=NOVEL_CONTEXTS_DIR,
                    input_filename=str(getattr(self.adapter, 'input_file_path', '') or ''),
                    fallback_name="text",
                    resume_snapshot=resume_snapshot,
                    log_callback=log_callback,
                )
                if resume_snapshot and context_session and log_callback:
                    log_callback(
                        "novel_context_resume",
                        f"Restored context from chunk {resume_snapshot_index} snapshot.",
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
            failed_count = 0
            completed_count = len(restored_completed)

            async def _translate_unit(i):
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

                base_instructions = (prompt_options or {}).get('custom_instructions', '')
                attempt_options = prompt_options
                result = None

                if auto_update_context and context_session:
                    if log_callback:
                        log_callback(
                            "novel_context_updating",
                            f"Analyzing source context for unit {i+1} before translation...",
                        )
                    try:
                        change_logs = await context_session.analyze_source(
                            llm_client=llm_client,
                            model_name=model_name,
                            source_chunk=unit.content,
                            source_language=source_language,
                            target_language=target_language,
                            chunk_index=i + 1,
                            total_chunks=total_units,
                        )
                        if log_callback:
                            log_callback(
                                "novel_context_updated",
                                f"Novel context prepared for unit {i+1}.",
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

                for attempt in range(max_validation_attempts):
                    same_previous_chapter = (
                        i > 0
                        and units[i - 1].metadata.get("chapter_index")
                        == unit.metadata.get("chapter_index")
                    )
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
                        prompt_options=attempt_options
                    )

                    # API failure / empty result: existing failure semantics.
                    if not result:
                        return result

                    feedback = self.adapter.validate_unit_translation(
                        unit.unit_id, result
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

            def _record_failure(i, unit):
                nonlocal failed_count
                if log_callback:
                    log_callback("unit_failed", f"Failed to translate unit {i+1}/{total_units}")
                failed_count += 1
                self.checkpoint_manager.save_checkpoint(
                    translation_id=self.translation_id,
                    chunk_index=i,
                    original_text=unit.content,
                    translated_text=None,
                    chunk_data=unit.metadata,
                    total_chunks=total_units,
                    failed_chunks=failed_count
                )
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': completed_count,
                        'failed_chunks': failed_count
                    })

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
                    _record_failure(i, unit)
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
                    _record_failure(i, unit)
                    next_index = i + 1
                    continue

                translated_content = result
                if translated_content:
                    save_success = await self.adapter.save_unit_translation(
                        unit.unit_id, translated_content
                    )
                    if not save_success:
                        if log_callback:
                            log_callback("save_failed",
                                f"Failed to save translation for unit {unit.unit_id}")
                        failed_count += 1
                        next_index = i + 1
                        continue

                    completed_count += 1
                    
                    # Save the full context snapshot in the chunk checkpoint
                    if context_session:
                        if unit.metadata is None:
                            unit.metadata = {}
                        unit.metadata['context_snapshot'] = context_session.snapshot()

                    self.checkpoint_manager.save_checkpoint(
                        translation_id=self.translation_id,
                        chunk_index=i,
                        original_text=unit.content,
                        translated_text=translated_content,
                        chunk_data=unit.metadata,
                        total_chunks=total_units,
                        completed_chunks=completed_count
                    )
                    if stats_callback:
                        stats_callback({
                            'total_chunks': total_units,
                            'completed_chunks': completed_count,
                            'failed_chunks': failed_count
                        })

                    if sequential:
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

def resync_context_snapshots_background(translation_id: str, start_chunk_index: int, initial_compressed_snapshot: str, socketio=None, was_active=False, auto_resume_callback=None):
    """Entry point for the background thread."""
    import asyncio
    from src.config import NOVEL_CONTEXTS_DIR
    
    # Try to use existing loop if we are in one, otherwise run new
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_resync_context_snapshots_async(translation_id, start_chunk_index, initial_compressed_snapshot, socketio, was_active, auto_resume_callback), loop)
            return
    except RuntimeError:
        pass
        
    asyncio.run(_resync_context_snapshots_async(translation_id, start_chunk_index, initial_compressed_snapshot, socketio, was_active, auto_resume_callback))

async def _resync_context_snapshots_async(translation_id: str, start_chunk_index: int, initial_compressed_snapshot: str, socketio=None, was_active=False, auto_resume_callback=None):
    """Forward-pass through chunks to re-evaluate the context using the LLM."""
    from src.api.translation_state import get_state_manager
    from src.utils.novel_context import (
        build_novel_context,
        compress_dynamic_state,
        decode_context_snapshot,
        load_novel_context,
        normalize_novel_context_filename,
        resolve_novel_context_path,
        save_novel_context,
        update_novel_context_chunk,
    )
    from src.core.llm_client import LLMClient
    from src.config import NOVEL_CONTEXTS_DIR
    from datetime import datetime
    from src.utils.unified_logger import get_logger
    from src.api.websocket import emit_update
    
    state_manager = get_state_manager()
    logger = get_logger("context_resync")
    
    def append_and_emit(msg_str):
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg_str}"
        state_manager.append_log(translation_id, log_entry)
        if socketio:
            structured_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "general",
                "message": msg_str,
                "data": {
                    "ui_step": "context_resync",
                    "phase": "context",
                },
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
    
    msg = f"Starting background context resync for {translation_id} from chunk {start_chunk_index}"
    logger.info(msg)
    append_and_emit(f"🔄 {msg}")
    
    if was_active:
        msg_pause = "Waiting for active translation to pause before resyncing..."
        logger.info(msg_pause)
        append_and_emit(f"⏸️ {msg_pause}")
        
        import asyncio
        paused = False
        for _ in range(60):  # Wait up to 60 seconds
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
            err_msg = "Active translation did not pause within 60 seconds; resync aborted to avoid racing with translation."
            logger.error(err_msg)
            append_and_emit(f"❌ {err_msg}")
            return False
    
    checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint_data:
        append_and_emit("❌ Translation checkpoint is no longer available.")
        return False
        
    config = checkpoint_data.get('job', {}).get('config', {})
    chunks = checkpoint_data.get('chunks', [])
    
    # We only re-sync completed chunks that are ahead of start_chunk_index
    completed_chunks = [c for c in chunks if c.get('status') in ('completed', 'partial') and c.get('chunk_index') is not None]
    completed_chunks.sort(key=lambda x: x['chunk_index'])
    
    chunks_to_process = [c for c in completed_chunks if c['chunk_index'] > start_chunk_index]
    
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

    if not chunks_to_process:
        latest_completed = [c['chunk_index'] for c in completed_chunks]
        if not latest_completed or start_chunk_index != max(latest_completed):
            append_and_emit("❌ The selected context snapshot is not a completed chunk.")
            return False
        if path:
            try:
                save_novel_context(path.name, path.parent, current_full_context)
                append_and_emit(f"✅ Context file '{path.name}' updated successfully.")
            except Exception as e:
                err_msg = f"Failed to update context file '{path.name}': {e}"
                logger.error(err_msg)
                append_and_emit(f"❌ {err_msg}")
                return False
        append_and_emit("✅ Background context resync completed.")
        if auto_resume_callback:
            append_and_emit("▶️ Auto-resuming active translation...")
            auto_resume_callback()
        return True

    llm_provider = config.get('llm_provider', 'ollama')
    model_name = config.get('model') or config.get('model_name')
    provider_key = config.get(f'{llm_provider}_api_key')
    if not provider_key:
        import os
        import src.config as live_config

        env_var = f"{llm_provider.upper()}_API_KEY"
        provider_key = (
            os.getenv(env_var)
            or getattr(live_config, env_var, '')
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
    
    for chunk in chunks_to_process:
        idx = chunk['chunk_index']
        source_text = chunk.get('original_text') or ''
        
        try:
            msg_resync = f"Resyncing chunk {idx}..."
            logger.info(msg_resync)
            append_and_emit(f"🔄 {msg_resync}")
            global_lore, current_dynamic_text, change_logs = await update_novel_context_chunk(
                llm_client=llm_client,
                model_name=model_name,
                current_global_lore=global_lore,
                current_dynamic_state=current_dynamic_text,
                source_chunk=source_text,
                translated_chunk=None,
                source_language=config.get('source_language'),
                target_language=config.get('target_language'),
                chunk_index=idx + 1,
                total_chunks=len(chunks),
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
                        state_manager.checkpoint_manager.db.save_chunk(
                            translation_id=translation_id,
                            chunk_index=idx,
                            original_text=c.get('original_text'),
                            translated_text=c.get('translated_text'),
                            chunk_data=c.get('chunk_data'),
                            status=c.get('status') or 'completed'
                        )
                        snapshot_saved = True
                        break
                if not snapshot_saved:
                    append_and_emit(f"❌ Chunk {idx} disappeared during resync.")
                    return False
                
                # If this is the last completed chunk, update the file
                latest_completed = [
                    c.get('chunk_index')
                    for c in latest_cp['chunks']
                    if c.get('status') in ('completed', 'partial')
                ]
                if path and latest_completed and idx == max(latest_completed):
                    try:
                        save_novel_context(path.name, path.parent, new_full_context)
                    except Exception as e:
                        err_msg = f"Failed to update context file: {e}"
                        logger.error(err_msg)
                        append_and_emit(f"❌ {err_msg}")
                        return False
                        
        except Exception as e:
            err_msg = f"Resync failed at chunk {idx}: {e}"
            logger.error(err_msg)
            append_and_emit(f"❌ {err_msg}")
            return False
            
    msg_end = "Background context resync completed."
    logger.info(msg_end)
    append_and_emit(f"✅ {msg_end}")
    
    if auto_resume_callback:
        logger.info("Triggering auto-resume callback after resync")
        append_and_emit(f"▶️ Auto-resuming active translation...")
        auto_resume_callback()
    return True
