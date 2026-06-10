"""
Generic translator orchestrator using the adapter pattern.

This module provides a unified translation workflow that works with any file format
through the FormatAdapter interface.
"""

from typing import Callable, Optional, Dict, Any
from pathlib import Path

from .format_adapter import FormatAdapter
from .translation_unit import TranslationUnit


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
                        'model_name': model_name,
                        'llm_provider': llm_provider,
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
            from src.config import resolve_parallel_workers
            from src.core.common.parallel import iter_ordered_concurrent
            from src.core.llm.exceptions import RateLimitError

            workers = resolve_parallel_workers(llm_provider, parallel_workers)
            sequential = workers == 1
            prompt_options = llm_kwargs.get('prompt_options', {})

            last_context = ""
            failed_count = 0
            completed_count = len(restored_completed)

            async def _translate_unit(i):
                """Translate one unit. Reads last_context only in sequential mode
                (parallel runs have no stable 'previous translation')."""
                unit = units[i]
                if log_callback:
                    log_callback("unit_start",
                        f"Translating unit {i+1}/{total_units} ({unit.unit_id})")
                return await generate_translation_request(
                    main_content=unit.content,
                    context_before=unit.context_before,
                    context_after=unit.context_after,
                    previous_translation_context=(last_context if sequential else ""),
                    source_language=source_language,
                    target_language=target_language,
                    model=model_name,
                    llm_client=llm_client,
                    log_callback=log_callback,
                    prompt_options=prompt_options
                )

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
