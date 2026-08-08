"""The callback surface a translation job is wired with before it runs.

perform_actual_translation used to open with ~320 lines that built the job's
logger, its interruption probe, its progress emitter and its five stats
callbacks. None of that is translation logic -- it is the seam between the
engine's plain callbacks and the web layer's sockets, state manager and
checkpoint store -- so it lives here instead.

build_job_callbacks returns the nine callables the job body actually uses;
everything else in this module stays captured in their closures.
"""
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.core.progress import snapshot_from_legacy_stats
from src.utils.unified_logger import LogType, setup_web_logger

from .websocket import emit_update


@dataclass(frozen=True)
class JobCallbacks:
    """The callables perform_actual_translation hands to the engine.

    Frozen: the job body only ever reads these, and the engine keeps
    references to them for the lifetime of the run.
    """

    logger: Any
    should_interrupt_current_task: Callable[[], bool]
    log_message_callback: Callable[..., Any]
    phase_boundary_review_repair: Callable[..., Any]
    translate_stats_callback: Callable[..., Any]
    refine_after_stats_callback: Callable[..., Any]
    refine_only_stats_callback: Callable[..., Any]
    finalize_stats_callback: Callable[..., Any]
    openrouter_cost_callback: Callable[..., Any]


def build_job_callbacks(
    translation_id: str,
    config: dict,
    state_manager: Any,
    output_dir: str,
    socketio: Optional[Any],
    checkpoint_manager: Any,
) -> JobCallbacks:
    """Wire one job's callbacks against the web layer and return them."""
    def should_interrupt_current_task():
        if state_manager.exists(translation_id) and state_manager.get_translation_field(translation_id, 'interrupted'):
            _log_message_callback("interruption_check", f"Interruption signal detected for job {translation_id}. Halting processing.")
            return True
        return False

    # Setup unified logger for web interface
    def web_callback(log_entry):
        """Callback for WebSocket emission"""
        # Send full log entry for structured processing on client side
        emit_update(socketio, translation_id, {'log': log_entry['message'], 'log_entry': log_entry}, state_manager)
    
    def storage_callback(log_entry):
        """Callback for storing logs.

        Uses the atomic append so concurrent workers can't drop each other's
        entries the way a get-mutate-set round trip does.
        """
        state_manager.append_log(translation_id, log_entry)
    
    logger = setup_web_logger(web_callback, storage_callback)

    def _ui_step_metadata(message_key):
        """Mark user-visible workflow milestones for the activity log."""
        key = (message_key or "").lower()
        if key.endswith(("_request", "_response")):
            return None
        visible = (
            key.startswith((
                "novel_context",
                "refine",
                "refinement",
                "epub_refine",
                "docx_refine",
                "srt_refine",
            ))
            or key in {
                "prepare_start",
                "units_found",
                "unit_start",
                "unit_complete",
                "reconstruct_start",
                "reconstruct_complete",
                "translation_complete",
                "translation_interrupted",
            }
            or key.endswith((
                "_start",
                "_complete",
                "_done",
                "_updated",
                "_refined",
                "_fallback",
                "_failed",
                "_error",
                "_interrupted",
            ))
        )
        if not visible:
            return None
        if "context" in key:
            phase = "context"
        elif "refine" in key or "refinement" in key:
            phase = "refinement"
        else:
            phase = "translation"
        return {"ui_step": message_key, "phase": phase}
    
    def _log_message_callback(message_key_from_translate_module, message_content="", data=None):
        """Legacy callback wrapper for backward compatibility"""
        # Skip debug messages for web interface
        if message_key_from_translate_module in ["llm_prompt_debug", "llm_raw_response_preview"]:
            return
        
        step_metadata = _ui_step_metadata(message_key_from_translate_module)
        structured_data = dict(data) if isinstance(data, dict) else {}
        if step_metadata:
            structured_data.update(step_metadata)

        # Handle structured data from new logging system
        if structured_data:
            log_type = structured_data.get('type')
            if log_type == 'llm_request':
                logger.debug("LLM Request", LogType.LLM_REQUEST, structured_data)
            elif log_type == 'llm_response':
                # Use INFO level to ensure translation preview works even when DEBUG_MODE=false
                logger.info("LLM Response", LogType.LLM_RESPONSE, structured_data)
            elif log_type == 'refinement_request':
                # Refinement uses same log type as LLM request for UI display
                logger.debug("Refinement Request", LogType.REFINEMENT_REQUEST, structured_data)
            elif log_type == 'refinement_response':
                # Refinement uses same log type as LLM response for UI display
                # Use INFO level to ensure translation preview works even when DEBUG_MODE=false
                logger.info("Refinement Response", LogType.REFINEMENT_RESPONSE, structured_data)
            elif log_type == 'progress':
                logger.info("Progress Update", LogType.PROGRESS, structured_data)
            elif log_type == 'novel_context_state':
                logger.info(message_content or "Context Updated", LogType.NOVEL_CONTEXT_STATE, structured_data)
            else:
                logger.info(message_content, data=structured_data)
        else:
            # Map specific message patterns to appropriate log types
            if "error" in message_key_from_translate_module.lower():
                logger.error(message_content)
            elif "warning" in message_key_from_translate_module.lower():
                logger.warning(message_content)
            else:
                logger.info(message_content)

    def _auto_review_event(event, batch):
        """Emit localized-client metadata plus a concise terminal milestone."""
        payload = {
            "event": event,
            "batch_id": batch.get("batch_id"),
            "translation_id": translation_id,
            "scope": batch.get("scope"),
            "phase": batch.get("phase"),
            "status": batch.get("status"),
            "stay_paused": False,
            "total_items": int(batch.get("total_items") or 0),
            "completed_items": int(batch.get("completed_items") or 0),
            "succeeded_items": int(batch.get("succeeded_items") or 0),
            "failed_items": int(batch.get("failed_items") or 0),
        }
        for key in (
            "chunk_index", "position", "boundary", "output_status",
        ):
            if key in batch:
                payload[key] = batch[key]
        print(
            f"[{time.strftime('%H:%M:%S')}] [Automatic review repair] "
            f"{event} phase={payload['phase']} "
            f"progress={payload['completed_items']}/{payload['total_items']}",
            flush=True,
        )
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "type": "general",
            "message": f"Automatic review repair: {event}",
            "data": {
                "ui_step": "editor_repair_batch",
                "editor_repair_batch": payload,
            },
        }
        logs = list(
            state_manager.get_translation_field(translation_id, 'logs', [])
            or []
        )
        logs.append(entry)
        state_manager.set_translation_field(translation_id, 'logs', logs[-1000:])
        emit_update(
            socketio,
            translation_id,
            {"editor_repair_batch": payload},
            state_manager,
        )

    try:
        auto_review_threshold = int(
            (config.get('prompt_options') or {}).get(
                'auto_review_repair_threshold', 3,
            )
        )
    except (TypeError, ValueError):
        auto_review_threshold = 3
    auto_review_threshold = max(0, min(auto_review_threshold, 20))
    auto_review_coordinator = None
    if (config.get('prompt_options') or {}).get('reflection_mode'):
        from src.core.editor.auto_review_repair import AutoReviewRepairCoordinator

        auto_review_coordinator = AutoReviewRepairCoordinator(
            translation_id=translation_id,
            checkpoint_manager=checkpoint_manager,
            output_dir=Path(output_dir),
            threshold=auto_review_threshold,
            event_callback=_auto_review_event,
        )

    def _threshold_review_repair(phase):
        if auto_review_coordinator is None:
            return
        try:
            auto_review_coordinator.repair_if_needed_blocking(phase)
        except Exception as exc:
            print(
                f"[{time.strftime('%H:%M:%S')}] [Automatic review repair] "
                f"threshold check failed: {type(exc).__name__}",
                flush=True,
            )

    async def _phase_boundary_review_repair(phase):
        if auto_review_coordinator is None:
            return {"status": "disabled"}
        result = await auto_review_coordinator.repair_if_needed(
            phase, boundary=True,
        )
        if result.get("status") == "empty":
            _auto_review_event("boundary_clear", {
                "batch_id": f"boundary_{phase}_{int(time.time())}",
                "scope": "auto_review_boundary",
                "phase": phase,
                "status": "completed",
                "total_items": 0,
                "completed_items": 0,
                "succeeded_items": 0,
                "failed_items": 0,
                "boundary": True,
            })
        diagnostics = checkpoint_manager.db.get_editor_diagnostics(
            translation_id
        )
        remaining = len(diagnostics.get("current_review_queue") or [])
        current_stats = dict(
            state_manager.get_translation_field(translation_id, 'stats', {})
            or {}
        )
        current_stats['review_required_chunks'] = remaining
        state_manager.set_translation_field(
            translation_id, 'stats', current_stats,
        )
        return result

    # Single progress-emit seam. The translate_file → refine_file orchestration
    # still runs two independent engine-side trackers, but the *workflow phase*
    # is now owned here and passed explicitly per phase (TRANSLATING vs
    # REFINING) by whichever call is driving the emit — replacing the old
    # mutable `_workflow_meta` side-channel. A monotonic floor on the canonical
    # `percent` guarantees the bar never regresses across the phase boundary,
    # which is what previously required a manual counter reset at the
    # transition.
    _progress_floor = {'value': 0.0}

    def _context_chunk_indices():
        prompt_options = config.get('prompt_options') or {}
        if not (
            prompt_options.get('novel_context_file')
            or prompt_options.get('auto_update_context')
        ):
            return []
        try:
            rows = state_manager.get_checkpoint_manager().db.get_chunks(
                translation_id
            )
        except Exception:
            return []
        return sorted({
            row['chunk_index']
            for row in rows
            if (
                isinstance(row.get('chunk_index'), int)
                and row.get('status') in ('completed', 'partial', 'failed')
                and (row.get('chunk_data') or {}).get('context_snapshot')
            )
        })

    def _emit_progress(new_stats_dict, phase_meta):
        if not state_manager.exists(translation_id):
            return
        state_manager.update_stats(translation_id, {**new_stats_dict, **phase_meta})
        current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        current_stats['elapsed_time'] = time.time() - current_stats.get('start_time', time.time())
        # Attach the canonical progress contract alongside the legacy fields,
        # applying the monotonic floor so the global bar never moves backward.
        snapshot = snapshot_from_legacy_stats(current_stats).to_dict()
        if snapshot['percent'] < _progress_floor['value']:
            snapshot['percent'] = _progress_floor['value']
        else:
            _progress_floor['value'] = snapshot['percent']
        current_stats.update(snapshot)
        current_stats['context_chunk_indices'] = _context_chunk_indices()
        state_manager.set_translation_field(translation_id, 'stats', current_stats)
        emit_update(socketio, translation_id, {'stats': current_stats}, state_manager)

        # Update logger progress for CLI display
        completed = current_stats.get('completed_chunks', 0)
        total = current_stats.get('total_chunks', 0)
        if total > 0:
            logger.update_progress(completed, total)

    def _translate_stats_callback(new_stats_dict):
        # Phase 1. enable_refinement advertises the two-phase bar up-front when
        # a refine-after pass will follow, so phase 1 maps to the [0, 50] band.
        _emit_progress(new_stats_dict, {
            'enable_refinement': bool(config.get('refine_after')),
            'current_phase': 1,
        })
        _threshold_review_repair('translation')

    def _refine_after_stats_callback(new_stats_dict):
        # Phase 2 of a refine-after workflow: maps to the [50, 100] band.
        _emit_progress(new_stats_dict, {'enable_refinement': True, 'current_phase': 2})
        _threshold_review_repair('refinement')

    def _refine_only_stats_callback(new_stats_dict):
        # Single-phase refine-only: the whole bar is the refinement pass.
        _emit_progress(new_stats_dict, {
            'enable_refinement': False, 'refine_only': True, 'current_phase': 1,
        })
        _threshold_review_repair('refinement')

    def _finalize_stats_callback(new_stats_dict):
        # Finalization pushes (e.g. final elapsed_time) must not re-assert a
        # phase; the stored stats already carry the terminal phase/percent.
        _emit_progress(new_stats_dict, {})

    def _openrouter_cost_callback(cost_data):
        """Update OpenRouter cost in state. No emit: this callback runs on the
        provider's HTTP response thread, and a cross-thread emit can overtake
        the main loop's stats emit on the wire (showing a stale snapshot and
        rolling the progress bar backward). The cost is picked up by the next
        chunk's stats_callback, which is the same thread that owns progress."""
        if state_manager.exists(translation_id):
            state_manager.update_stats(translation_id, {
                'openrouter_cost': cost_data['session_cost'],
                'openrouter_prompt_tokens': cost_data['total_prompt_tokens'],
                'openrouter_completion_tokens': cost_data['total_completion_tokens']
            })
    return JobCallbacks(
        logger=logger,
        should_interrupt_current_task=should_interrupt_current_task,
        log_message_callback=_log_message_callback,
        phase_boundary_review_repair=_phase_boundary_review_repair,
        translate_stats_callback=_translate_stats_callback,
        refine_after_stats_callback=_refine_after_stats_callback,
        refine_only_stats_callback=_refine_only_stats_callback,
        finalize_stats_callback=_finalize_stats_callback,
        openrouter_cost_callback=_openrouter_cost_callback,
    )
