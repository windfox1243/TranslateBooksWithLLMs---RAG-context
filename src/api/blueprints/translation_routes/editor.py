"""Routes for editor retries, repair batches and editor diagnostics."""
import asyncio
import threading
import time
import uuid
from pathlib import Path

from flask import jsonify, request

from src.api.websocket import emit_update

from .helpers import (
    _active_translation_conflict,
    _claim_editor_batch,
    _claim_editor_retry,
    _release_editor_batch,
    _release_editor_retry,
    logger,
)


def register(bp, deps, shared):
    """Register the editor routes on bp."""
    state_manager = deps.state_manager
    output_dir = deps.output_dir
    socketio = deps.socketio
    _editor_batch_activity = shared.editor_batch_activity
    make_editor_retry_auto_resume_callback = shared.make_editor_retry_auto_resume_callback

    @bp.route('/api/translation/<translation_id>/editor-diagnostics', methods=['GET'])
    @bp.route('/<translation_id>/editor-diagnostics', methods=['GET'])
    def get_editor_diagnostics_route(translation_id):
        """Return locally persisted Senior Editor outcomes for one job."""
        job = state_manager.checkpoint_manager.get_job(translation_id)
        if not job:
            return jsonify({"error": "Translation job not found"}), 404
        result = state_manager.checkpoint_manager.db.get_editor_diagnostics(
            translation_id
        )
        from src.core.editor_retry import editor_retry_state

        checkpoint = state_manager.checkpoint_manager.load_checkpoint(
            translation_id
        ) or {}
        retry_states = {
            str(chunk.get("chunk_index")): editor_retry_state(
                chunk.get("chunk_data")
            )
            for chunk in checkpoint.get("chunks", [])
        }
        chunks_by_index = {
            int(chunk.get("chunk_index", -1)): chunk
            for chunk in checkpoint.get("chunks", [])
        }
        for overlay in state_manager.checkpoint_manager.db.get_active_refinement_results(
            translation_id
        ):
            if overlay.get("base_chunk_index") is None:
                continue
            retry_states[str(int(overlay["base_chunk_index"]))] = editor_retry_state(
                overlay.get("chunk_data")
            )
        for item in result.get("current_review_queue") or []:
            chunk = chunks_by_index.get(int(item.get("chunk_index", -1))) or {}
            validation = (chunk.get("chunk_data") or {}).get("editor_validation") or {}
            actionable = []
            for issue in list(validation.get("unresolved_issues") or [])[:5]:
                if not isinstance(issue, dict):
                    continue
                replacement = issue.get("draft_replacement") or {}
                actionable.append({
                    "issue_id": str(issue.get("issue_id") or "")[:80],
                    "category": str(issue.get("category") or "unknown")[:80],
                    "problem_text": str(replacement.get("draft") or "")[:160],
                    "expected_text": str(replacement.get("replacement") or "")[:160],
                })
            item["issues"] = actionable
            reason_codes = item.get("reason_codes") or []
            first_reason = str(reason_codes[0] if reason_codes else "review_required")
            if "source_residue" in first_reason:
                item["reason_key"] = "source_residue"
            elif "narrator" in first_reason:
                item["reason_key"] = "narrator_mismatch"
            elif "placeholder" in first_reason or "newline" in first_reason:
                item["reason_key"] = "structure_mismatch"
            elif "locator" in first_reason or "local_patch" in first_reason:
                item["reason_key"] = "replacement_not_found"
            elif item.get("outcome") == "transport_failed":
                item["reason_key"] = "transport_failed"
            else:
                item["reason_key"] = "review_required"
        result["retry_states"] = retry_states
        db = state_manager.checkpoint_manager.db
        result["active_repair_batch"] = db.find_active_editor_repair_batch(
            translation_id
        )
        get_latest_batch = getattr(db, "get_latest_editor_repair_batch", None)
        result["latest_repair_batch"] = (
            get_latest_batch(translation_id)
            if callable(get_latest_batch) else result["active_repair_batch"]
        )
        for run in result.get("runs") or []:
            run["retry_state"] = retry_states.get(
                str(run.get("chunk_index")), {"status": "idle"}
            )
        return jsonify(result)
    @bp.route(
        '/api/translation/<translation_id>/chunks/<int:chunk_index>/retry-editor',
        methods=['POST'],
    )
    @bp.route(
        '/<translation_id>/chunks/<int:chunk_index>/retry-editor',
        methods=['POST'],
    )
    def retry_editor_route(translation_id, chunk_index):
        """Queue an editor-only retry, pausing the same active job safely."""
        request_data = request.get_json(silent=True) or {}
        requested_phase = str(request_data.get("phase") or "effective").casefold()
        if requested_phase not in {"effective", "translation", "refinement"}:
            return jsonify({"error": "Unsupported repair phase"}), 400
        live_job = state_manager.get_translation(translation_id) or {}
        live_status = live_job.get("status")
        pause_active_job = live_status == "running"
        active_error = _active_translation_conflict(
            state_manager,
            action="retry the Senior Editor",
            ignore_translation_id=(
                translation_id if pause_active_job else None
            ),
        )
        if active_error is not None:
            return active_error
        diagnostics = state_manager.checkpoint_manager.db.get_editor_diagnostics(
            translation_id
        )
        latest_run = next((
            run for run in reversed(diagnostics.get("runs") or [])
            if int(run.get("chunk_index", -1)) == chunk_index
        ), None)
        if not latest_run or latest_run.get("outcome") not in {
            "review_required", "transport_failed",
        }:
            return jsonify({
                "error": "Only review-required or transport-failed editor runs can be retried"
            }), 409
        checkpoint = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint:
            return jsonify({"error": "Translation job not found"}), 404
        chunk = next((
            item for item in checkpoint.get("chunks", [])
            if int(item.get("chunk_index", -1)) == chunk_index
        ), None)
        if not chunk or not str(chunk.get("translated_text") or "").strip():
            return jsonify({"error": "No preserved draft is available"}), 409
        if not _claim_editor_retry(translation_id, chunk_index):
            return jsonify({"error": "A Senior Editor retry is already running"}), 409

        from src.core.editor_retry import _save_retry_state, run_editor_retry

        chunk_data = dict(chunk.get("chunk_data") or {})
        # Keep the previous validation payload intact.  run_editor_retry uses
        # its unresolved_issues as the bounded repair contract; replacing it
        # here silently turns a focused retry into another full-chunk audit.
        queued_state = {
            "status": "queued",
            "requested_at": int(time.time()),
            "source_run_id": latest_run.get("id"),
            "pause_requested": pause_active_job,
            "auto_resume": pause_active_job,
        }
        chunk_data["editor_retry"] = queued_state
        chunk_data["editor_retry_pending"] = True
        saved = state_manager.checkpoint_manager.save_checkpoint(
            translation_id=translation_id,
            chunk_index=chunk_index,
            original_text=chunk.get("original_text") or "",
            translated_text=chunk.get("translated_text"),
            chunk_data=chunk_data,
            chunk_status="editor_retry",
        )
        if not saved:
            _release_editor_retry(translation_id, chunk_index)
            return jsonify({"error": "Could not queue editor retry"}), 500

        if pause_active_job:
            state_manager.set_interrupted(translation_id, True)
            emit_update(
                socketio,
                translation_id,
                {
                    "log": (
                        "Senior Editor retry requested: waiting for the active "
                        "chunk to finish before pausing translation."
                    ),
                },
                state_manager,
            )

        def run_retry_background():
            paused_status = None
            try:
                if pause_active_job:
                    live_config = live_job.get("config") or {}
                    try:
                        pause_timeout = int(
                            live_config.get("request_timeout", 120)
                        ) + 15
                    except (TypeError, ValueError):
                        pause_timeout = 135
                    pause_timeout = max(60, min(pause_timeout, 600))
                    for _ in range(pause_timeout):
                        status_data = (
                            state_manager.get_translation(translation_id) or {}
                        )
                        paused_status = status_data.get("status")
                        if not paused_status:
                            persisted_job = (
                                state_manager.checkpoint_manager.get_job(
                                    translation_id
                                ) or {}
                            )
                            paused_status = persisted_job.get("status")
                        if paused_status in {
                            "paused", "interrupted", "partial", "completed",
                            "failed", "error",
                        }:
                            break
                        time.sleep(1)
                    else:
                        state_manager.set_interrupted(translation_id, False)
                        raise TimeoutError(
                            "Active translation did not pause within "
                            f"{pause_timeout} seconds; Senior Editor retry "
                            "aborted to avoid racing with translation."
                        )

                latest_checkpoint = (
                    state_manager.checkpoint_manager.load_checkpoint(
                        translation_id
                    ) or {}
                )
                latest_chunk = next((
                    item for item in latest_checkpoint.get("chunks", [])
                    if int(item.get("chunk_index", -1)) == int(chunk_index)
                ), chunk)
                _save_retry_state(
                    state_manager.checkpoint_manager,
                    translation_id,
                    latest_chunk,
                    {
                        **queued_state,
                        "status": "running",
                        "started_at": int(time.time()),
                    },
                    chunk_status="editor_retry",
                )
                asyncio.run(run_editor_retry(
                    translation_id=translation_id,
                    chunk_index=chunk_index,
                    checkpoint_manager=state_manager.checkpoint_manager,
                    output_dir=Path(output_dir),
                    phase=requested_phase,
                ))
            except Exception as exc:
                logger.error(
                    "Immediate Senior Editor retry failed for "
                    f"{translation_id} chunk {chunk_index}: {exc}"
                )
                failed_checkpoint = (
                    state_manager.checkpoint_manager.load_checkpoint(
                        translation_id
                    ) or {}
                )
                failed_chunk = next((
                    item for item in failed_checkpoint.get("chunks", [])
                    if int(item.get("chunk_index", -1)) == int(chunk_index)
                ), chunk)
                _save_retry_state(
                    state_manager.checkpoint_manager,
                    translation_id,
                    failed_chunk,
                    {
                        **queued_state,
                        "status": "failed",
                        "message": str(exc),
                        "completed_at": int(time.time()),
                    },
                    chunk_status="completed",
                )
            finally:
                if pause_active_job and paused_status in {
                    "paused", "interrupted", "partial",
                }:
                    make_editor_retry_auto_resume_callback(
                        translation_id
                    )()
                elif pause_active_job and paused_status in {
                    "completed", "failed", "error",
                }:
                    state_manager.set_interrupted(translation_id, False)
                _release_editor_retry(translation_id, chunk_index)

        thread = threading.Thread(
            target=run_retry_background,
            name=f"editor-retry-{translation_id}-{chunk_index}",
            daemon=True,
        )
        thread.start()
        return jsonify({
            "success": True,
            "translation_id": translation_id,
            "chunk_index": chunk_index,
            "status": "queued",
            "retry_state": queued_state,
        }), 202
    @bp.route(
        '/api/translation/<translation_id>/editor-repair-batches',
        methods=['POST'],
    )
    def create_editor_repair_batch_route(translation_id):
        """Pause safely and repair the current narrator/review work queue."""
        db = state_manager.checkpoint_manager.db
        job = state_manager.checkpoint_manager.get_job(translation_id)
        if not job:
            return jsonify({"error": "Translation job not found"}), 404
        data = request.get_json(silent=True) or {}
        scope = str(data.get("scope") or "review_required").casefold()
        phase = str(data.get("phase") or "effective").casefold()
        stay_paused = bool(data.get("stay_paused", True))
        if scope not in {"review_required", "narrator_stale"}:
            return jsonify({"error": "Unsupported repair batch scope"}), 400
        if phase not in {"effective", "translation", "refinement"}:
            return jsonify({"error": "Unsupported repair phase"}), 400
        active = db.find_active_editor_repair_batch(translation_id)
        if active:
            return jsonify({"error": "A repair batch is already active", "batch": active}), 409

        live_job = state_manager.get_translation(translation_id) or {}
        pause_active_job = str(live_job.get("status") or "").casefold() in {
            "running", "refining", "resyncing",
        }
        # A batch launched from an already paused/completed job cannot resume
        # work that it did not pause itself.
        if not pause_active_job:
            stay_paused = True

        if scope == "narrator_stale":
            from src.core.editor_retry import audit_completed_narrator_conformance
            audit = audit_completed_narrator_conformance(
                translation_id=translation_id,
                checkpoint_manager=state_manager.checkpoint_manager,
            )
            chunk_indices = sorted(set(int(value) for value in audit.get("queued") or []))
        else:
            diagnostics = db.get_editor_diagnostics(translation_id)
            chunk_indices = sorted(set(
                int(item["chunk_index"])
                for item in diagnostics.get("current_review_queue") or []
                if item.get("retryable")
                and phase in {"effective", str(item.get("phase") or "translation")}
            ))

        batch_id = f"erb_{uuid.uuid4().hex}"
        if not db.create_editor_repair_batch(
            batch_id, translation_id, scope, phase, chunk_indices,
            stay_paused=stay_paused,
        ):
            return jsonify({"error": "Could not persist repair batch"}), 500
        _editor_batch_activity(
            db, translation_id, batch_id, "queued",
            f"Batch {batch_id} queued with {len(chunk_indices)} item(s).",
        )
        if not chunk_indices:
            db.update_editor_repair_batch(
                batch_id, status="completed", completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            )
            _editor_batch_activity(
                db, translation_id, batch_id, "completed",
                f"Batch {batch_id} completed; no repairable items were found.",
            )
            return jsonify(db.get_editor_repair_batch(batch_id)), 200
        if not _claim_editor_batch(translation_id):
            db.update_editor_repair_batch(
                batch_id, status="blocked", error="editor_retry_conflict",
                completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            )
            return jsonify({"error": "Another editor operation is active"}), 409

        if pause_active_job:
            state_manager.set_interrupted(translation_id, True)
            _editor_batch_activity(
                db, translation_id, batch_id, "pause_requested",
                f"Batch {batch_id} requested a safe translation pause.",
            )

        def run_batch_background():
            completed = succeeded = failed = 0
            try:
                if pause_active_job:
                    db.update_editor_repair_batch(batch_id, status="pausing")
                    _editor_batch_activity(
                        db, translation_id, batch_id, "pausing",
                        f"Batch {batch_id} is waiting for the in-flight unit to finish.",
                    )
                    live_config = live_job.get("config") or {}
                    try:
                        timeout = int(live_config.get("request_timeout", 120)) + 15
                    except (TypeError, ValueError):
                        timeout = 135
                    timeout = max(60, min(timeout, 600))
                    for _ in range(timeout):
                        status = str((state_manager.get_translation(translation_id) or {}).get("status") or "").casefold()
                        if status in {"paused", "interrupted", "partial", "completed", "failed", "error"}:
                            break
                        time.sleep(1)
                    else:
                        raise TimeoutError("Active translation did not pause before the repair batch")
                db.update_editor_repair_batch(
                    batch_id, status="running",
                    started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                _editor_batch_activity(
                    db, translation_id, batch_id, "running",
                    f"Batch {batch_id} started editor repairs.",
                )
                from src.core.editor_retry import _refresh_output, run_editor_retry
                for position, chunk_index in enumerate(chunk_indices, start=1):
                    current = db.get_editor_repair_batch(batch_id) or {}
                    if current.get("cancel_requested"):
                        break
                    db.update_editor_repair_batch_item(
                        batch_id, chunk_index, phase, status="running",
                        started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    )
                    _editor_batch_activity(
                        db, translation_id, batch_id, "item_started",
                        (
                            f"Batch {batch_id} is repairing chunk {chunk_index + 1} "
                            f"({position}/{len(chunk_indices)})."
                        ),
                        chunk_index=chunk_index,
                        position=position,
                    )
                    ok = False
                    try:
                        result = asyncio.run(run_editor_retry(
                            translation_id=translation_id,
                            chunk_index=chunk_index,
                            checkpoint_manager=state_manager.checkpoint_manager,
                            output_dir=Path(output_dir),
                            refresh_output=False,
                            phase=phase,
                        ))
                        item_status = str(result.get("status") or "failed")
                        ok = item_status == "succeeded"
                        succeeded += int(ok)
                        failed += int(not ok)
                        db.update_editor_repair_batch_item(
                            batch_id, chunk_index, phase,
                            status=("succeeded" if ok else "failed"),
                            outcome=result.get("outcome") or item_status,
                            message=str(result.get("message") or "")[:500],
                            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        )
                    except Exception as exc:
                        failed += 1
                        db.update_editor_repair_batch_item(
                            batch_id, chunk_index, phase, status="failed",
                            outcome="failed", message=str(exc)[:500],
                            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        )
                    completed += 1
                    db.update_editor_repair_batch(
                        batch_id, completed_items=completed,
                        succeeded_items=succeeded, failed_items=failed,
                    )
                    _editor_batch_activity(
                        db, translation_id, batch_id,
                        "item_succeeded" if ok else "item_failed",
                        (
                            f"Batch {batch_id} finished chunk {chunk_index + 1}: "
                            f"{'succeeded' if ok else 'failed'}."
                        ),
                        chunk_index=chunk_index,
                        position=position,
                        item_status="succeeded" if ok else "failed",
                    )
                current = db.get_editor_repair_batch(batch_id) or {}
                if completed:
                    _editor_batch_activity(
                        db, translation_id, batch_id, "rebuilding",
                        f"Batch {batch_id} is rebuilding the translated output.",
                    )
                    checkpoint = state_manager.checkpoint_manager.load_checkpoint(translation_id) or {}
                    config = dict((checkpoint.get("job") or {}).get("config") or {})
                    output_result = asyncio.run(_refresh_output(
                        translation_id, state_manager.checkpoint_manager,
                        Path(output_dir), str(config.get("output_filename") or ""),
                        bool(config.get("bilingual_output")),
                    ))
                    _editor_batch_activity(
                        db, translation_id, batch_id, "output_rebuilt",
                        (
                            f"Batch {batch_id} output rebuild finished: "
                            f"{output_result.get('status') or 'unknown'}."
                        ),
                        output_status=output_result.get("status") or "unknown",
                    )
                final_status = "cancelled" if current.get("cancel_requested") else "completed"
                db.update_editor_repair_batch(
                    batch_id, status=final_status,
                    completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                should_resume = (
                    final_status == "completed"
                    and pause_active_job
                    and not stay_paused
                    and failed == 0
                )
                if should_resume:
                    db.update_editor_repair_batch(
                        batch_id, error="resuming",
                    )
                    _editor_batch_activity(
                        db, translation_id, batch_id, "resuming",
                        f"Batch {batch_id} completed; resuming translation.",
                    )
                    resumed = make_editor_retry_auto_resume_callback(
                        translation_id
                    )()
                    if resumed:
                        db.update_editor_repair_batch(batch_id, error=None)
                        _editor_batch_activity(
                            db, translation_id, batch_id, "resumed",
                            f"Batch {batch_id} completed and translation resumed.",
                        )
                    else:
                        db.update_editor_repair_batch(
                            batch_id, error="resume_failed",
                        )
                        _editor_batch_activity(
                            db, translation_id, batch_id, "resume_failed",
                            f"Batch {batch_id} completed but translation resume failed.",
                        )
                else:
                    _editor_batch_activity(
                        db, translation_id, batch_id, final_status,
                        f"Batch {batch_id} {final_status}; the translation remains paused.",
                    )
            except Exception as exc:
                db.update_editor_repair_batch(
                    batch_id, status="failed", error=str(exc)[:500],
                    completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                _editor_batch_activity(
                    db, translation_id, batch_id, "failed",
                    f"Batch {batch_id} failed: {type(exc).__name__}.",
                    error_type=type(exc).__name__,
                )
            finally:
                # A repair batch is an explicit pause-and-fix action.  Never
                # clear the interruption flag or auto-resume the job here.
                _release_editor_batch(translation_id)

        threading.Thread(
            target=run_batch_background,
            name=f"editor-repair-batch-{translation_id}", daemon=True,
        ).start()
        return jsonify(db.get_editor_repair_batch(batch_id)), 202
    @bp.route(
        '/api/translation/<translation_id>/editor-repair-batches/<batch_id>',
        methods=['GET'],
    )
    def get_editor_repair_batch_route(translation_id, batch_id):
        batch = state_manager.checkpoint_manager.db.get_editor_repair_batch(batch_id)
        if not batch or batch.get("translation_id") != translation_id:
            return jsonify({"error": "Repair batch not found"}), 404
        return jsonify(batch)
    @bp.route(
        '/api/translation/<translation_id>/editor-repair-batches/<batch_id>/cancel',
        methods=['POST'],
    )
    def cancel_editor_repair_batch_route(translation_id, batch_id):
        db = state_manager.checkpoint_manager.db
        batch = db.get_editor_repair_batch(batch_id)
        if not batch or batch.get("translation_id") != translation_id:
            return jsonify({"error": "Repair batch not found"}), 404
        if batch.get("status") not in {"queued", "pausing", "running"}:
            return jsonify(batch), 409
        db.update_editor_repair_batch(batch_id, cancel_requested=1)
        _editor_batch_activity(
            db, translation_id, batch_id, "cancel_requested",
            f"Cancellation requested for batch {batch_id}.",
        )
        return jsonify(db.get_editor_repair_batch(batch_id)), 202
