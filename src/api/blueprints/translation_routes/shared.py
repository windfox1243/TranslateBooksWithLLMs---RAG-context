"""Closures shared by more than one route domain.

They capture the same dependencies the route handlers do, so they are
built once per blueprint and handed to each register() call.
"""
import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

from flask import jsonify, request

from src.api.websocket import emit_update

from .helpers import _provider_credentials_error, _rehydrate_resume_credentials, logger


@dataclass(frozen=True)
class SharedHelpers:
    """The cross-domain closures, bound to one blueprint's dependencies."""
    editor_batch_activity: Callable[..., Any]
    context_revision: Callable[..., Any]
    revision_conflict: Callable[..., Any]
    export_structured_context: Callable[..., Any]
    make_context_resync_auto_resume_callback: Callable[..., Any]
    make_editor_retry_auto_resume_callback: Callable[..., Any]


def build_shared(deps) -> SharedHelpers:
    """Bind the cross-domain closures to deps."""
    state_manager = deps.state_manager
    start_translation_job = deps.start_translation_job
    socketio = deps.socketio

    def _editor_batch_activity(
        db, translation_id, batch_id, event, message, **details,
    ):
        """Persist and publish a safe repair-batch activity milestone."""
        batch = db.get_editor_repair_batch(batch_id) or {}
        payload = {
            "event": str(event),
            "batch_id": batch_id,
            "translation_id": translation_id,
            "status": batch.get("status"),
            "total_items": int(batch.get("total_items") or 0),
            "completed_items": int(batch.get("completed_items") or 0),
            "succeeded_items": int(batch.get("succeeded_items") or 0),
            "failed_items": int(batch.get("failed_items") or 0),
            "stay_paused": bool(batch.get("stay_paused", True)),
            **details,
        }
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        log_entry = {
            "timestamp": timestamp,
            "level": "ERROR" if event == "failed" else "INFO",
            "type": "general",
            "message": str(message),
            "data": {
                "ui_step": "editor_repair_batch",
                "editor_repair_batch": payload,
            },
        }
        print(
            f"[{time.strftime('%H:%M:%S')}] [Editor repair] {message}",
            flush=True,
        )
        if state_manager.exists(translation_id):
            logs = list(
                state_manager.get_translation_field(
                    translation_id, 'logs', [],
                ) or []
            )
            logs.append(log_entry)
            state_manager.set_translation_field(
                translation_id, 'logs', logs[-1000:],
            )
            if socketio is not None:
                emit_update(
                    socketio,
                    translation_id,
                    {"editor_repair_batch": payload},
                    state_manager,
                )
    def _context_revision(translation_id):
        job = state_manager.checkpoint_manager.db.get_job(translation_id) or {}
        try:
            return int((job.get("config") or {}).get("context_revision", 0))
        except (TypeError, ValueError):
            return 0
    def _revision_conflict(translation_id, payload):
        expected = payload.get("expected_revision")
        current = _context_revision(translation_id)
        if expected is None:
            return None, current
        try:
            matches = int(expected) == current
        except (TypeError, ValueError):
            matches = False
        if matches:
            return None, current
        return (jsonify({
            "error": "Context revision changed; reload structured context before saving.",
            "expected_revision": expected,
            "current_revision": current,
        }), 409), current
    def _export_structured_context(translation_id):
        """Export accepted structured state to markdown and the latest snapshot."""

        db = state_manager.checkpoint_manager.db
        job = db.get_job(translation_id) or {}
        options = (job.get("config") or {}).get("prompt_options") or {}
        filename = options.get("novel_context_file")
        if not filename:
            return False
        from src.config import NOVEL_CONTEXTS_DIR
        from src.utils.db_addressing import apply_db_addressing_to_context
        from src.utils.novel_context import (
            compress_dynamic_state,
            decode_context_snapshot,
            load_novel_context,
            normalize_novel_context_filename,
            resolve_novel_context_path,
            save_novel_context,
        )
        from src.utils.relationship_sync import apply_relationship_graph_to_context

        filename = normalize_novel_context_filename(filename)
        path = resolve_novel_context_path(filename, NOVEL_CONTEXTS_DIR)
        current = load_novel_context(path.name, path.parent)
        updated = apply_db_addressing_to_context(current, translation_id, db)
        updated = apply_relationship_graph_to_context(updated, translation_id, db)
        save_novel_context(path.name, path.parent, updated)

        chunks = db.get_chunks(translation_id)
        if chunks:
            latest = max(chunks, key=lambda item: item.get("chunk_index", -1))
            chunk_data = dict(latest.get("chunk_data") or {})
            snapshot = chunk_data.get("context_snapshot")
            if snapshot:
                snapshot_context, _global, _dynamic = decode_context_snapshot(
                    snapshot,
                    updated,
                )
                snapshot_context = apply_db_addressing_to_context(
                    snapshot_context, translation_id, db, fallback_context=updated
                )
                snapshot_context = apply_relationship_graph_to_context(
                    snapshot_context, translation_id, db, fallback_context=updated
                )
                chunk_data["context_snapshot"] = compress_dynamic_state(snapshot_context)
                db.save_chunk(
                    translation_id=translation_id,
                    chunk_index=latest["chunk_index"],
                    original_text=latest.get("original_text"),
                    translated_text=latest.get("translated_text"),
                    chunk_data=chunk_data,
                    status=latest.get("status") or "completed",
                )
        return True
    def make_context_resync_auto_resume_callback(translation_id):
        def resume_cb():
            logger.info(
                f"Auto-resuming translation {translation_id} after resync"
            )
            try:
                fresh_checkpoint = (
                    state_manager.checkpoint_manager.load_checkpoint(
                        translation_id
                    )
                )
                if not fresh_checkpoint:
                    raise RuntimeError(
                        "Translation checkpoint is unavailable."
                    )
                config = copy.deepcopy(fresh_checkpoint["job"]["config"])

                preserved_path = config.get("preserved_input_path")
                if not preserved_path:
                    preserved_path = (
                        state_manager.checkpoint_manager
                        .get_preserved_input_path(translation_id)
                    )
                if not preserved_path:
                    raise RuntimeError(
                        "The preserved input file is unavailable."
                    )

                config["file_path"] = preserved_path
                config["resume_from_index"] = fresh_checkpoint[
                    "resume_from_index"
                ]
                config["is_resume"] = True

                live_config = (
                    state_manager.get_translation_field(
                        translation_id,
                        "config",
                    )
                    or {}
                )
                provider = (config.get("llm_provider") or "ollama").lower()
                live_key = live_config.get(f"{provider}_api_key")
                _rehydrate_resume_credentials(
                    config,
                    {"api_key": live_key} if live_key else None,
                )
                credential_error = _provider_credentials_error(config)
                if credential_error is not None:
                    raise RuntimeError(credential_error["message"])

                if not state_manager.exists(translation_id):
                    if not state_manager.restore_job_from_checkpoint(
                        translation_id
                    ):
                        raise RuntimeError(
                            "Could not restore the translation job."
                        )

                state_manager.set_interrupted(translation_id, False)
                state_manager.set_translation_field(
                    translation_id,
                    "status",
                    "running",
                )
                state_manager.checkpoint_manager.mark_running(translation_id)
                emit_update(
                    socketio,
                    translation_id,
                    {
                        "status": "running",
                        "log": "Translation auto-resumed after context resync.",
                    },
                    state_manager,
                )
                start_translation_job(translation_id, config)
            except Exception as e:
                logger.error(f"Failed to auto-resume translation: {e}")
                if state_manager.exists(translation_id):
                    state_manager.set_translation_field(
                        translation_id,
                        "status",
                        "error",
                    )
                    state_manager.set_translation_field(
                        translation_id,
                        "error",
                        str(e),
                    )
                emit_update(
                    socketio,
                    translation_id,
                    {
                        "status": "error",
                        "error": str(e),
                        "log": (
                            "Context re-sync finished, but "
                            f"translation could not resume: {e}"
                        ),
                    },
                    state_manager,
                )

        return resume_cb
    def make_editor_retry_auto_resume_callback(translation_id):
        """Resume a job paused specifically for an editor-only retry."""

        def resume_cb():
            logger.info(
                f"Auto-resuming translation {translation_id} after editor retry"
            )
            try:
                fresh_checkpoint = (
                    state_manager.checkpoint_manager.load_checkpoint(
                        translation_id
                    )
                )
                if not fresh_checkpoint:
                    raise RuntimeError(
                        "Translation checkpoint is unavailable."
                    )
                config = copy.deepcopy(fresh_checkpoint["job"]["config"])

                preserved_path = config.get("preserved_input_path")
                if not preserved_path:
                    preserved_path = (
                        state_manager.checkpoint_manager
                        .get_preserved_input_path(translation_id)
                    )
                if not preserved_path:
                    raise RuntimeError(
                        "The preserved input file is unavailable."
                    )

                config["file_path"] = preserved_path
                config["resume_from_index"] = fresh_checkpoint[
                    "resume_from_index"
                ]
                config["is_resume"] = True

                live_config = (
                    state_manager.get_translation_field(
                        translation_id,
                        "config",
                    )
                    or {}
                )
                provider = (config.get("llm_provider") or "ollama").lower()
                live_key = live_config.get(f"{provider}_api_key")
                _rehydrate_resume_credentials(
                    config,
                    {"api_key": live_key} if live_key else None,
                )
                credential_error = _provider_credentials_error(config)
                if credential_error is not None:
                    raise RuntimeError(credential_error["message"])

                if not state_manager.exists(translation_id):
                    if not state_manager.restore_job_from_checkpoint(
                        translation_id
                    ):
                        raise RuntimeError(
                            "Could not restore the translation job."
                        )

                state_manager.set_interrupted(translation_id, False)
                state_manager.set_translation_field(
                    translation_id,
                    "status",
                    "running",
                )
                state_manager.checkpoint_manager.mark_running(translation_id)
                emit_update(
                    socketio,
                    translation_id,
                    {
                        "status": "running",
                        "log": (
                            "Translation auto-resumed after Senior Editor "
                            "retry."
                        ),
                    },
                    state_manager,
                )
                start_translation_job(translation_id, config)
                return True
            except Exception as exc:
                logger.error(
                    "Failed to auto-resume translation after Senior Editor "
                    f"retry: {exc}"
                )
                if state_manager.exists(translation_id):
                    state_manager.set_translation_field(
                        translation_id,
                        "status",
                        "error",
                    )
                    state_manager.set_translation_field(
                        translation_id,
                        "error",
                        str(exc),
                    )
                emit_update(
                    socketio,
                    translation_id,
                    {
                        "status": "error",
                        "error": str(exc),
                        "log": (
                            "Senior Editor retry finished, but translation "
                            f"could not resume: {exc}"
                        ),
                    },
                    state_manager,
                )
                return False

        return resume_cb

    return SharedHelpers(
        editor_batch_activity=_editor_batch_activity,
        context_revision=_context_revision,
        revision_conflict=_revision_conflict,
        export_structured_context=_export_structured_context,
        make_context_resync_auto_resume_callback=make_context_resync_auto_resume_callback,
        make_editor_retry_auto_resume_callback=make_editor_retry_auto_resume_callback,
    )
