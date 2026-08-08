"""Routes for reading and resynchronizing the novel-context snapshot."""
import copy
import threading
import time

from flask import jsonify, request

from src.api.websocket import emit_update
from src.utils.unified_logger import get_logger

from .helpers import (
    _apply_resume_overrides,
    _available_context_chunk_indices,
    _build_corrective_refinement_config,
    _claim_context_resync,
    _context_resync_state_from_config,
    _is_context_resync_active,
    _provider_credentials_error,
    _refresh_inactive_context_resync_state,
    _rehydrate_resume_credentials,
    _release_context_resync,
    _update_context_resync_state,
    logger,
)


def register(bp, deps, shared):
    """Register the context routes on bp."""
    state_manager = deps.state_manager
    start_translation_job = deps.start_translation_job
    socketio = deps.socketio
    make_context_resync_auto_resume_callback = shared.make_context_resync_auto_resume_callback

    @bp.route('/api/translation/<translation_id>/context/<int:chunk_index>', methods=['GET'])
    def get_context_snapshot(translation_id, chunk_index):
        """Fetch the dynamic context snapshot for a specific chunk"""
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "Translation not found"}), 404
            
        chunks = checkpoint_data.get('chunks', [])
        
        # Find the specific chunk index (may not exist yet during active translation)
        target_chunk = None
        for chunk in chunks:
            if chunk.get('chunk_index') == chunk_index:
                target_chunk = chunk
                break
        
        # Extract snapshot from chunk data if available
        snapshot = None
        if target_chunk:
            chunk_data = target_chunk.get('chunk_data') or {}
            snapshot = chunk_data.get('context_snapshot')
            
        plain_text_context = ""
        
        config = checkpoint_data.get('job', {}).get('config', {}) or {}
        novel_context_file = config.get('prompt_options', {}).get('novel_context_file')
        auto_update_context = config.get('prompt_options', {}).get('auto_update_context', False)
        
        if not novel_context_file and auto_update_context:
            from src.utils.novel_context import make_novel_context_filename
            novel_context_file = make_novel_context_filename(
                config.get('output_filename', 'translation')
            )
            
            # Update config copy and save back to the DB to repair permanently
            new_config = dict(config)
            if 'prompt_options' not in new_config:
                new_config['prompt_options'] = {}
            else:
                new_config['prompt_options'] = dict(new_config['prompt_options'])
            new_config['prompt_options']['novel_context_file'] = novel_context_file
            
            try:
                state_manager.checkpoint_manager.update_job_config(translation_id, new_config)
            except Exception as persist_err:
                from src.utils.unified_logger import get_logger
                get_logger(__name__).warning(f"Could not persist repaired novel_context_file to database: {persist_err}")
        
        if novel_context_file:
            from src.config import NOVEL_CONTEXTS_DIR
            from src.utils.novel_context import (
                decode_context_snapshot,
                load_novel_context,
                normalize_novel_context_filename,
                resolve_novel_context_path,
            )
            
            full_context = ""
            try:
                novel_context_file = normalize_novel_context_filename(novel_context_file)
                path = resolve_novel_context_path(novel_context_file, NOVEL_CONTEXTS_DIR)
                full_context = load_novel_context(path.name, path.parent)
            except Exception as e:
                from src.utils.unified_logger import get_logger
                get_logger(__name__).error(
                    f"Failed to load or parse context snapshot for file "
                    f"{novel_context_file}: {e}"
                )
            
            if snapshot:
                historical_context, _, _ = decode_context_snapshot(
                    snapshot,
                    full_context,
                )
                if request.args.get('scope') == 'global_lore':
                    from src.utils.novel_context import normalize_refinement_context

                    # Explicit global edits should use the latest book-wide
                    # lore while borrowing this chunk's dynamic-state anchor.
                    plain_text_context = normalize_refinement_context(
                        historical_context,
                        full_context,
                    )
                else:
                    # Historical chunk views should be timeline-safe: show the
                    # exact stored snapshot instead of mixing in future global
                    # facts from the latest context file.
                    plain_text_context = historical_context
            else:
                plain_text_context = full_context
        
        return jsonify({
            "translation_id": translation_id,
            "chunk_index": chunk_index,
            "context_content": plain_text_context,
            "has_novel_context": bool(novel_context_file) or bool(auto_update_context),
            "status": target_chunk.get('status') if target_chunk else 'pending',
            "available_chunk_indices": _available_context_chunk_indices(
                checkpoint_data
            ),
        }), 200
    @bp.route('/api/translation/<translation_id>/context/<int:chunk_index>/resync', methods=['POST'])
    def resync_context_snapshot(translation_id, chunk_index):
        """Update a context snapshot and trigger a background re-sync for subsequent chunks"""
        from src.utils.unified_logger import get_logger
        logger = get_logger(__name__)
        logger.info(f"Received context resync request for translation {translation_id} at chunk {chunk_index}")
        
        data = request.json
        if not data or 'context_content' not in data:
            logger.error("Context resync failed: Missing context_content in request data")
            return jsonify({"error": "Missing context_content"}), 400
            
        new_content = data['context_content']
        if not isinstance(new_content, str):
            return jsonify({"error": "context_content must be a string"}), 400
        if len(new_content.encode('utf-8')) > 2 * 1024 * 1024:
            return jsonify({"error": "Context content is too large"}), 413

        from src.utils.novel_context import (
            build_novel_context,
            compress_dynamic_state,
            decode_context_snapshot,
            extract_dynamic_state_from_text,
            extract_global_lore,
        )
        dynamic_state = extract_dynamic_state_from_text(new_content)
        if dynamic_state is None:
            return jsonify({
                "error": "Context content must include DYNAMIC_STATE_START and DYNAMIC_STATE_END markers"
            }), 400
        new_content = build_novel_context(
            extract_global_lore(new_content),
            dynamic_state,
        )
        compressed_snapshot = compress_dynamic_state(new_content)
        
        # 1. Update the DB for the target chunk
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            logger.error(f"Context resync failed: Translation {translation_id} not found")
            return jsonify({"error": "Translation not found"}), 404
            
        chunks = checkpoint_data.get('chunks', [])
        target_chunk_idx = None
        for i, chunk in enumerate(chunks):
            if chunk.get('chunk_index') == chunk_index:
                target_chunk_idx = i
                break
                
        if target_chunk_idx is None:
            logger.info(
                f"Context snapshot {chunk_index} is no longer available for "
                f"translation {translation_id}."
            )
            return jsonify({"error": "Chunk is not available for resync"}), 409

        target_chunk = chunks[target_chunk_idx]
        status = target_chunk.get('status')
        if status not in ('completed', 'partial', 'failed'):
            return jsonify({"error": "Only chunks with context snapshots can be resynced"}), 409
        if not (target_chunk.get('chunk_data') or {}).get('context_snapshot'):
            return jsonify({"error": "Chunk has no context snapshot to resync"}), 409
        if not _claim_context_resync(translation_id):
            return jsonify({"error": "A context resync is already running for this translation"}), 409

        requested_scope = data.get("scope")
        previous_snapshot = (target_chunk.get('chunk_data') or {}).get(
            'context_snapshot'
        )
        _, _, previous_dynamic_state = decode_context_snapshot(
            previous_snapshot,
            "",
        )
        global_only_resync = (
            requested_scope == "global_lore"
            and previous_dynamic_state.strip() == dynamic_state.strip()
        )

        if target_chunk.get('chunk_data') is None:
            target_chunk['chunk_data'] = {}
        target_chunk['chunk_data']['context_snapshot'] = compressed_snapshot

        original_text = target_chunk.get('original_text')
        translated_text = target_chunk.get('translated_text')
        chunk_data = target_chunk.get('chunk_data')
            
        try:
            state_manager.checkpoint_manager.db.save_chunk(
                translation_id=translation_id,
                chunk_index=chunk_index,
                original_text=original_text,
                translated_text=translated_text,
                chunk_data=chunk_data,
                status=status
            )
        except Exception:
            _release_context_resync(translation_id)
            raise
        
        # Any refinement produced from the previous snapshots is now stale.
        context_revision = (
            state_manager.checkpoint_manager.mark_refinement_stale(
                translation_id
            )
        )
        if context_revision is not None:
            logger.info(
                f"Context revision {context_revision} recorded for "
                f"translation {translation_id}."
            )

        # 2. Trigger background resync task
        from src.core.adapters.generic_translator import (
            resync_context_snapshots_background,
        )
        
        job_status = state_manager.get_translation(translation_id)
        was_active = False
        auto_resume_callback = None
        post_resync_callback = None
        post_resync_message = None

        def start_corrective_refinement():
            """Replay refinement once against the fully re-synced snapshots."""
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

                persisted_config = copy.deepcopy(
                    fresh_checkpoint["job"]["config"]
                )
                live_output_path = state_manager.get_translation_field(
                    translation_id,
                    "output_filepath",
                )
                correction_config = _build_corrective_refinement_config(
                    persisted_config,
                    live_output_path,
                )
                if correction_config is None:
                    raise RuntimeError(
                        "The preserved first-pass translation or final output "
                        "is unavailable. Run refinement manually to apply the "
                        "re-synced context."
                    )

                live_config = (
                    state_manager.get_translation_field(
                        translation_id,
                        "config",
                    )
                    or {}
                )
                provider = (
                    correction_config.get("llm_provider") or "ollama"
                ).lower()
                live_key = live_config.get(f"{provider}_api_key")
                _rehydrate_resume_credentials(
                    correction_config,
                    {"api_key": live_key} if live_key else None,
                )
                credential_error = _provider_credentials_error(
                    correction_config
                )
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
                state_manager.set_translation_field(
                    translation_id,
                    "output_filepath",
                    correction_config["output_filepath"],
                )
                state_manager.checkpoint_manager.mark_running(translation_id)
                emit_update(
                    socketio,
                    translation_id,
                    {
                        "status": "running",
                        "log": (
                            "Context re-sync changed refinement inputs; "
                            "restarting refinement from the preserved "
                            "first-pass translation."
                        ),
                    },
                    state_manager,
                )
                start_translation_job(
                    translation_id,
                    correction_config,
                )
            except Exception as e:
                logger.error(
                    f"Failed to start corrective refinement: {e}"
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
                        str(e),
                    )
                emit_update(
                    socketio,
                    translation_id,
                    {
                        "status": "error",
                        "error": str(e),
                        "log": (
                            "Context re-sync completed, but corrective "
                            f"refinement could not start: {e}"
                        ),
                    },
                    state_manager,
                )
        
        if job_status and job_status.get('status') == 'running':
            was_active = True
            logger.info(f"Translation {translation_id} is running. Interrupting for context resync...")
            state_manager.set_interrupted(translation_id, True)
            emit_update(
                socketio,
                translation_id,
                {
                    "log": (
                        "⏸️ Context re-sync requested: waiting for active "
                        "chunk to finish before pausing translation..."
                    ),
                },
                state_manager,
            )

            current_phase = (job_status.get("stats") or {}).get(
                "current_phase"
            )
            if (
                checkpoint_data.get("job", {}).get("config", {}).get(
                    "refine_after"
                )
                and current_phase == 2
            ):
                post_resync_callback = start_corrective_refinement
                post_resync_message = (
                    "Context timeline repaired; restarting corrective "
                    "refinement..."
                )
            else:
                auto_resume_callback = (
                    make_context_resync_auto_resume_callback(translation_id)
                )
        else:
            persisted_status = (
                checkpoint_data.get("job", {})
                .get("progress", {})
                .get("status")
            )
            persisted_config = (
                checkpoint_data.get("job", {}).get("config", {})
            )
            if (
                persisted_status == "completed"
                and persisted_config.get("refine_after")
            ):
                post_resync_callback = start_corrective_refinement
                post_resync_message = (
                    "Context timeline repaired; starting corrective "
                    "refinement..."
                )

        follow_up_kind = None
        if post_resync_callback:
            follow_up_kind = "corrective_refinement"
        elif auto_resume_callback:
            follow_up_kind = "auto_resume_translation"

        persisted_state = _update_context_resync_state(
            state_manager.checkpoint_manager,
            translation_id,
            {
                "status": "running",
                "pause_requested": False,
                "start_chunk_index": chunk_index,
                "last_processed_chunk": chunk_index,
                "context_revision": context_revision,
                "follow_up_kind": follow_up_kind,
                "mode": "global_lore" if global_only_resync else "timeline_replay",
                "was_active": was_active,
                "updated_at": time.time(),
            },
        )
        if persisted_state is None:
            _release_context_resync(translation_id)
            return jsonify({"error": "Failed to persist context resync state"}), 500
            
        logger.info(f"Dispatching background context resync thread for translation {translation_id} starting at chunk {chunk_index}")
        
        # Run in a background thread so we don't block the API
        def run_resync():
            try:
                resync_context_snapshots_background(
                    translation_id,
                    chunk_index,
                    compressed_snapshot,
                    socketio,
                    was_active,
                    auto_resume_callback,
                    post_resync_callback,
                    post_resync_message,
                    global_only_resync,
                )
            finally:
                _release_context_resync(translation_id)

        thread = threading.Thread(
            target=run_resync,
            name=f"context-resync-{translation_id}",
        )
        thread.daemon = True
        try:
            thread.start()
        except Exception:
            _release_context_resync(translation_id)
            raise
        
        return jsonify({
            "message": "Context resync started successfully",
            "translation_id": translation_id,
            "chunk_index": chunk_index,
            "context_revision": context_revision,
            "resync_state": persisted_state,
        }), 200
    @bp.route('/api/translation/<translation_id>/context/resync/status', methods=['GET'])
    def get_context_resync_status(translation_id):
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "Translation not found"}), 404
        config = checkpoint_data.get('job', {}).get('config', {}) or {}
        state = _refresh_inactive_context_resync_state(
            state_manager.checkpoint_manager,
            translation_id,
            config,
        )
        return jsonify({
            "translation_id": translation_id,
            "active": _is_context_resync_active(translation_id),
            "resync_state": state,
            "staging_state": (
                state_manager.checkpoint_manager.db
                .get_latest_context_resync_run(translation_id)
            ),
        }), 200
    @bp.route('/api/translation/<translation_id>/context/resync/pause', methods=['POST'])
    def pause_context_resync(translation_id):
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "Translation not found"}), 404
        config = checkpoint_data.get('job', {}).get('config', {}) or {}
        state = _context_resync_state_from_config(config)
        if state.get("status") not in ("running", "pause_requested"):
            return jsonify({"error": "No running context resync to pause"}), 409
        updated = _update_context_resync_state(
            state_manager.checkpoint_manager,
            translation_id,
            {
                "status": "pause_requested",
                "pause_requested": True,
                "updated_at": time.time(),
            },
            base_config=config,
        )
        if updated is None:
            return jsonify({"error": "Failed to persist context resync pause"}), 500
        return jsonify({
            "message": "Context resync pause requested",
            "translation_id": translation_id,
            "resync_state": updated,
        }), 200
    @bp.route('/api/translation/<translation_id>/context/resync/resume', methods=['POST'])
    def resume_context_resync(translation_id):
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "Translation not found"}), 404
        config = copy.deepcopy(checkpoint_data.get('job', {}).get('config', {}) or {})
        state = _context_resync_state_from_config(config)
        if state.get("status") not in ("paused", "pause_requested", "running"):
            return jsonify({"error": "No paused context resync to resume"}), 409
        if _is_context_resync_active(translation_id):
            return jsonify({"error": "A context resync is already running for this translation"}), 409

        resume_chunk = state.get("last_processed_chunk", state.get("start_chunk_index"))
        try:
            resume_chunk = int(resume_chunk)
        except (TypeError, ValueError):
            return jsonify({"error": "Context resync resume point is invalid"}), 409

        source_chunk = next(
            (
                chunk for chunk in checkpoint_data.get('chunks', [])
                if chunk.get('chunk_index') == resume_chunk
            ),
            None,
        )
        resume_snapshot = (
            (source_chunk or {}).get('chunk_data') or {}
        ).get('context_snapshot')
        if not resume_snapshot:
            return jsonify({"error": "Context resync resume snapshot is unavailable"}), 409

        overrides = request.get_json(silent=True) or {}
        override_error = _apply_resume_overrides(config, overrides)
        if override_error is not None:
            return override_error

        config['_context_resync'] = {
            **state,
            "status": "running",
            "pause_requested": False,
            "last_processed_chunk": resume_chunk,
            "updated_at": time.time(),
        }
        if not state_manager.checkpoint_manager.update_job_config(translation_id, config):
            return jsonify({"error": "Failed to persist context resync resume"}), 500
        if not _claim_context_resync(translation_id):
            return jsonify({"error": "A context resync is already running for this translation"}), 409

        from src.core.adapters.generic_translator import (
            resync_context_snapshots_background,
        )
        auto_resume_callback = None
        if state.get("follow_up_kind") == "auto_resume_translation":
            auto_resume_callback = make_context_resync_auto_resume_callback(
                translation_id
            )

        def run_resync():
            try:
                resync_context_snapshots_background(
                    translation_id,
                    resume_chunk,
                    resume_snapshot,
                    socketio,
                    False,
                    auto_resume_callback,
                    None,
                    None,
                    state.get("mode") == "global_lore",
                )
            finally:
                _release_context_resync(translation_id)

        thread = threading.Thread(
            target=run_resync,
            name=f"context-resync-{translation_id}",
        )
        thread.daemon = True
        try:
            thread.start()
        except Exception:
            _release_context_resync(translation_id)
            raise

        return jsonify({
            "message": "Context resync resumed successfully",
            "translation_id": translation_id,
            "resume_from_chunk": resume_chunk,
            "model": config.get('model'),
            "llm_provider": config.get('llm_provider'),
            "resync_state": config['_context_resync'],
        }), 200
