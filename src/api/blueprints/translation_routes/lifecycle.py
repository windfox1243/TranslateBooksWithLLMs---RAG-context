"""Routes that start, observe and steer a translation job."""
import copy
import time
from pathlib import Path

from flask import jsonify, request

import src.config as _config
from src.api.api_keys import provider_env_var
from src.api.api_keys import resolve_api_key as _resolve_api_key
from src.api.services.path_validator import PathValidator
from src.config import (
    AUTO_PAUSE_ON_RATE_LIMIT,
    MIN_CHUNK_SIZE,
    OLLAMA_NUM_CTX,
    REQUEST_TIMEOUT,
)
from src.tts.tts_config import TTSConfig

from .helpers import (
    _active_translation_conflict,
    _apply_resume_overrides,
    _available_context_chunk_indices,
    _clamp_chunk_tokens,
    _clamp_parallel_workers,
    _continued_output_filename,
    _is_context_resync_active,
    _prompt_options_from_start_request,
    _refresh_inactive_context_resync_state,
    _strip_api_keys,
    _unfinished_context_resync_state,
    _update_context_resync_state,
)


def register(bp, deps, shared):
    """Register the lifecycle routes on bp."""
    state_manager = deps.state_manager
    start_translation_job = deps.start_translation_job
    uploads_dir = deps.uploads_dir

    @bp.route('/api/translate', methods=['POST'])
    def start_translation_request():
        """Start a new translation job"""
        data = request.json

        # Validate required fields
        if 'file_path' in data:
            required_fields = ['file_path', 'source_language', 'target_language',
                             'model', 'llm_api_endpoint', 'output_filename', 'file_type']
        else:
            required_fields = ['text', 'source_language', 'target_language',
                             'model', 'llm_api_endpoint', 'output_filename']

        for field in required_fields:
            if field not in data or (isinstance(data[field], str) and not data[field].strip()) or (not isinstance(data[field], str) and data[field] is None):
                if field == 'text' and data.get('file_type') == 'txt' and data.get('text') == "":
                    pass
                else:
                    return jsonify({"error": f"Missing or empty field: {field}"}), 400

        # Generate unique translation ID
        translation_id = f"trans_{int(time.time() * 1000)}"

        prompt_options = _prompt_options_from_start_request(data)
        editor_provider = str(prompt_options.get('editor_provider') or '').strip().lower()
        editor_model = str(prompt_options.get('editor_model') or '').strip()
        allowed_editor_providers = {
            '', 'ollama', 'gemini', 'openai', 'openrouter', 'mistral',
            'deepseek', 'poe', 'nim', 'litellm',
        }
        if editor_provider not in allowed_editor_providers:
            return jsonify({"error": "Unsupported Senior Editor provider"}), 400
        translation_provider = str(data.get('llm_provider') or 'ollama').lower()
        if editor_provider and editor_provider != translation_provider and not editor_model:
            return jsonify({
                "error": "A Senior Editor model is required for a separate provider"
            }), 400
        if editor_provider not in {'', 'ollama', 'openai', 'litellm'}:
            env_name = provider_env_var(editor_provider)
            editor_key = _resolve_api_key(
                data.get(f'{editor_provider}_api_key'), env_name,
            )
            if not editor_key:
                return jsonify({
                    "error": f"Senior Editor provider {editor_provider} requires an API key"
                }), 400
        if (
            prompt_options.get('auto_update_context')
            and not prompt_options.get('novel_context_file')
        ):
            from src.utils.novel_context import make_novel_context_filename
            prompt_options['novel_context_file'] = make_novel_context_filename(
                data.get('output_filename', 'translation')
            )
        if prompt_options.get('novel_context_file'):
            from src.utils.novel_context import normalize_novel_context_filename
            try:
                prompt_options['novel_context_file'] = normalize_novel_context_filename(
                    prompt_options['novel_context_file']
                )
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

        # Build configuration
        config = {
            'source_language': data['source_language'],
            'target_language': data['target_language'],
            'model': data['model'],
            'llm_api_endpoint': data['llm_api_endpoint'],
            # Keep the user-facing source name in the job config. The upload
            # path may be hashed, and the UI needs a stable name to restore a
            # running job after a browser refresh.
            'input_filename': (
                data.get('input_filename')
                or prompt_options.get('input_filename')
            ),
            'request_timeout': int(data.get('timeout', REQUEST_TIMEOUT)),
            'context_window': int(data.get('context_window', OLLAMA_NUM_CTX)),
            'max_attempts': int(data.get('max_attempts', 2)),
            'retry_delay': int(data.get('retry_delay', 2)),
            'parallel_workers': _clamp_parallel_workers(data.get('parallel_workers')),
            'output_filename': data['output_filename'],
            'llm_provider': data.get('llm_provider', 'ollama'),
            'gemini_api_key': _resolve_api_key(data.get('gemini_api_key'), 'GEMINI_API_KEY'),
            'openai_api_key': _resolve_api_key(data.get('openai_api_key'), 'OPENAI_API_KEY'),
            'openrouter_api_key': _resolve_api_key(data.get('openrouter_api_key'), 'OPENROUTER_API_KEY'),
            'mistral_api_key': _resolve_api_key(data.get('mistral_api_key'), 'MISTRAL_API_KEY'),
            'deepseek_api_key': _resolve_api_key(data.get('deepseek_api_key'), 'DEEPSEEK_API_KEY'),
            'poe_api_key': _resolve_api_key(data.get('poe_api_key'), 'POE_API_KEY'),
            'nim_api_key': _resolve_api_key(data.get('nim_api_key'), 'NIM_API_KEY'),
            # Prompt options (optional instructions to include in the system prompt)
            'prompt_options': prompt_options,
            # Auto-pause on rate limit toggle (request overrides .env default)
            'auto_pause_on_rate_limit': data.get('auto_pause_on_rate_limit', AUTO_PAUSE_ON_RATE_LIMIT),
            # Bilingual output (original + translation interleaved)
            'bilingual_output': data.get('bilingual_output', False),
            # Refine-only mode (skip translation, run only refinement on input)
            'refine_only': data.get('refine_only', False),
            # Chained refinement pass after translation
            'refine_after': data.get('refine_after', False),
            # TTS configuration
            'tts_enabled': data.get('tts_enabled', False),
            'tts_config': TTSConfig.from_web_request(data).to_dict() if data.get('tts_enabled') else None,
            # Chunker settings persisted for resume consistency
            'max_tokens_per_chunk': _clamp_chunk_tokens(
                data.get('max_tokens_per_chunk')
            ),
            'soft_limit_ratio': float(
                data.get('soft_limit_ratio')
                or _config.SOFT_LIMIT_RATIO
            ),
            'min_chunk_size': int(data.get('min_chunk_size') or MIN_CHUNK_SIZE),
        }

        # Add file-specific or text-specific configuration
        if 'file_path' in data:
            # The client supplies this path, so it must be confined to the
            # uploads directory — otherwise any server-readable file (.env, SSH
            # keys, /etc/passwd) could be "translated" into a downloadable
            # output. See issue #209.
            safe_path, path_error = PathValidator.validate_upload_path(
                data['file_path'], uploads_dir
            )
            if path_error is not None:
                return jsonify({"error": path_error}), 403
            config['file_path'] = str(safe_path)
            config['file_type'] = data['file_type']
            if data.get('refinement_original_path'):
                original_path, original_error = PathValidator.validate_upload_path(
                    data['refinement_original_path'],
                    uploads_dir,
                )
                if original_error is not None:
                    return jsonify({"error": original_error}), 403
                config['refinement_original_path'] = str(original_path)
        else:
            config['text'] = data['text']
            config['file_type'] = data.get('file_type', 'txt')

        # Create translation in state manager
        state_manager.create_translation(translation_id, config)

        # Start translation job
        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation queued.",
            # Strip keys from a copy — `config` is the live job's dict, and a
            # '__USE_ENV__' request must not get the resolved .env key back.
            "config_received": _strip_api_keys(dict(config))
        })
    @bp.route('/api/translation/<translation_id>', methods=['GET'])
    def get_translation_job_status(translation_id):
        """Get status of a translation job"""
        job_data = state_manager.get_translation(translation_id)
        if not job_data:
            return jsonify({"error": "Translation not found"}), 404

        stats = job_data.get('stats', {
            'start_time': time.time(),
            'total_chunks': 0,
            'completed_chunks': 0,
            'failed_chunks': 0,
            'review_required_chunks': 0,
        })

        # Calculate elapsed time
        if job_data.get('status') == 'running' or job_data.get('status') == 'queued':
            elapsed = time.time() - stats.get('start_time', time.time())
        else:
            elapsed = stats.get('elapsed_time', time.time() - stats.get('start_time', time.time()))

        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(
            translation_id
        )

        return jsonify({
            "translation_id": translation_id,
            "status": job_data.get('status'),
            "quality_status": job_data.get(
                'quality_status',
                'review_required'
                if stats.get('review_required_chunks', 0) else 'not_checked',
            ),
            "progress": job_data.get('progress'),
            "stats": {
                'total_chunks': stats.get('total_chunks', 0),
                'completed_chunks': stats.get('completed_chunks', 0),
                'failed_chunks': stats.get('failed_chunks', 0),
                'review_required_chunks': stats.get('review_required_chunks', 0),
                'start_time': stats.get('start_time'),
                'elapsed_time': elapsed,
                'context_chunk_indices': _available_context_chunk_indices(
                    checkpoint_data
                ),
            },
            "logs": job_data.get('logs', [])[-100:],
            "result_preview": "[Preview functionality removed. Download file to view content.]" if job_data.get('status') in ['completed', 'interrupted', 'partial'] else None,
            "error": job_data.get('error'),
            "config": _strip_api_keys(dict(job_data['config'])) if job_data.get('config') else None,
            "output_filepath": job_data.get('output_filepath')
        })
    @bp.route('/api/translation/<translation_id>/interrupt', methods=['POST'])
    def interrupt_translation_job(translation_id):
        """Interrupt a running translation job"""
        if not state_manager.exists(translation_id):
            return jsonify({"error": "Translation not found"}), 404

        job_data = state_manager.get_translation(translation_id)
        status = job_data.get('status')
        if status in ('running', 'queued'):
            state_manager.set_interrupted(translation_id, True)
            return jsonify({
                "message": "Interruption signal sent. Translation will stop after the current segment."
            }), 200

        if status == 'rate_limited':
            # Cancels any in-flight auto-resume sleep and stops the UI from treating
            # the job as still-active.
            state_manager.set_interrupted(translation_id, True)
            state_manager.set_translation_field(translation_id, 'status', 'interrupted')
            return jsonify({
                "message": "Auto-resume cancelled. Translation marked interrupted; you can resume manually later."
            }), 200

        return jsonify({
            "message": "The translation is not in an interruptible state (e.g., already completed or failed)."
        }), 400
    @bp.route('/api/translations', methods=['GET'])
    def list_all_translations():
        """List all translation jobs"""
        summary_list = state_manager.get_translation_summaries()
        return jsonify({"translations": summary_list})
    @bp.route('/api/resumable', methods=['GET'])
    def list_resumable_jobs():
        """List all jobs that can be resumed.

        Persisted checkpoints no longer hold API keys (issue #213), but strip
        defensively anyway — the resume endpoint resolves keys server-side from
        .env or the request body, so the client never needs them.
        """
        resumable_jobs = state_manager.get_resumable_jobs()
        for job in resumable_jobs:
            config = job.get('config') or {}
            job['context_resync'] = _refresh_inactive_context_resync_state(
                state_manager.checkpoint_manager,
                job.get('translation_id'),
                config,
            )
            _strip_api_keys(job.get('config'))
        return jsonify({"resumable_jobs": resumable_jobs})
    @bp.route('/api/resume/<translation_id>', methods=['POST'])
    def resume_translation_job_endpoint(translation_id):
        """Resume a paused or interrupted translation job"""
        # Check if there are any active translations
        active_error = _active_translation_conflict(state_manager)
        if active_error is not None:
            return active_error

        # Check if checkpoint exists
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "No checkpoint found for this translation"}), 404

        # Get job config and add resume parameters
        job = checkpoint_data['job']
        config = copy.deepcopy(job['config'])  # Create a deep copy to avoid mutating the stored config
        resync_state = _unfinished_context_resync_state(config)
        if resync_state:
            if (
                resync_state.get("status") == "running"
                and not _is_context_resync_active(translation_id)
            ):
                resync_state = _update_context_resync_state(
                    state_manager.checkpoint_manager,
                    translation_id,
                    {
                        "status": "paused",
                        "pause_requested": False,
                        "updated_at": time.time(),
                    },
                    base_config=config,
                ) or resync_state
            return jsonify({
                "error": (
                    "Context re-sync is not complete. Resume or finish the "
                    "context re-sync before resuming translation."
                ),
                "context_resync_state": resync_state,
            }), 409

        # Restore job into state manager
        restored = state_manager.restore_job_from_checkpoint(translation_id)
        if not restored:
            return jsonify({"error": "Failed to restore job from checkpoint"}), 500

        # Get preserved input file path if exists
        # Always use preserved_input_path from config (stored during job creation)
        # This ensures consistent file path across multiple resume cycles
        preserved_path = config.get('preserved_input_path')
        if preserved_path:
            # Verify that the preserved file actually exists
            from pathlib import Path
            if Path(preserved_path).exists():
                config['file_path'] = preserved_path
            else:
                return jsonify({
                    "error": "Preserved input file not found",
                    "message": f"The preserved input file for this job no longer exists: {preserved_path}",
                    "suggestion": "This job cannot be resumed. Please delete this checkpoint and start a new translation."
                }), 404
        else:
            # Fallback: try to get it from checkpoint manager
            preserved_path_fallback = state_manager.checkpoint_manager.get_preserved_input_path(translation_id)
            if preserved_path_fallback:
                config['file_path'] = preserved_path_fallback
            else:
                return jsonify({
                    "error": "No preserved input file",
                    "message": "This job has no preserved input file and cannot be resumed.",
                    "suggestion": "Please delete this checkpoint and start a new translation."
                }), 404

        # Add resume parameters to config
        config['resume_from_index'] = checkpoint_data['resume_from_index']
        config['is_resume'] = True

        # Optional model/provider overrides for the remaining chunks (issue #183).
        # No body = unchanged behavior.
        overrides = request.get_json(silent=True) or {}
        override_error = _apply_resume_overrides(config, overrides)
        if override_error is not None:
            return override_error

        # Update both the in-memory state and the durable checkpoint database
        state_manager.checkpoint_manager.update_job_config(translation_id, config)
        state_manager.set_translation_field(translation_id, 'config', config)

        # Mark as running in database
        state_manager.checkpoint_manager.mark_running(translation_id)

        # Start the translation job (the wrapper will inject dependencies)
        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation resumed successfully",
            "resume_from_chunk": checkpoint_data['resume_from_index'],
            "model": config.get('model'),
            "llm_provider": config.get('llm_provider')
        }), 200
    @bp.route('/api/continue/<translation_id>', methods=['POST'])
    def continue_translation_job_endpoint(translation_id):
        """Create a new job that translates only content added after a checkpoint."""
        active_error = _active_translation_conflict(
            state_manager,
            action="continue",
        )
        if active_error is not None:
            return active_error

        data = request.get_json(silent=True) or {}
        if not data.get('file_path'):
            return jsonify({"error": "Missing updated file_path"}), 400

        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(
            translation_id
        )
        if not checkpoint_data:
            return jsonify({"error": "No checkpoint found for this translation"}), 404

        base_job = checkpoint_data['job']
        base_config = copy.deepcopy(base_job.get('config') or {})
        base_file_type = base_job.get('file_type') or base_config.get('file_type')

        safe_path, path_error = PathValidator.validate_upload_path(
            data['file_path'],
            uploads_dir,
        )
        if path_error is not None:
            return jsonify({"error": path_error}), 403

        updated_file_type = data.get('file_type') or base_file_type
        if updated_file_type != base_file_type:
            return jsonify({
                "error": (
                    "Updated file type must match the previous translation "
                    f"({base_file_type})."
                )
            }), 400

        new_translation_id = f"trans_{int(time.time() * 1000)}"
        config = base_config
        config.update({
            'file_path': str(safe_path),
            'preserved_input_path': str(safe_path),
            'input_filename': data.get('input_filename') or safe_path.name,
            'output_filename': _continued_output_filename(
                data.get('output_filename') or base_config.get('output_filename')
            ),
            'file_type': base_file_type,
            'resume_from_index': 0,
            'is_resume': False,
            'continuation_base_id': translation_id,
            'continuation_mode': 'matching_prefix',
        })
        for transient_key in (
            '_context_resync',
            '_context_resync_refinement',
            '_force_output_filepath',
            'output_filepath',
        ):
            config.pop(transient_key, None)

        prompt_options = dict(config.get('prompt_options') or {})
        if prompt_options.get('novel_context_file'):
            from src.utils.novel_context import normalize_novel_context_filename
            try:
                prompt_options['novel_context_file'] = (
                    normalize_novel_context_filename(
                        prompt_options['novel_context_file']
                    )
                )
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        config['prompt_options'] = prompt_options

        state_manager.create_translation(new_translation_id, config)
        start_translation_job(new_translation_id, config)

        return jsonify({
            "translation_id": new_translation_id,
            "base_translation_id": translation_id,
            "message": "Continuation queued.",
            "output_filename": config['output_filename'],
        }), 200
    @bp.route('/api/checkpoint/<translation_id>', methods=['DELETE'])
    def delete_checkpoint_endpoint(translation_id):
        """Delete a checkpoint (manual cleanup by user)"""
        delete_context = str(
            request.args.get('delete_novel_context', '')
        ).strip().lower() in ('1', 'true', 'yes')
        result = state_manager.delete_checkpoint(
            translation_id, delete_novel_context=delete_context
        )

        if result.deleted:
            return jsonify({
                "message": "Checkpoint deleted successfully",
                "translation_id": translation_id,
                "novel_context_removed": result.novel_context_removed,
                "novel_context_kept_for": result.novel_context_kept_for,
            }), 200
        else:
            return jsonify({"error": "Failed to delete checkpoint or checkpoint not found"}), 404
