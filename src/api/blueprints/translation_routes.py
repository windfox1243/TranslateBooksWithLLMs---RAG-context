"""
Translation job management routes
"""
import os
import asyncio
import time
import copy
import threading
import shutil
from pathlib import Path
from flask import Blueprint, request, jsonify

import src.config as _config
from src.api.websocket import emit_update
from src.api.services.path_validator import PathValidator
from src.config import (
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    AUTO_PAUSE_ON_RATE_LIMIT,
    MAX_PARALLEL_TRANSLATIONS,
    MIN_CHUNK_SIZE,
)
from src.tts.tts_config import TTSConfig
from src.api.api_keys import (
    provider_env_var,
    resolve_api_key as _resolve_api_key,
)
from src.utils.unified_logger import get_logger

logger = get_logger(__name__)


def _clamp_parallel_workers(value):
    """Clamp the requested worker count to [1, MAX_PARALLEL_TRANSLATIONS].

    Falls back to the PARALLEL_TRANSLATIONS default when absent or malformed.
    Local-provider gating happens later in resolve_parallel_workers().
    """
    if value is None:
        return _config.PARALLEL_TRANSLATIONS
    try:
        return max(1, min(MAX_PARALLEL_TRANSLATIONS, int(value)))
    except (TypeError, ValueError):
        return _config.PARALLEL_TRANSLATIONS


def _clamp_chunk_tokens(value):
    """Resolve the per-job token budget from request or live .env config."""
    if value in (None, ""):
        _config.reload_config()
    try:
        resolved = int(value or _config.MAX_TOKENS_PER_CHUNK)
    except (TypeError, ValueError):
        resolved = int(_config.MAX_TOKENS_PER_CHUNK)
    return max(50, resolved)


def _prompt_options_from_start_request(data):
    """Return start-request prompt options with legacy reflection fallback."""
    prompt_options = dict((data or {}).get('prompt_options') or {})
    prompt_options.setdefault('use_relationship_reasoning', 'project')
    prompt_options.setdefault('use_relationship_llm_judge', 'selective')
    prompt_options.setdefault('context_contract_version', 3)
    prompt_options.setdefault('source_residue_validation', True)
    if 'reflection_mode' not in prompt_options:
        prompt_options['reflection_mode'] = (
            str(getattr(_config, 'ENABLE_CHUNK_REFLECTION', 'false')).lower()
            == 'true'
        )
    from src.core.llm.generation_controls import normalize_thinking_mode

    prompt_options['draft_thinking_level'] = normalize_thinking_mode(
        prompt_options.get('draft_thinking_level')
    )
    prompt_options['editor_thinking_level'] = normalize_thinking_mode(
        prompt_options.get('editor_thinking_level')
    )
    raw_output = str(
        prompt_options.get('editor_max_output_tokens') or 'auto'
    ).strip().casefold()
    if raw_output not in {'auto', 'model_max'}:
        try:
            raw_output = str(max(1024, min(int(raw_output), 65536)))
        except (TypeError, ValueError):
            raw_output = 'auto'
    prompt_options['editor_max_output_tokens'] = raw_output
    try:
        prompt_options['editor_model_output_limit'] = max(
            0,
            min(int(prompt_options.get('editor_model_output_limit') or 0), 65536),
        )
    except (TypeError, ValueError):
        prompt_options['editor_model_output_limit'] = 0
    prompt_options['draft_reasoning_supported'] = bool(
        prompt_options.get('draft_reasoning_supported')
    )
    prompt_options['editor_reasoning_supported'] = bool(
        prompt_options.get('editor_reasoning_supported')
    )
    return prompt_options


# Cloud providers whose key lives in config['<provider>_api_key'] and env var
# '<PROVIDER>_API_KEY'. The mapping is mechanical, so supporting a new provider
# in the resume-override path requires only adding it here (and nowhere else in
# this file).
_KEY_PROVIDERS = ('gemini', 'openai', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim')

# Providers that talk to a user-supplied endpoint; the others use a built-in one.
_ENDPOINT_PROVIDERS = ('ollama', 'openai')

_CONTEXT_RESYNC_LOCK = threading.Lock()
_ACTIVE_CONTEXT_RESYNCS = set()
_EDITOR_RETRY_LOCK = threading.Lock()
_ACTIVE_EDITOR_RETRIES = set()


def _claim_context_resync(translation_id):
    with _CONTEXT_RESYNC_LOCK:
        if translation_id in _ACTIVE_CONTEXT_RESYNCS:
            return False
        _ACTIVE_CONTEXT_RESYNCS.add(translation_id)
        return True


def _release_context_resync(translation_id):
    with _CONTEXT_RESYNC_LOCK:
        _ACTIVE_CONTEXT_RESYNCS.discard(translation_id)


def _is_context_resync_active(translation_id):
    with _CONTEXT_RESYNC_LOCK:
        return translation_id in _ACTIVE_CONTEXT_RESYNCS


def _claim_editor_retry(translation_id, chunk_index):
    key = (str(translation_id), int(chunk_index))
    with _EDITOR_RETRY_LOCK:
        if any(item[0] == key[0] for item in _ACTIVE_EDITOR_RETRIES):
            return False
        _ACTIVE_EDITOR_RETRIES.add(key)
        return True


def _release_editor_retry(translation_id, chunk_index):
    with _EDITOR_RETRY_LOCK:
        _ACTIVE_EDITOR_RETRIES.discard((str(translation_id), int(chunk_index)))


def _context_resync_state_from_config(config):
    state = (config or {}).get('_context_resync')
    return dict(state) if isinstance(state, dict) else {}


def _unfinished_context_resync_state(config):
    state = _context_resync_state_from_config(config)
    if state.get("status") in ("running", "pause_requested", "paused"):
        return state
    return None


def _refresh_inactive_context_resync_state(checkpoint_manager, translation_id, config):
    state = _context_resync_state_from_config(config)
    if state.get("status") == "running" and not _is_context_resync_active(translation_id):
        state = _update_context_resync_state(
            checkpoint_manager,
            translation_id,
            {
                "status": "paused",
                "pause_requested": False,
                "updated_at": time.time(),
            },
            base_config=config,
        ) or state
        config['_context_resync'] = state
    return state


def _update_context_resync_state(
    checkpoint_manager,
    translation_id,
    updates,
    *,
    base_config=None,
):
    job = checkpoint_manager.get_job(translation_id)
    if not job:
        return None
    config = copy.deepcopy(base_config if base_config is not None else job.get('config') or {})
    state = _context_resync_state_from_config(config)
    state.update(updates)
    config['_context_resync'] = state
    if not checkpoint_manager.update_job_config(translation_id, config):
        return None
    return state


def _strip_api_keys(config):
    """Remove every API key from a config dict in place (for API responses).

    Persisted checkpoints no longer hold keys (issue #213), but the in-memory
    config of a live job does — it must never be echoed back to the browser,
    since '__USE_ENV__' requests get their key resolved from .env server-side.
    """
    if isinstance(config, dict):
        for key in [k for k in config if k == 'api_key' or k.endswith('_api_key')]:
            config.pop(key, None)
    return config


def _provider_credentials_error(config):
    """Return a credential error payload, or None when the config can run."""
    provider = (config.get('llm_provider') or 'ollama').lower()
    prompt_options = config.get('prompt_options') or {}
    editor_provider = str(prompt_options.get('editor_provider') or provider).lower()
    providers = {provider: config.get('llm_api_endpoint')}
    if editor_provider != provider:
        providers[editor_provider] = prompt_options.get('editor_api_endpoint')

    for current_provider, current_endpoint in providers.items():
        if current_provider not in _KEY_PROVIDERS:
            continue
        env_var = f"{current_provider.upper()}_API_KEY"
        # 'openai' also covers OpenAI-compatible local endpoints (llama.cpp,
        # LM Studio, vLLM) where a key is legitimately absent — only require
        # one for the official API, mirroring the factory's heuristic.
        key_required = (
            current_provider != 'openai'
            or 'api.openai.com' in (current_endpoint or _config.OPENAI_API_ENDPOINT)
        )
        if key_required and not (
            config.get(f"{current_provider}_api_key") or os.getenv(env_var)
        ):
            return {
                "error": "Missing API key for provider",
                "message": (f"Resuming with '{current_provider}' requires an API key. "
                            f"Set {env_var} in .env or include it in the request."),
            }

    if provider in _ENDPOINT_PROVIDERS and not config.get('llm_api_endpoint'):
        return {
            "error": "Missing API endpoint for provider",
            "message": f"Resuming with '{provider}' requires an API endpoint.",
        }

    return None


def _validate_provider_credentials(config):
    """Return a Flask error response when a resume config cannot run."""
    error = _provider_credentials_error(config)
    if error is not None:
        return jsonify(error), 400
    return None


def _rehydrate_resume_credentials(config, overrides=None):
    """Restore non-persisted provider credentials into a resume config.

    Checkpoints deliberately exclude secrets. Every path that reconstructs a
    job from a checkpoint, including background context re-sync, must call this
    helper before starting a worker.
    """
    provider = (config.get('llm_provider') or 'ollama').lower()
    editor_provider = str(
        (config.get('prompt_options') or {}).get('editor_provider') or provider
    ).lower()
    raw_key = overrides.get('api_key') if isinstance(overrides, dict) else None
    raw_editor_key = overrides.get('editor_api_key') if isinstance(overrides, dict) else None

    active_key_providers = {provider, editor_provider} & set(_KEY_PROVIDERS)
    for key_provider in active_key_providers:
        env_var = f"{key_provider.upper()}_API_KEY"
        provider_override = (
            overrides.get(f'{key_provider}_api_key')
            if isinstance(overrides, dict) else None
        )
        key_override = provider_override or (
            raw_key
            if key_provider == provider
            and raw_key not in (None, '')
            else raw_editor_key
            if key_provider == editor_provider
            and raw_editor_key not in (None, '')
            else None
        )
        resolved = _resolve_api_key(
            key_override,
            env_var,
            getattr(_config, env_var, ''),
        )
        if resolved or f"{key_provider}_api_key" in config:
            config[f"{key_provider}_api_key"] = resolved


def _apply_resume_overrides(config, overrides):
    """Merge optional model/provider override fields into a resume config in place.

    Lets the resume request switch model/provider for the remaining chunks
    (issue #183). An empty/absent body leaves `config` untouched. API keys flow
    through `_resolve_api_key` exactly like the start endpoint, and a multi-key
    string is passed through unchanged so the key-rotation pool still works.

    Also merges prompt_options overrides (e.g. reflection_mode) so that
    resumed legacy jobs can adopt newly introduced settings from the current
    UI state.

    Credentials are validated even with an empty body: checkpoints no longer
    persist API keys (issue #213), so every resume must find its key in .env
    or in the request.

    Returns a Flask (response, status) tuple to abort with on validation failure,
    or None on success.
    """
    if isinstance(overrides, dict) and overrides:
        if overrides.get('model'):
            config['model'] = overrides['model']
        if overrides.get('llm_provider'):
            config['llm_provider'] = str(overrides['llm_provider']).lower()
        if overrides.get('llm_api_endpoint'):
            config['llm_api_endpoint'] = overrides['llm_api_endpoint']
        if overrides.get('context_window') is not None:
            try:
                config['context_window'] = int(overrides['context_window'])
            except (TypeError, ValueError):
                return jsonify({"error": "context_window must be an integer"}), 400

        # Merge prompt_options overrides so newly introduced settings apply to
        # resumed jobs.
        prompt_options_overrides = overrides.get('prompt_options')
        if isinstance(prompt_options_overrides, dict) and prompt_options_overrides:
            existing_opts = config.get('prompt_options') or {}
            existing_opts.update(prompt_options_overrides)
            config['prompt_options'] = _prompt_options_from_start_request({
                'prompt_options': existing_opts,
            })

        # Merge editor overrides into prompt_options
        if (
            overrides.get('editor_model') is not None
            or overrides.get('editor_provider') is not None
            or overrides.get('editor_api_endpoint') is not None
        ):
            if 'prompt_options' not in config or not isinstance(config['prompt_options'], dict):
                config['prompt_options'] = {}
            if overrides.get('editor_model') is not None:
                config['prompt_options']['editor_model'] = overrides['editor_model']
            if overrides.get('editor_provider') is not None:
                config['prompt_options']['editor_provider'] = str(overrides['editor_provider']).lower()
            if overrides.get('editor_api_endpoint') is not None:
                config['prompt_options']['editor_api_endpoint'] = overrides['editor_api_endpoint']

    _rehydrate_resume_credentials(config, overrides)

    return _validate_provider_credentials(config)


def _available_context_chunk_indices(checkpoint_data):
    """Return canonical checkpoint indices that contain editable snapshots."""
    indices = []
    for chunk in (checkpoint_data or {}).get('chunks', []):
        chunk_data = chunk.get('chunk_data') or {}
        index = chunk.get('chunk_index')
        if (
            isinstance(index, int)
            and chunk.get('status') in ('completed', 'partial', 'failed')
            and chunk_data.get('context_snapshot')
        ):
            indices.append(index)
    return sorted(set(indices))


def _build_corrective_refinement_config(config, output_filepath=None):
    """Build a one-pass refinement replay after context re-sync.

    The replay always starts from the preserved first-pass translation, never
    from the already-refined output. Returning ``None`` means the checkpoint is
    legacy or incomplete and cannot safely replay refinement automatically.
    """
    if not config.get("refine_after"):
        return None

    source_path = config.get("refinement_source_path")
    final_output_path = output_filepath or config.get("output_filepath")
    if not source_path or not final_output_path:
        return None
    if not Path(source_path).is_file() or not Path(final_output_path).is_file():
        return None

    correction = copy.deepcopy(config)
    correction.update({
        "file_path": str(Path(source_path).resolve()),
        "preserved_input_path": str(Path(source_path).resolve()),
        "output_filepath": str(Path(final_output_path).resolve()),
        "output_filename": Path(final_output_path).name,
        "resume_from_index": 0,
        "is_resume": True,
        "refine_only": True,
        "refine_after": False,
        "_context_resync_refinement": True,
        "_force_output_filepath": str(Path(final_output_path).resolve()),
    })
    return correction


def _active_translation_conflict(
    state_manager,
    *,
    action="resume",
    ignore_translation_id=None,
):
    active_translations = []
    for tid, tdata in state_manager.get_all_translations().items():
        status = tdata.get('status')
        if status in ['running', 'queued'] and tid != ignore_translation_id:
            active_translations.append({
                'id': tid,
                'status': status,
                'output_filename': (
                    tdata.get('config', {}).get('output_filename', 'unknown')
                ),
            })
    if not active_translations:
        return None
    action_label = {
        "continue": "start continuation",
        "retry the Senior Editor": "retry the Senior Editor",
    }.get(action, "resume")
    action_detail = {
        "continue": "adding new content",
        "retry the Senior Editor": "retrying the Senior Editor",
    }.get(action, "resuming")
    active_info = ', '.join(
        f"{item['output_filename']} ({item['status']})"
        for item in active_translations
    )
    return jsonify({
        "error": f"Cannot {action_label}: active translation in progress",
        "message": (
            "Please wait for active translation(s) to complete or interrupt "
            f"them before {action_detail}. Active: {active_info}"
        ),
        "active_translations": active_translations,
    }), 409


def _continued_output_filename(filename):
    path = Path(filename or "continued_translation.txt")
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    if not stem:
        stem = "continued_translation"
    return f"{stem} - continued{suffix}"


def create_translation_blueprint(state_manager, start_translation_job, output_dir, socketio=None):
    """
    Create and configure the translation blueprint

    Args:
        state_manager: Translation state manager instance
        start_translation_job: Function to start translation jobs
        output_dir: Base directory for file operations; uploaded source files
            live in '<output_dir>/uploads' and a client-supplied file_path must
            resolve inside it.
    """
    bp = Blueprint('translation', __name__)

    uploads_dir = Path(output_dir) / 'uploads'

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
        from src.utils.relationship_sync import apply_relationship_graph_to_context
        from src.utils.novel_context import (
            compress_dynamic_state,
            decode_context_snapshot,
            load_novel_context,
            normalize_novel_context_filename,
            resolve_novel_context_path,
            save_novel_context,
        )

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

        return resume_cb

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
            'failed_chunks': 0
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
            "progress": job_data.get('progress'),
            "stats": {
                'total_chunks': stats.get('total_chunks', 0),
                'completed_chunks': stats.get('completed_chunks', 0),
                'failed_chunks': stats.get('failed_chunks', 0),
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
        success = state_manager.delete_checkpoint(translation_id)

        if success:
            return jsonify({
                "message": "Checkpoint deleted successfully",
                "translation_id": translation_id
            }), 200
        else:
            return jsonify({"error": "Failed to delete checkpoint or checkpoint not found"}), 404

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
            from src.utils.novel_context import (
                decode_context_snapshot,
                load_novel_context,
                normalize_novel_context_filename,
                resolve_novel_context_path,
            )
            from src.config import NOVEL_CONTEXTS_DIR
            
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
        from src.core.adapters.generic_translator import resync_context_snapshots_background
        
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

        from src.core.adapters.generic_translator import resync_context_snapshots_background
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

    @bp.route('/api/translation/<translation_id>/addressing-rules', methods=['POST', 'PUT'])
    @bp.route('/<translation_id>/addressing-rules', methods=['POST', 'PUT'])
    def upsert_addressing_rule_route(translation_id):
        """Create or update one user-owned directed addressing rule."""

        data = request.get_json() or {}
        conflict, _current_revision = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        from src.utils.addressing_schema import AddressingCandidateV2
        from src.utils.context_merge_engine import ContextMergeEngine

        candidate = AddressingCandidateV2.from_dict(
            data,
            source_language=str(data.get("source_language") or ""),
            provenance="user_manual",
        )
        if not candidate or candidate.action != "upsert":
            return jsonify({"error": "A complete addressing rule is required"}), 400
        candidate.confidence = max(candidate.confidence, 0.99)
        db = state_manager.checkpoint_manager.db
        known_names = [
            str(node.get("canonical_name") or "")
            for node in db.get_relationship_nodes(translation_id)
        ]
        engine = ContextMergeEngine(db=db)
        with db.context_state_transaction():
            applied = engine.apply_delta(
                translation_id=translation_id,
                chunk_index=-1,
                delta=candidate.to_delta(),
                trigger_source="user_manual",
                target_language=str(data.get("target_language") or ""),
                known_character_names=known_names,
                active_character_names=[candidate.speaker, candidate.addressee],
                source_text="",
                source_language=str(data.get("source_language") or ""),
            )
            if not applied:
                return jsonify({
                    "error": "Addressing rule was rejected by deterministic validation"
                }), 422
            db.upsert_addressing_rule(
                translation_id,
                candidate.speaker,
                candidate.addressee,
                candidate.self_reference,
                candidate.second_person,
                vocative=candidate.vocative,
                register=candidate.register,
                social_basis=candidate.social_basis,
                scope=candidate.scope,
                contract_version=3,
                confidence=max(candidate.confidence, 0.99),
                is_locked=1 if data.get("is_locked", True) else 0,
                chunk_index=-1,
                notes=candidate.notes,
            )
            db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=-1,
                speaker_name=candidate.speaker,
                addressee_name=candidate.addressee,
                old_state=None,
                new_state={"status": "accepted", **candidate.to_dict()},
                trigger_source="user_manual",
                evidence_quote=candidate.evidence_quote,
                confidence=max(candidate.confidence, 0.99),
            )
        _export_structured_context(translation_id)
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        return jsonify({
            "message": "Addressing rule saved",
            "translation_id": translation_id,
            "context_revision": revision,
        }), 200

    @bp.route('/api/translation/<translation_id>/addressing-rules', methods=['GET'])
    @bp.route('/<translation_id>/addressing-rules', methods=['GET'])
    def get_addressing_rules_route(translation_id):
        """Get directed character addressing rules for a translation job."""
        db = state_manager.checkpoint_manager.db
        requested_status = str(request.args.get('status') or '').strip().lower()
        if requested_status not in {'active', 'quarantined', 'all'}:
            requested_status = 'all'
        active_rules = db.get_addressing_rules(translation_id, 'active')
        quarantined_rules = db.get_addressing_rules(translation_id, 'quarantined')
        rules = (
            active_rules if requested_status == 'active'
            else quarantined_rules if requested_status == 'quarantined'
            else active_rules
        )
        from src.utils.relationship_reasoning_engine import (
            relationship_support_for_addressing,
        )

        audits = db.get_context_audit_logs(translation_id, limit=500)
        rejection_by_pair = {}
        rejections = []
        for item in audits:
            state = item.get("new_state") or {}
            if state.get("status") != "rejected":
                continue
            pair = (
                str(item.get("speaker_name") or "").casefold(),
                str(item.get("addressee_name") or "").casefold(),
            )
            reason = str(state.get("reason") or "")
            rejection_by_pair.setdefault(pair, reason)
            rejections.append({
                "speaker_name": item.get("speaker_name"),
                "addressee_name": item.get("addressee_name"),
                "reason": reason,
                "chunk_index": item.get("chunk_index"),
                "confidence": item.get("confidence"),
                "timestamp": item.get("timestamp"),
            })
        for rule in rules:
            resolution = relationship_support_for_addressing(
                db,
                translation_id,
                str(rule.get("speaker_name") or ""),
                str(rule.get("addressee_name") or ""),
            )
            rule["derivation_path"] = resolution.get("path", [])
            rule["derived_hierarchy"] = resolution.get("hierarchy", "unknown")
            rule["relationship_confidence"] = resolution.get("confidence", 0.0)
            rule["rejection_reason"] = rejection_by_pair.get((
                str(rule.get("speaker_name") or "").casefold(),
                str(rule.get("addressee_name") or "").casefold(),
            ))
        return jsonify({
            "translation_id": translation_id,
            "context_revision": _context_revision(translation_id),
            "rules": rules,
            "count": len(rules),
            "active_count": len(active_rules),
            "quarantined_count": len(quarantined_rules),
            "quarantined_rules": quarantined_rules,
            "rejections": rejections,
        }), 200

    @bp.route('/<translation_id>/addressing-audit-log', methods=['GET'])
    def get_addressing_audit_log_route(translation_id):
        """Get audit log of character addressing updates for a translation job."""
        limit = request.args.get('limit', 100, type=int)
        logs = state_manager.checkpoint_manager.db.get_context_audit_logs(translation_id, limit=limit)
        return jsonify({
            "translation_id": translation_id,
            "audit_logs": logs,
            "count": len(logs)
        }), 200

    @bp.route('/api/translation/<translation_id>/addressing-resolution', methods=['GET'])
    @bp.route('/<translation_id>/addressing-resolution', methods=['GET'])
    @bp.route('/api/translation/<translation_id>/derived-seniority', methods=['GET'])
    @bp.route('/<translation_id>/derived-seniority', methods=['GET'])
    def get_addressing_resolution_route(translation_id):
        """Explain current addressing state and relationship-derived seniority."""

        speaker = str(request.args.get('speaker') or request.args.get('source') or '').strip()
        addressee = str(request.args.get('addressee') or request.args.get('target') or '').strip()
        if not speaker or not addressee:
            return jsonify({"error": "speaker/source and addressee/target are required"}), 400
        from src.utils.relationship_reasoning_engine import (
            relationship_support_for_addressing,
        )
        from src.utils.relationship_schema import normalize_relationship_name

        db = state_manager.checkpoint_manager.db
        current_rule = next((
            rule for rule in db.get_addressing_rules(translation_id)
            if str(rule.get("speaker_name") or "").casefold() == speaker.casefold()
            and str(rule.get("addressee_name") or "").casefold() == addressee.casefold()
        ), None)
        support = relationship_support_for_addressing(
            db,
            translation_id,
            speaker,
            addressee,
        )
        audit = db.get_relationship_pair_audit(
            translation_id,
            normalize_relationship_name(speaker),
            normalize_relationship_name(addressee),
        )
        return jsonify({
            "translation_id": translation_id,
            "speaker": speaker,
            "addressee": addressee,
            "addressing_rule": current_rule,
            "resolution": support,
            "evidence": audit.get("evidence", []),
            "conflicts": audit.get("conflicts", []),
            "derivations": audit.get("derivations", []),
        }), 200

    @bp.route('/<translation_id>/addressing-rules/lock', methods=['POST'])
    def set_addressing_rule_lock_route(translation_id):
        """Lock or unlock an addressing rule from being overwritten by LLM deltas."""
        data = request.get_json() or {}
        speaker_name = data.get('speaker_name')
        addressee_name = data.get('addressee_name')
        is_locked = bool(data.get('is_locked', True))

        if not speaker_name or not addressee_name:
            return jsonify({"error": "speaker_name and addressee_name are required"}), 400

        success = state_manager.checkpoint_manager.db.set_addressing_rule_lock(
            translation_id, speaker_name, addressee_name, is_locked
        )
        if success:
            return jsonify({
                "message": "Addressing rule lock updated successfully",
                "translation_id": translation_id,
                "speaker_name": speaker_name,
                "addressee_name": addressee_name,
                "is_locked": is_locked
            }), 200
        return jsonify({"error": "Failed to update addressing rule lock"}), 500

    @bp.route('/<translation_id>/addressing-rules', methods=['DELETE'])
    def delete_addressing_rule_route(translation_id):
        """Delete a directed character addressing rule for a translation job."""
        data = request.get_json() or {}
        speaker_name = data.get('speaker_name')
        addressee_name = data.get('addressee_name')

        if not speaker_name or not addressee_name:
            return jsonify({"error": "speaker_name and addressee_name are required"}), 400

        db = state_manager.checkpoint_manager.db
        existing = next(
            (
                rule for rule in db.get_addressing_rules(translation_id)
                if rule.get("speaker_name") == speaker_name
                and rule.get("addressee_name") == addressee_name
            ),
            None,
        )
        success = db.delete_addressing_rule(
            translation_id, speaker_name, addressee_name
        )
        if success:
            db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=-1,
                speaker_name=speaker_name,
                addressee_name=addressee_name,
                old_state=existing,
                new_state={"status": "deleted", "reason": "manual API delete"},
                trigger_source="rest_api:delete",
                evidence_quote="",
                confidence=1.0,
            )
            return jsonify({
                "message": "Addressing rule deleted successfully",
                "translation_id": translation_id,
                "speaker_name": speaker_name,
                "addressee_name": addressee_name,
            }), 200
        return jsonify({"error": "Addressing rule not found or could not be deleted"}), 404

    @bp.route('/api/translation/<translation_id>/relationship-edges', methods=['POST', 'PUT'])
    @bp.route('/<translation_id>/relationship-edges', methods=['POST', 'PUT'])
    def upsert_relationship_edge_route(translation_id):
        """Create or update one locked-by-default manual relationship fact."""

        data = request.get_json() or {}
        conflict, _current_revision = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        from src.utils.relationship_reasoning_engine import RelationshipReasoningEngine
        from src.utils.relationship_schema import RelationshipCandidate

        candidate = RelationshipCandidate.from_dict(
            {**data, "provenance": "user_manual", "confidence": data.get("confidence", 1.0)},
            default_provenance="user_manual",
            parser_status="rest_api",
        )
        if not candidate:
            return jsonify({"error": "A valid relationship candidate is required"}), 400
        db = state_manager.checkpoint_manager.db
        engine = RelationshipReasoningEngine(db=db)
        with db.context_state_transaction():
            decision = engine.merge_candidate(
                translation_id,
                -1,
                candidate,
                known_character_names=[candidate.source, candidate.target],
                language=str(data.get("language") or ""),
            )
            if decision.status not in {"accepted", "unchanged"}:
                return jsonify({
                    "error": decision.reason,
                    "status": decision.status,
                    "validator": decision.validator,
                }), 422
            if decision.edge_id and data.get("is_locked", True):
                db.set_relationship_edge_lock(
                    translation_id,
                    decision.edge_id,
                    True,
                )
        _export_structured_context(translation_id)
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        return jsonify({
            "message": "Relationship edge saved",
            "translation_id": translation_id,
            "edge_id": decision.edge_id,
            "status": decision.status,
            "context_revision": revision,
        }), 200

    @bp.route('/api/translation/<translation_id>/relationship-graph', methods=['GET'])
    @bp.route('/<translation_id>/relationship-graph', methods=['GET'])
    def get_relationship_graph_route(translation_id):
        """Get relationship graph nodes and edges for a translation job."""

        db = state_manager.checkpoint_manager.db
        status = request.args.get('status')
        statuses = [status] if status else None
        nodes = db.get_relationship_nodes(translation_id)
        edges = db.get_relationship_edges(translation_id, statuses=statuses)
        return jsonify({
            "translation_id": translation_id,
            "context_revision": _context_revision(translation_id),
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }), 200

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
        result["retry_states"] = retry_states
        for run in result.get("runs") or []:
            run["retry_state"] = retry_states.get(
                str(run.get("chunk_index")), {"status": "idle"}
            )
        return jsonify(result)

    @bp.route('/api/maintenance/jobs-db/compact', methods=['POST'])
    def compact_jobs_database_route():
        """Back up and compact the checkpoint database while the app is idle."""

        active_statuses = {'queued', 'running', 'rate_limited', 'resyncing', 'refining'}
        active_jobs = [
            translation_id
            for translation_id, job in state_manager.get_all_translations().items()
            if str(job.get('status') or '').casefold() in active_statuses
        ]
        with _CONTEXT_RESYNC_LOCK:
            active_resyncs = sorted(_ACTIVE_CONTEXT_RESYNCS)
        if active_jobs or active_resyncs:
            return jsonify({
                "error": "Checkpoint maintenance requires an idle application",
                "active_jobs": active_jobs,
                "active_resyncs": active_resyncs,
            }), 409
        try:
            result = state_manager.checkpoint_manager.db.optimize_database()
            return jsonify({"success": True, **result})
        except Exception as exc:
            logger.error(f"Checkpoint database optimization failed: {exc}")
            return jsonify({"error": str(exc)}), 500

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
        chunk_data["editor_validation"] = {
            "stage": "manual_retry",
            "status": "review_required",
            "final_reason_codes": ["manual_editor_retry_requested"],
        }
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

    @bp.route('/api/translation/<translation_id>/relationship-conflicts', methods=['GET'])
    @bp.route('/<translation_id>/relationship-conflicts', methods=['GET'])
    def get_relationship_conflicts_route(translation_id):
        """Get open or historical relationship validation conflicts."""

        status = request.args.get('status')
        limit = request.args.get('limit', 200, type=int)
        conflicts = (
            state_manager.checkpoint_manager.db.get_relationship_conflicts(
                translation_id,
                status=status,
                limit=max(1, min(limit, 1000)),
            )
        )
        return jsonify({
            "translation_id": translation_id,
            "conflicts": conflicts,
            "count": len(conflicts),
        }), 200

    @bp.route('/api/translation/<translation_id>/relationship-audit', methods=['GET'])
    @bp.route('/<translation_id>/relationship-audit', methods=['GET'])
    def get_relationship_pair_audit_route(translation_id):
        """Get accepted state, evidence, and conflicts for one character pair."""

        source_name = str(request.args.get('source') or '').strip()
        target_name = str(request.args.get('target') or '').strip()
        if not source_name or not target_name:
            return jsonify({"error": "source and target are required"}), 400
        from src.utils.relationship_schema import normalize_relationship_name

        audit = state_manager.checkpoint_manager.db.get_relationship_pair_audit(
            translation_id,
            normalize_relationship_name(source_name),
            normalize_relationship_name(target_name),
        )
        return jsonify({
            "translation_id": translation_id,
            "source": source_name,
            "target": target_name,
            **audit,
        }), 200

    @bp.route('/api/translation/<translation_id>/relationship-edges/<int:edge_id>/lock', methods=['POST'])
    @bp.route('/<translation_id>/relationship-edges/<int:edge_id>/lock', methods=['POST'])
    def set_relationship_edge_lock_route(translation_id, edge_id):
        """Lock or unlock a relationship edge."""

        data = request.get_json() or {}
        is_locked = bool(data.get('is_locked', True))
        success = state_manager.checkpoint_manager.db.set_relationship_edge_lock(
            translation_id,
            edge_id,
            is_locked,
        )
        if not success:
            return jsonify({"error": "Relationship edge not found"}), 404
        return jsonify({
            "message": "Relationship edge lock updated successfully",
            "translation_id": translation_id,
            "edge_id": edge_id,
            "is_locked": is_locked,
        }), 200

    @bp.route('/api/translation/<translation_id>/relationship-edges/<int:edge_id>/quarantine', methods=['POST'])
    @bp.route('/<translation_id>/relationship-edges/<int:edge_id>/quarantine', methods=['POST'])
    def quarantine_relationship_edge_route(translation_id, edge_id):
        """Quarantine an unlocked relationship edge and retain its audit state."""

        db = state_manager.checkpoint_manager.db
        existing = next((
            edge for edge in db.get_relationship_edges(translation_id)
            if edge.get("id") == edge_id
        ), None)
        if not existing:
            return jsonify({"error": "Relationship edge not found"}), 404
        if existing.get("is_locked"):
            return jsonify({"error": "Locked relationship edges cannot be quarantined"}), 409
        if not db.set_relationship_edge_status(translation_id, edge_id, "quarantined"):
            return jsonify({"error": "Relationship edge could not be quarantined"}), 409
        db.add_relationship_conflict(
            translation_id=translation_id,
            source_name=existing.get("source_name") or "",
            target_name=existing.get("target_name") or "",
            severity="warning",
            validator="manual_quarantine",
            reason="Relationship edge quarantined through the REST API.",
            remediation_hint="Review the source evidence before restoring this edge.",
            candidate=existing,
            chunk_index=-1,
            edge_id=edge_id,
        )
        return jsonify({
            "message": "Relationship edge quarantined successfully",
            "translation_id": translation_id,
            "edge_id": edge_id,
        }), 200

    @bp.route('/api/translation/<translation_id>/relationship-edges/<int:edge_id>', methods=['DELETE'])
    @bp.route('/<translation_id>/relationship-edges/<int:edge_id>', methods=['DELETE'])
    def delete_relationship_edge_route(translation_id, edge_id):
        """Delete an unlocked relationship edge and retain an audit marker."""

        db = state_manager.checkpoint_manager.db
        existing = next((
            edge for edge in db.get_relationship_edges(translation_id)
            if edge.get("id") == edge_id
        ), None)
        if not existing:
            return jsonify({"error": "Relationship edge not found"}), 404
        if existing.get("is_locked"):
            return jsonify({"error": "Locked relationship edges cannot be deleted"}), 409
        conflict_id = db.add_relationship_conflict(
            translation_id=translation_id,
            source_name=existing.get("source_name") or "",
            target_name=existing.get("target_name") or "",
            severity="info",
            validator="manual_delete",
            reason="Relationship edge deleted through the REST API.",
            remediation_hint="No action required.",
            candidate=existing,
            chunk_index=-1,
            edge_id=edge_id,
        )
        success = db.delete_relationship_edge(translation_id, edge_id)
        if not success:
            return jsonify({"error": "Relationship edge could not be deleted"}), 409
        if conflict_id:
            db.resolve_relationship_conflict(translation_id, conflict_id)
        return jsonify({
            "message": "Relationship edge deleted successfully",
            "translation_id": translation_id,
            "edge_id": edge_id,
        }), 200

    @bp.route('/api/translation/<translation_id>/recovery-preview', methods=['GET'])
    @bp.route('/<translation_id>/recovery-preview', methods=['GET'])
    def recovery_preview_route(translation_id):
        """Preview a non-destructive editor/context recovery operation."""

        if translation_id != "trans_1783673914101":
            return jsonify({"error": "No dedicated recovery recipe exists for this job"}), 404
        db = state_manager.checkpoint_manager.db
        job = db.get_job(translation_id)
        if not job:
            return jsonify({"error": "Translation job not found"}), 404
        chunks = db.get_chunks(translation_id)
        failed = [
            item.get("chunk_index") for item in chunks
            if item.get("status") == "failed"
        ]
        completed = [
            item.get("chunk_index") for item in chunks
            if item.get("status") == "completed"
        ]
        pair_edges = [
            edge for edge in db.get_relationship_edges(translation_id)
            if {
                str(edge.get("source_name") or "").casefold(),
                str(edge.get("target_name") or "").casefold(),
            } == {"frondier de roach", "enfer de roach"}
        ]
        config = job.get("config") or {}
        return jsonify({
            "translation_id": translation_id,
            "backup_required": True,
            "database_path": str(db.db_path),
            "context_file": (config.get("prompt_options") or {}).get("novel_context_file"),
            "output_path": config.get("output_filepath"),
            "completed_chunks_preserved": completed,
            "failed_chunks_reset": [index for index in failed if index in {2, 3, 4, 5}],
            "relationship_edges_quarantined": [edge.get("id") for edge in pair_edges],
            "relationship_correction": {
                "source": "Enfer De Roach",
                "target": "Frondier De Roach",
                "relationship_type": "parent",
                "hierarchy": "source_senior",
                "relative_age": "source_older",
                "rank_relation": "source_higher",
                "is_locked": True,
            },
        }), 200

    @bp.route('/api/translation/<translation_id>/recover-editor-context', methods=['POST'])
    @bp.route('/<translation_id>/recover-editor-context', methods=['POST'])
    def recover_editor_context_route(translation_id):
        """Back up and repair the explicitly selected interrupted job."""

        if translation_id != "trans_1783673914101":
            return jsonify({"error": "No dedicated recovery recipe exists for this job"}), 404
        data = request.get_json() or {}
        if data.get("confirm") is not True:
            return jsonify({"error": "Explicit recovery confirmation is required"}), 400
        db = state_manager.checkpoint_manager.db
        job = db.get_job(translation_id)
        if not job:
            return jsonify({"error": "Translation job not found"}), 404
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = Path(db.db_path).resolve().parent / "recovery_backups" / f"{translation_id}-{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        db.backup_to(str(backup_dir / "jobs.db"))
        config = job.get("config") or {}
        copied = []
        output_path = config.get("output_filepath")
        if output_path and Path(output_path).is_file():
            target = backup_dir / Path(output_path).name
            shutil.copy2(output_path, target)
            copied.append(str(target))
        context_name = (config.get("prompt_options") or {}).get("novel_context_file")
        if context_name:
            from src.config import NOVEL_CONTEXTS_DIR
            from src.utils.novel_context import resolve_novel_context_path
            context_path = resolve_novel_context_path(context_name, NOVEL_CONTEXTS_DIR)
            if context_path.is_file():
                target = backup_dir / context_path.name
                shutil.copy2(context_path, target)
                copied.append(str(target))

        from src.utils.relationship_reasoning_engine import RelationshipReasoningEngine
        from src.utils.relationship_schema import RelationshipCandidate
        with db.context_state_transaction():
            for edge in db.get_relationship_edges(translation_id):
                if {
                    str(edge.get("source_name") or "").casefold(),
                    str(edge.get("target_name") or "").casefold(),
                } == {"frondier de roach", "enfer de roach"} and not edge.get("is_locked"):
                    db.set_relationship_edge_status(translation_id, edge["id"], "quarantined")
            decision = RelationshipReasoningEngine(db=db).merge_candidate(
                translation_id,
                -1,
                RelationshipCandidate(
                    source="Enfer De Roach",
                    target="Frondier De Roach",
                    relationship_type="parent",
                    direction="directed",
                    hierarchy="source_senior",
                    relative_age="source_older",
                    rank_relation="source_higher",
                    confidence=1.0,
                    provenance="user_manual",
                    details="Enfer De Roach is Frondier De Roach's father.",
                ),
                known_character_names=["Enfer De Roach", "Frondier De Roach"],
            )
            if decision.edge_id:
                db.set_relationship_edge_lock(translation_id, decision.edge_id, True)
            for chunk in db.get_chunks(translation_id):
                if chunk.get("chunk_index") not in {2, 3, 4, 5} or chunk.get("status") != "failed":
                    continue
                chunk_data = dict(chunk.get("chunk_data") or {})
                chunk_data.pop("editor_validation", None)
                db.save_chunk(
                    translation_id=translation_id,
                    chunk_index=chunk["chunk_index"],
                    original_text=chunk.get("original_text"),
                    translated_text=None,
                    chunk_data=chunk_data,
                    status="failed",
                )
        _export_structured_context(translation_id)
        state_manager.checkpoint_manager.mark_partial(translation_id)
        revision = state_manager.checkpoint_manager.mark_refinement_stale(translation_id)
        return jsonify({
            "message": "Recovery prepared; resume the job to retry failed chunks.",
            "translation_id": translation_id,
            "backup_directory": str(backup_dir),
            "backed_up_files": copied,
            "context_revision": revision,
            "resume_from_chunk": 2,
        }), 200

    return bp
