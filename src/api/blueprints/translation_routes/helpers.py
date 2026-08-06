"""Shared state and helpers for the translation blueprint modules.

These were module-level in translation_routes.py before it became a
package; the process-wide claim registries below are the reason they
must stay in exactly one module.
"""
import os
import asyncio
import time
import copy
import threading
import shutil
import uuid
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
    prompt_options.setdefault('context_contract_version', 5)
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
    try:
        prompt_options['auto_review_repair_threshold'] = max(
            0,
            min(int(prompt_options.get('auto_review_repair_threshold', 3)), 20),
        )
    except (TypeError, ValueError):
        prompt_options['auto_review_repair_threshold'] = 3
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
_ACTIVE_EDITOR_BATCHES = set()


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
        if key[0] in _ACTIVE_EDITOR_BATCHES:
            return False
        if any(item[0] == key[0] for item in _ACTIVE_EDITOR_RETRIES):
            return False
        _ACTIVE_EDITOR_RETRIES.add(key)
        return True


def _release_editor_retry(translation_id, chunk_index):
    with _EDITOR_RETRY_LOCK:
        _ACTIVE_EDITOR_RETRIES.discard((str(translation_id), int(chunk_index)))


def _claim_editor_batch(translation_id):
    key = str(translation_id)
    with _EDITOR_RETRY_LOCK:
        if key in _ACTIVE_EDITOR_BATCHES:
            return False
        if any(item[0] == key for item in _ACTIVE_EDITOR_RETRIES):
            return False
        _ACTIVE_EDITOR_BATCHES.add(key)
        return True


def _release_editor_batch(translation_id):
    with _EDITOR_RETRY_LOCK:
        _ACTIVE_EDITOR_BATCHES.discard(str(translation_id))


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
