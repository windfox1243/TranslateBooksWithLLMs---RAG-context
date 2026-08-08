"""Translation job management routes.

create_translation_blueprint was a single 2,879-line factory registering
60 routes. The routes now live in one module per domain, each exposing
register(bp, deps, shared); every registration targets the same Blueprint
object, so rules, endpoint names and method sets are unchanged.
"""
from pathlib import Path

from flask import Blueprint

from . import (
    addressing,
    context,
    editor,
    lifecycle,
    maintenance,
    narrator,
    recovery,
    relationships,
)
from .deps import TranslationRouteDeps

# Re-exported unchanged: every name the flat module bound at module level.
# Callers import helpers from here, and tests reach _config and threading
# through this module to patch them.
from .helpers import (
    _ACTIVE_CONTEXT_RESYNCS,
    _ACTIVE_EDITOR_BATCHES,
    _ACTIVE_EDITOR_RETRIES,
    _CONTEXT_RESYNC_LOCK,
    _EDITOR_RETRY_LOCK,
    _ENDPOINT_PROVIDERS,
    _KEY_PROVIDERS,
    AUTO_PAUSE_ON_RATE_LIMIT,
    MAX_PARALLEL_TRANSLATIONS,
    MIN_CHUNK_SIZE,
    OLLAMA_NUM_CTX,
    REQUEST_TIMEOUT,
    Blueprint,
    Path,
    PathValidator,
    TTSConfig,
    _active_translation_conflict,
    _apply_resume_overrides,
    _available_context_chunk_indices,
    _build_corrective_refinement_config,
    _claim_context_resync,
    _claim_editor_batch,
    _claim_editor_retry,
    _clamp_chunk_tokens,
    _clamp_parallel_workers,
    _config,
    _context_resync_state_from_config,
    _continued_output_filename,
    _is_context_resync_active,
    _prompt_options_from_start_request,
    _provider_credentials_error,
    _refresh_inactive_context_resync_state,
    _rehydrate_resume_credentials,
    _release_context_resync,
    _release_editor_batch,
    _release_editor_retry,
    _resolve_api_key,
    _strip_api_keys,
    _unfinished_context_resync_state,
    _update_context_resync_state,
    _validate_provider_credentials,
    asyncio,
    copy,
    emit_update,
    get_logger,
    jsonify,
    logger,
    os,
    provider_env_var,
    request,
    shutil,
    threading,
    time,
    uuid,
)
from .shared import build_shared


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
    reconcile_operations = getattr(
        state_manager.checkpoint_manager.db,
        "reconcile_interrupted_operations",
        None,
    )
    if callable(reconcile_operations):
        reconcile_operations()
    bp = Blueprint('translation', __name__)

    deps = TranslationRouteDeps(
        state_manager=state_manager,
        start_translation_job=start_translation_job,
        output_dir=output_dir,
        socketio=socketio,
        uploads_dir=Path(output_dir) / 'uploads',
    )
    shared = build_shared(deps)

    lifecycle.register(bp, deps, shared)
    context.register(bp, deps, shared)
    addressing.register(bp, deps, shared)
    relationships.register(bp, deps, shared)
    editor.register(bp, deps, shared)
    narrator.register(bp, deps, shared)
    maintenance.register(bp, deps, shared)
    recovery.register(bp, deps, shared)

    return bp
