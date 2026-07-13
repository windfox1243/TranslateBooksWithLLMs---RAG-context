"""Immediate, checkpoint-backed Senior Editor retries."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional


TERMINAL_RETRY_STATES = {
    "succeeded", "review_required", "failed", "blocked",
}


def editor_retry_state(chunk_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a public retry state, including legacy queued checkpoints."""

    data = dict(chunk_data or {})
    state = data.get("editor_retry")
    if isinstance(state, dict) and state.get("status"):
        return dict(state)
    if data.get("editor_retry_pending"):
        return {
            "status": "ready",
            "legacy_pending": True,
        }
    return {"status": "idle"}


def _custom_instructions(options: Dict[str, Any]) -> str:
    inline = str(options.get("custom_instructions") or "").strip()
    if inline:
        return inline
    filename = str(options.get("custom_instruction_file") or "").strip()
    if not filename:
        return ""
    try:
        from src.config import CUSTOM_INSTRUCTIONS_DIR
        from src.utils.custom_instructions import load_custom_instructions

        loaded = load_custom_instructions(filename, CUSTOM_INSTRUCTIONS_DIR)
        return str(loaded.get("translation") or "").strip()
    except Exception:
        return ""


def _novel_context(chunk_data: Dict[str, Any], options: Dict[str, Any]) -> str:
    snapshot = str(chunk_data.get("context_snapshot") or "")
    fallback = str(options.get("novel_context") or "")
    filename = str(options.get("novel_context_file") or "").strip()
    if not fallback and filename:
        try:
            from src.config import NOVEL_CONTEXTS_DIR
            from src.utils.novel_context import load_novel_context

            fallback = load_novel_context(filename, NOVEL_CONTEXTS_DIR)
        except Exception:
            fallback = ""
    if not snapshot:
        return fallback
    try:
        from src.utils.novel_context import decode_context_snapshot

        full_context, _global_lore, _dynamic_state = decode_context_snapshot(
            snapshot,
            fallback,
        )
        return full_context
    except Exception:
        return fallback


def _active_character_names(chunk_data: Dict[str, Any]) -> list[str]:
    attribution = chunk_data.get("dialogue_attribution") or {}
    values = []
    for turn in attribution.get("turns") or []:
        values.extend((turn.get("speaker"), turn.get("addressee")))
    state_after = attribution.get("state_after") or {}
    values.extend((state_after.get("speaker"), state_after.get("addressee")))
    result = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean.casefold() != "unknown" and clean not in result:
            result.append(clean)
    return result


def _latest_editor_run(db: Any, translation_id: str, chunk_index: int) -> Dict[str, Any]:
    diagnostics = db.get_editor_diagnostics(translation_id)
    return next((
        run for run in reversed(diagnostics.get("runs") or [])
        if int(run.get("chunk_index", -1)) == int(chunk_index)
    ), {})


def _save_retry_state(
    checkpoint_manager: Any,
    translation_id: str,
    chunk: Dict[str, Any],
    state: Dict[str, Any],
    *,
    translated_text: Optional[str] = None,
    chunk_status: Optional[str] = None,
) -> bool:
    chunk_data = dict(chunk.get("chunk_data") or {})
    chunk_data["editor_retry"] = dict(state)
    if state.get("status") in TERMINAL_RETRY_STATES:
        chunk_data.pop("editor_retry_pending", None)
    else:
        chunk_data["editor_retry_pending"] = True
    return checkpoint_manager.save_checkpoint(
        translation_id=translation_id,
        chunk_index=int(chunk["chunk_index"]),
        original_text=chunk.get("original_text") or "",
        translated_text=(
            translated_text
            if translated_text is not None
            else chunk.get("translated_text")
        ),
        chunk_data=chunk_data,
        chunk_status=chunk_status or chunk.get("status") or "editor_retry",
    )


async def _refresh_output(
    translation_id: str,
    checkpoint_manager: Any,
    output_dir: Path,
    output_filename: str,
    bilingual: bool,
) -> Dict[str, Any]:
    from src.core.adapters import build_translated_output

    safe_name = Path(output_filename).name
    if not safe_name:
        return {"status": "pending", "reason": "missing_output_filename"}
    output_path = output_dir / safe_name
    output_bytes, error = await build_translated_output(
        translation_id,
        checkpoint_manager,
        output_file_path=str(output_path),
        bilingual_output=bool(bilingual),
    )
    if output_bytes is None:
        return {"status": "pending", "reason": str(error or "rebuild_failed")}
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=output_dir,
            prefix=f".{safe_name}.", suffix=".tmp",
        ) as handle:
            handle.write(output_bytes)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
    return {"status": "updated", "filename": safe_name}


async def run_editor_retry(
    *,
    translation_id: str,
    chunk_index: int,
    checkpoint_manager: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one Senior Editor pass without resuming the translation job."""

    checkpoint = checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint:
        raise ValueError("Translation job not found")
    chunk = next((
        item for item in checkpoint.get("chunks", [])
        if int(item.get("chunk_index", -1)) == int(chunk_index)
    ), None)
    if not chunk or not str(chunk.get("translated_text") or "").strip():
        raise ValueError("No preserved draft is available")

    job = checkpoint.get("job") or {}
    config = dict(job.get("config") or {})
    options = dict(config.get("prompt_options") or {})
    draft_provider = str(config.get("llm_provider") or "ollama").casefold()
    draft_model = str(config.get("model") or "").strip()
    editor_provider = str(options.get("editor_provider") or draft_provider).casefold()
    editor_model = str(options.get("editor_model") or draft_model).strip()
    editor_endpoint = options.get("editor_api_endpoint")
    if not editor_endpoint and editor_provider == draft_provider:
        editor_endpoint = config.get("llm_api_endpoint")

    from src.core.llm.runtime import build_runtime_spec, create_runtime_client

    credentials = {
        key: value for key, value in config.items()
        if key.endswith("_api_key") and value
    }
    editor_spec = build_runtime_spec(
        editor_provider,
        editor_model,
        api_endpoint=editor_endpoint,
        credentials=credentials,
    )
    if editor_spec.key_required and not editor_spec.api_key:
        raise RuntimeError(
            f"Senior Editor provider {editor_spec.provider} requires an API key"
        )
    editor_client = create_runtime_client(
        editor_spec,
        context_window=config.get("context_window"),
    )

    source_text = str(chunk.get("original_text") or "")
    draft_text = str(chunk.get("translated_text") or "")
    chunk_data = dict(chunk.get("chunk_data") or {})
    options.update({
        "_editor_llm_client": editor_client,
        "editor_provider_resolved": editor_spec.provider,
        "editor_model_resolved": editor_spec.model,
        "llm_provider": draft_provider,
        "model": draft_model,
        "translation_id": translation_id,
        "jobs_db_path": getattr(checkpoint_manager.db, "db_path", None),
        "chunk_index": int(chunk_index),
        "file_type": job.get("file_type") or config.get("file_type") or "",
        "editor_phase": "manual_retry",
        "source_language": config.get("source_language") or "",
        "target_language": config.get("target_language") or "",
        "active_character_names": _active_character_names(chunk_data),
    })

    from src.core.translator import (
        ReflectionValidationError,
        _build_chunk_glossary_block,
        run_chunk_reflection_pass,
    )

    glossary_block = _build_chunk_glossary_block(source_text, options)
    try:
        result = await run_chunk_reflection_pass(
            source_chunk=source_text,
            draft_translation=draft_text,
            target_language=str(config.get("target_language") or ""),
            model_name=draft_model,
            llm_client=editor_client,
            novel_context=_novel_context(chunk_data, options),
            custom_instructions=_custom_instructions(options),
            glossary_block=glossary_block,
            prompt_options=options,
        )
    except ReflectionValidationError as exc:
        latest = _latest_editor_run(checkpoint_manager.db, translation_id, chunk_index)
        state = {
            "status": "blocked",
            "outcome": latest.get("outcome") or "blocked",
            "message": str(exc),
            "completed_at": int(time.time()),
        }
        _save_retry_state(
            checkpoint_manager, translation_id, chunk, state,
            translated_text=exc.draft_translation or draft_text,
            chunk_status="failed",
        )
        return state

    latest = _latest_editor_run(checkpoint_manager.db, translation_id, chunk_index)
    outcome = str(latest.get("outcome") or "review_required")
    if outcome in {"no_issues", "warnings_only", "locally_repaired", "llm_repaired"}:
        status = "succeeded"
    elif outcome == "review_required":
        status = "review_required"
    else:
        status = "failed"
    state = {
        "status": status,
        "outcome": outcome,
        "editor_run_id": latest.get("id"),
        "output_sync": {"status": "pending"},
        "completed_at": int(time.time()),
    }
    final_chunk_status = (
        "completed"
        if status in {"succeeded", "review_required"}
        else chunk.get("status") or "failed"
    )
    _save_retry_state(
        checkpoint_manager,
        translation_id,
        chunk,
        state,
        translated_text=str(result or draft_text),
        chunk_status=final_chunk_status,
    )
    if status in {"succeeded", "review_required"}:
        state["output_sync"] = await _refresh_output(
            translation_id,
            checkpoint_manager,
            Path(output_dir),
            str(config.get("output_filename") or ""),
            bool(config.get("bilingual_output")),
        )
    else:
        state["output_sync"] = {
            "status": "unchanged",
            "reason": "editor_retry_failed",
        }
    _save_retry_state(
        checkpoint_manager,
        translation_id,
        {
            **chunk,
            "translated_text": str(result or draft_text),
            "status": final_chunk_status,
        },
        state,
        translated_text=str(result or draft_text),
        chunk_status=final_chunk_status,
    )
    return state
