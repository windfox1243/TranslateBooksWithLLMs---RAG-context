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


def _active_refinement_results(db: Any, translation_id: str) -> list[Dict[str, Any]]:
    getter = getattr(db, "get_active_refinement_results", None)
    return list(getter(translation_id) or []) if callable(getter) else []


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


def _unresolved_issue_seed(chunk_data: Dict[str, Any]) -> tuple[list[Dict[str, Any]], list[str]]:
    """Load the previous bounded local-edit contract for a focused retry."""

    validation = chunk_data.get("editor_validation")
    if not isinstance(validation, dict):
        return [], []
    issues = [
        dict(item) for item in list(validation.get("unresolved_issues") or [])[:12]
        if isinstance(item, dict)
        and str(item.get("repair_kind") or "").casefold() == "local_replace"
    ]
    reasons = [
        str(item) for item in list(validation.get("final_reason_codes") or [])[:12]
        if item
    ]
    return issues, reasons


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
    if state.get("status") == "succeeded":
        chunk_data["quality_status"] = "passed"
    elif state.get("status") in {"review_required", "blocked", "failed"}:
        chunk_data["quality_status"] = "review_required"
    chunk_data.pop("execution_failure_class", None)
    if state.get("status") in TERMINAL_RETRY_STATES:
        chunk_data.pop("editor_retry_pending", None)
        if state.get("status") == "succeeded":
            chunk_data.pop("narrator_voice_stale", None)
            narrator_backfill = dict(chunk_data.get("narrator_backfill") or {})
            if narrator_backfill:
                narrator_backfill["status"] = "succeeded"
                narrator_backfill["attempts"] = (
                    int(narrator_backfill.get("attempts") or 0) + 1
                )
                narrator_backfill.pop("last_failure", None)
                chunk_data["narrator_backfill"] = narrator_backfill
    else:
        chunk_data["editor_retry_pending"] = True
    refinement_pass_id = str(chunk_data.get("refinement_pass_id") or "")
    if (
        chunk_data.get("effective_phase") == "refinement"
        and refinement_pass_id
    ):
        return checkpoint_manager.db.save_refinement_chunk_result(
            refinement_pass_id,
            translation_id,
            int(chunk["chunk_index"]),
            base_chunk_index=int(chunk["chunk_index"]),
            source_text=str(chunk.get("original_text") or ""),
            refined_text=(
                translated_text if translated_text is not None
                else str(chunk.get("translated_text") or "")
            ),
            chunk_data=chunk_data,
            status=(
                "completed" if chunk_status in {None, "completed", "editor_retry"}
                else str(chunk_status)
            ),
            quality_status=str(chunk_data.get("quality_status") or "not_checked"),
        )
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
    refresh_output: bool = True,
    phase: str = "effective",
) -> Dict[str, Any]:
    """Run one Senior Editor pass without resuming the translation job."""

    checkpoint = checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint:
        raise ValueError("Translation job not found")
    chunk = next((
        item for item in checkpoint.get("chunks", [])
        if int(item.get("chunk_index", -1)) == int(chunk_index)
    ), None)
    active_refinement = {
        int(item.get("base_chunk_index", -1)): item
        for item in _active_refinement_results(checkpoint_manager.db, translation_id)
        if item.get("base_chunk_index") is not None
    }.get(int(chunk_index)) if str(phase).casefold() != "translation" else None
    if str(phase).casefold() == "refinement" and not active_refinement:
        raise ValueError(
            "No mapped refinement draft is available; replay refinement first"
        )
    if chunk and active_refinement:
        chunk = {
            **chunk,
            "translated_text": active_refinement.get("refined_text"),
            "chunk_data": {
                **(chunk.get("chunk_data") or {}),
                **(active_refinement.get("chunk_data") or {}),
                "effective_phase": "refinement",
                "refinement_pass_id": active_refinement.get("pass_id"),
                "quality_status": active_refinement.get("quality_status") or "not_checked",
            },
        }
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

    from src.utils.narrator_voice import bootstrap_narrator_voice
    await bootstrap_narrator_voice(
        db=checkpoint_manager.db,
        translation_id=translation_id,
        chunks=checkpoint.get("chunks", []),
        target_language=str(config.get("target_language") or ""),
        model_name=editor_spec.model,
        llm_client=editor_client,
        file_type=str(job.get("file_type") or config.get("file_type") or ""),
    )

    source_text = str(chunk.get("original_text") or "")
    draft_text = str(chunk.get("translated_text") or "")
    chunk_data = dict(chunk.get("chunk_data") or {})
    retry_issue_seed, retry_reason_codes = _unresolved_issue_seed(chunk_data)
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
        "editor_phase": "refinement" if active_refinement else "manual_retry",
        "source_language": config.get("source_language") or "",
        "target_language": config.get("target_language") or "",
        "active_character_names": _active_character_names(chunk_data),
        "dialogue_attribution": chunk_data.get("dialogue_attribution") or {},
        "editor_retry_unresolved_issues": retry_issue_seed,
        "editor_retry_reason_codes": retry_reason_codes,
        "editor_retry_source_run_id": (
            (chunk_data.get("editor_retry") or {}).get("source_run_id")
            if isinstance(chunk_data.get("editor_retry"), dict) else None
        ),
        "_refinement_pass_id": (
            active_refinement.get("pass_id") if active_refinement else None
        ),
    })
    from src.utils.narrator_voice import build_narrator_voice_context

    options["_checkpoint_db"] = checkpoint_manager.db
    options["chapter_index"] = chunk_data.get("chapter_index")
    options["scene_key"] = str(chunk_data.get("scene_key") or "")
    narrative_voice_context = build_narrator_voice_context(
        translation_id, checkpoint_manager.db,
        chunk_index=int(chunk_index),
        target_language=str(config.get("target_language") or ""),
    )
    if narrative_voice_context:
        options["narrative_voice_context"] = narrative_voice_context
    from src.core.context.unit_pipeline import prepare_unit_prompt_options
    options = prepare_unit_prompt_options(
        options,
        unit_index=int(chunk_index),
        phase=("refinement" if active_refinement else "manual_retry"),
        file_type=str(job.get("file_type") or config.get("file_type") or ""),
        source_text=source_text,
        target_language=str(config.get("target_language") or ""),
        dialogue_attribution=chunk_data.get("dialogue_attribution") or {},
        chapter_index=chunk_data.get("chapter_index"),
        scene_key=str(chunk_data.get("scene_key") or ""),
    )

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
            chunk_status="completed",
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
    result_validation = getattr(result, "editor_validation", None)
    if isinstance(result_validation, dict) and result_validation:
        chunk_data["editor_validation"] = dict(result_validation)
    retry_chunk = {**chunk, "chunk_data": chunk_data}
    final_chunk_status = (
        "completed"
        if status in {"succeeded", "review_required"}
        else chunk.get("status") or "failed"
    )
    _save_retry_state(
        checkpoint_manager,
        translation_id,
        retry_chunk,
        state,
        translated_text=str(result or draft_text),
        chunk_status=final_chunk_status,
    )
    if status in {"succeeded", "review_required"} and refresh_output:
        state["output_sync"] = await _refresh_output(
            translation_id,
            checkpoint_manager,
            Path(output_dir),
            str(config.get("output_filename") or ""),
            bool(config.get("bilingual_output")),
        )
    elif status not in {"succeeded", "review_required"}:
        state["output_sync"] = {
            "status": "unchanged",
            "reason": "editor_retry_failed",
        }
    _save_retry_state(
        checkpoint_manager,
        translation_id,
        {
            **retry_chunk,
            "translated_text": str(result or draft_text),
            "status": final_chunk_status,
        },
        state,
        translated_text=str(result or draft_text),
        chunk_status=final_chunk_status,
    )
    return state


def audit_completed_narrator_conformance(
    *, translation_id: str, checkpoint_manager: Any,
) -> Dict[str, Any]:
    """Queue only completed units that fail the current deterministic policy."""

    from src.core.editor import (
        audit_narrator_conformance,
        conformance_fingerprint,
    )

    checkpoint = checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint:
        return {"audited": 0, "queued": [], "blocked": []}
    job = checkpoint.get("job") or {}
    config = dict(job.get("config") or {})
    options = dict(config.get("prompt_options") or {})
    queued: list[int] = []
    blocked: list[int] = []
    audited = 0
    refinement_overlays = {
        int(item.get("base_chunk_index", -1)): item
        for item in _active_refinement_results(checkpoint_manager.db, translation_id)
        if item.get("base_chunk_index") is not None
    }
    for chunk in checkpoint.get("chunks") or []:
        if chunk.get("status") not in {"completed", "partial"}:
            continue
        source_text = str(chunk.get("original_text") or "")
        index = int(chunk.get("chunk_index") or 0)
        overlay = refinement_overlays.get(index)
        target_text = str(
            (overlay or {}).get("refined_text") or chunk.get("translated_text") or ""
        )
        if not target_text.strip():
            continue
        chunk_data = {
            **(chunk.get("chunk_data") or {}),
            **((overlay or {}).get("chunk_data") or {}),
        }
        if overlay:
            chunk_data.update({
                "effective_phase": "refinement",
                "refinement_pass_id": overlay.get("pass_id"),
            })
        audit = audit_narrator_conformance(
            source_text=source_text,
            target_text=target_text,
            source_language=str(config.get("source_language") or ""),
            target_language=str(config.get("target_language") or ""),
            file_type=str(job.get("file_type") or config.get("file_type") or ""),
            dialogue_attribution=chunk_data.get("dialogue_attribution") or {},
            db=checkpoint_manager.db,
            translation_id=translation_id,
            chunk_index=index,
            explicit_override=str(
                options.get("narrator_self_reference_override")
                or options.get("narrator_self_reference")
                or ""
            ),
        )
        fingerprint = conformance_fingerprint(
            source_text=source_text, target_text=target_text, policy=audit,
        )
        previous = dict(chunk_data.get("narrator_backfill") or {})
        same_blocked = (
            previous.get("status") == "blocked"
            and previous.get("fingerprint") == fingerprint
        )
        chunk_data["narrator_conformance"] = audit
        chunk_data["narrator_conformance_fingerprint"] = fingerprint
        if audit.get("status") == "fail" and not same_blocked:
            chunk_data["narrator_voice_stale"] = True
            chunk_data["narrator_backfill"] = {
                "status": "queued", "fingerprint": fingerprint,
                "attempts": int(previous.get("attempts") or 0),
                "reason_codes": audit.get("reason_codes") or [],
            }
            queued.append(index)
        elif same_blocked:
            chunk_data["narrator_voice_stale"] = True
            blocked.append(index)
        else:
            chunk_data.pop("narrator_voice_stale", None)
            chunk_data["narrator_backfill"] = {
                "status": "not_required", "fingerprint": fingerprint,
                "attempts": int(previous.get("attempts") or 0),
            }
        if overlay:
            checkpoint_manager.db.save_refinement_chunk_result(
                str(overlay.get("pass_id")), translation_id, index,
                base_chunk_index=index, source_text=source_text,
                refined_text=target_text, chunk_data=chunk_data,
                status=str(overlay.get("status") or "completed"),
                quality_status=str(overlay.get("quality_status") or "not_checked"),
            )
        else:
            checkpoint_manager.save_checkpoint(
                translation_id=translation_id,
                chunk_index=index,
                original_text=source_text,
                translated_text=target_text,
                chunk_data=chunk_data,
                chunk_status=chunk.get("status") or "completed",
            )
        audited += 1
    return {"audited": audited, "queued": queued, "blocked": blocked}


async def run_pending_narrator_backfill(
    *, translation_id: str, checkpoint_manager: Any, output_dir: Path,
    log_callback: Any = None,
) -> Dict[str, Any]:
    """Run queued narrator repairs once per stale completed unit."""

    audit_summary = audit_completed_narrator_conformance(
        translation_id=translation_id, checkpoint_manager=checkpoint_manager,
    )
    checkpoint = checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint:
        return {"status": "idle", "completed": [], "failed": []}
    existing_blocked = [
        {
            "chunk_index": int(item.get("chunk_index", 0)),
            "status": "blocked",
        }
        for item in checkpoint.get("chunks", [])
        if ((item.get("chunk_data") or {}).get("narrator_backfill") or {}).get(
            "status"
        ) == "blocked"
    ]
    queued_indices = {int(value) for value in audit_summary.get("queued") or []}
    stale = [
        item for item in checkpoint.get("chunks", [])
        if item.get("status") in {"completed", "partial"}
        and int(item.get("chunk_index", -1)) in queued_indices
        and ((item.get("chunk_data") or {}).get("narrator_backfill") or {}).get(
            "status"
        ) != "blocked"
    ]
    if not stale:
        return {
            "status": "review_required" if existing_blocked else "idle",
            "completed": [], "failed": existing_blocked,
            "audit": audit_summary,
            "output_sync": {"status": "unchanged"},
        }
    completed = []
    failed = list(existing_blocked)
    for chunk in sorted(stale, key=lambda item: int(item.get("chunk_index", 0))):
        chunk_index = int(chunk.get("chunk_index", 0))
        if log_callback:
            log_callback(
                "narrator_backfill_started",
                f"Applying narrator voice to completed unit {chunk_index + 1}.",
            )
        try:
            result = await run_editor_retry(
                translation_id=translation_id,
                chunk_index=chunk_index,
                checkpoint_manager=checkpoint_manager,
                output_dir=Path(output_dir),
                refresh_output=False,
            )
            if result.get("status") == "succeeded":
                completed.append(chunk_index)
            else:
                failed.append({
                    "chunk_index": chunk_index,
                    "status": result.get("status") or "failed",
                })
        except Exception as exc:
            chunk_data = dict(chunk.get("chunk_data") or {})
            backfill_state = dict(chunk_data.get("narrator_backfill") or {})
            backfill_state.update({
                "status": "blocked",
                "attempts": int(backfill_state.get("attempts") or 0) + 1,
                "last_failure": type(exc).__name__,
            })
            chunk_data["narrator_backfill"] = backfill_state
            chunk_data["narrator_voice_stale"] = True
            chunk_data["quality_status"] = "review_required"
            chunk_data.pop("execution_failure_class", None)
            checkpoint_manager.save_checkpoint(
                translation_id=translation_id,
                chunk_index=chunk_index,
                original_text=chunk.get("original_text") or "",
                translated_text=chunk.get("translated_text") or "",
                chunk_data=chunk_data,
                chunk_status="completed",
            )
            failed.append({
                "chunk_index": chunk_index,
                "status": "failed",
                "error": type(exc).__name__,
            })
    output_sync = {"status": "unchanged"}
    if completed and not failed:
        refreshed = checkpoint_manager.load_checkpoint(translation_id) or {}
        refreshed_config = dict((refreshed.get("job") or {}).get("config") or {})
        output_sync = await _refresh_output(
            translation_id, checkpoint_manager, Path(output_dir),
            str(refreshed_config.get("output_filename") or ""),
            bool(refreshed_config.get("bilingual_output")),
        )
    return {
        "status": "completed" if not failed else "review_required",
        "completed": completed,
        "failed": failed,
        "audit": audit_summary,
        "output_sync": output_sync,
    }
