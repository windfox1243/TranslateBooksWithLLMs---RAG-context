"""Helpers for continuing a novel translation from a previous checkpoint."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _normalized_unit_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def matching_prefix_length(
    previous_chunks: Iterable[Dict[str, Any]],
    new_source_units: Sequence[str],
    *,
    offset: int = 0,
) -> int:
    """Return the contiguous translated prefix reusable for new source units."""
    by_index = {
        chunk.get("chunk_index"): chunk
        for chunk in previous_chunks or []
        if chunk.get("status") == "completed"
        and chunk.get("translated_text") is not None
        and isinstance(chunk.get("chunk_index"), int)
    }
    matched = 0
    for local_index, source_text in enumerate(new_source_units):
        previous = by_index.get(offset + local_index)
        if not previous:
            break
        if _normalized_unit_text(previous.get("original_text")) != _normalized_unit_text(source_text):
            break
        matched += 1
    return matched


def latest_context_seed(
    previous_chunks: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the latest context-bearing row from a previous checkpoint."""
    candidates = []
    for chunk in previous_chunks or []:
        chunk_data = chunk.get("chunk_data") or {}
        if (
            chunk.get("status") in ("completed", "partial", "failed")
            and isinstance(chunk.get("chunk_index"), int)
            and chunk_data.get("context_snapshot")
        ):
            candidates.append(chunk)
    if not candidates:
        return None

    latest = max(candidates, key=lambda chunk: chunk.get("chunk_index", -1))
    chunk_data = latest.get("chunk_data") or {}
    dialogue = chunk_data.get("dialogue_attribution") or {}
    return {
        "chunk_index": latest.get("chunk_index"),
        "context_snapshot": chunk_data.get("context_snapshot"),
        "dialogue_state": dialogue.get("state_after"),
        "dialogue_scene_key": dialogue.get("scene_key"),
        "dialogue_attribution": dialogue,
    }


def seed_matching_prefix(
    *,
    checkpoint_manager: Any,
    translation_id: str,
    previous_chunks: Iterable[Dict[str, Any]],
    new_source_units: Sequence[str],
    total_units: int,
    offset: int = 0,
    log_callback: Optional[Callable[[str, str], None]] = None,
    label: str = "unit",
) -> int:
    """Copy matching completed checkpoint rows into a new continuation job."""
    if not (
        checkpoint_manager
        and translation_id
        and hasattr(checkpoint_manager, "db")
    ):
        return 0

    existing = checkpoint_manager.db.get_chunks(translation_id) or []
    if any(
        chunk.get("status") == "completed"
        and isinstance(chunk.get("chunk_index"), int)
        and offset <= chunk.get("chunk_index") < offset + len(new_source_units)
        for chunk in existing
    ):
        return 0

    previous_by_index = {
        chunk.get("chunk_index"): chunk
        for chunk in previous_chunks or []
        if chunk.get("status") == "completed"
        and chunk.get("translated_text") is not None
        and isinstance(chunk.get("chunk_index"), int)
    }
    prefix = matching_prefix_length(
        previous_by_index.values(),
        new_source_units,
        offset=offset,
    )
    if prefix <= 0:
        if log_callback:
            log_callback(
                "continuation_prefix_none",
                f"Add New Content: no reusable translated {label}s found at this position.",
            )
        return 0

    for local_index in range(prefix):
        global_index = offset + local_index
        previous = previous_by_index[global_index]
        checkpoint_manager.db.save_chunk(
            translation_id=translation_id,
            chunk_index=global_index,
            original_text=new_source_units[local_index],
            translated_text=previous.get("translated_text"),
            chunk_data=dict(previous.get("chunk_data") or {}),
            status="completed",
        )

    checkpoint_manager.db.update_job_progress(
        translation_id,
        current_chunk_index=offset + prefix - 1,
        total_chunks=total_units,
        completed_chunks=offset + prefix,
        failed_chunks=0,
    )
    if log_callback:
        log_callback(
            "continuation_prefix_reused",
            f"Add New Content: reused {prefix} already translated {label}(s); "
            f"translation starts at {label} {offset + prefix + 1}.",
        )
    return prefix
