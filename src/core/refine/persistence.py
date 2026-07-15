"""Phase-aware persistence helpers shared by refinement adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional


def save_refinement_unit(
    prompt_options: Optional[Dict[str, Any]],
    *,
    unit_index: int,
    source_text: str,
    refined_text: str,
    status: str = "completed",
    quality_status: str = "passed",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Persist one unit when the caller belongs to a managed refinement pass."""
    options = prompt_options or {}
    pass_id = str(options.get("_refinement_pass_id") or "")
    translation_id = str(options.get("translation_id") or "")
    db = options.get("_checkpoint_db")
    if not pass_id or not translation_id or db is None:
        return False
    return bool(db.save_refinement_chunk_result(
        pass_id,
        translation_id,
        int(unit_index),
        base_chunk_index=int(unit_index),
        source_text=str(source_text or ""),
        refined_text=str(refined_text or ""),
        chunk_data=dict(metadata or {}),
        status=status,
        quality_status=quality_status,
    ))
