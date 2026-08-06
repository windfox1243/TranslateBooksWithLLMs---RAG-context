"""Context snapshot decoding and mapping for the refinement pass."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable

from .constants import DYNAMIC_STATE_START
from .document import (
    decompress_dynamic_state,
    extract_dynamic_state_from_text,
    extract_global_lore,
)
from .merge import build_novel_context
from .storage import (
    _safe_context_filename_stem,
    is_safe_filename,
)

def make_novel_context_filename(input_filename: str, fallback: str = "translation") -> str:
    """Create a safe, deterministic context filename from an input filename."""
    basename = str(input_filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(basename).stem
    safe_stem = _safe_context_filename_stem(stem)
    if not safe_stem:
        fallback_basename = str(fallback or "").replace("\\", "/").rsplit("/", 1)[-1]
        safe_stem = _safe_context_filename_stem(Path(fallback_basename).stem) or "translation"
    return f"{safe_stem}_context.txt"
def normalize_novel_context_filename(filename: str) -> str:
    """Return a safe context file reference.

    Web-managed contexts normally use a basename from ``Novel_Contexts``.
    Explicit absolute ``.txt`` paths are preserved so advanced users can keep
    context files on another drive.
    """
    raw = str(filename or "").strip()
    basename = raw.replace("\\", "/").rsplit("/", 1)[-1]
    if not is_safe_filename(basename):
        raise ValueError("Invalid novel context filename")
    if os.path.isabs(raw) and Path(raw).is_file():
        return str(Path(raw).resolve())
    return basename
def decode_context_snapshot(
    compressed_snapshot: Optional[str],
    fallback_context: str = "",
    canonicalize_full_snapshot: bool = True,
) -> Tuple[str, str, str]:
    """Decode full snapshots and legacy dynamic-only snapshots.

    Returns ``(full_context, global_lore, dynamic_state)``. New snapshots always
    contain the full canonical context, while old snapshots are combined with
    the supplied fallback global lore.
    """
    decoded = decompress_dynamic_state(compressed_snapshot) if compressed_snapshot else ""

    if DYNAMIC_STATE_START in decoded:
        global_lore = extract_global_lore(decoded)
        dynamic_state = extract_dynamic_state_from_text(decoded) or ""
        if not canonicalize_full_snapshot:
            return decoded.strip(), global_lore, dynamic_state
    else:
        fallback_global = extract_global_lore(fallback_context)
        fallback_dynamic = extract_dynamic_state_from_text(fallback_context) or ""
        global_lore = fallback_global
        dynamic_state = decoded or fallback_dynamic

    full_context = build_novel_context(global_lore, dynamic_state)
    canonical_global = extract_global_lore(full_context)
    canonical_dynamic = extract_dynamic_state_from_text(full_context) or ""
    return full_context, canonical_global, canonical_dynamic
def normalize_refinement_context(
    context_content: Optional[str],
    fallback_context: str = "",
) -> str:
    """Overlay final global lore onto a unit's historical dynamic state.

    Characters, proven genders, and glossary terminology are book-wide facts
    that later refinement units may discover after early units were translated.
    Addressing forms and relationship evolution are time-sensitive, so those
    remain sourced from the mapped historical snapshot.

    ``context_content`` may be a full snapshot or a legacy dynamic-only value.
    ``fallback_context`` should be the latest canonical context loaded from the
    context file; when it has no global lore, historical lore is used as a
    compatibility fallback.
    """
    final_global_lore = extract_global_lore(fallback_context)
    if not context_content:
        return build_novel_context(
            final_global_lore,
            extract_dynamic_state_from_text(fallback_context) or "",
        )
    if DYNAMIC_STATE_START in context_content:
        historical_global_lore = extract_global_lore(context_content)
        return build_novel_context(
            final_global_lore or historical_global_lore,
            extract_dynamic_state_from_text(context_content) or "",
        )
    return build_novel_context(
        final_global_lore,
        context_content,
    )
def map_context_snapshots_for_refinement(
    total_chunks: int,
    db_chunks: List[Dict[str, Any]],
    fallback_context: str = "",
    refinement_units: Optional[List[str]] = None,
) -> List[Optional[str]]:
    """Map translation snapshots onto refinement units using output provenance.

    When translation and refinement produce the same number of units, mapping
    is exact. Otherwise translated-text and refinement-unit lengths define a
    cumulative position timeline, which is substantially more accurate than
    mapping by chunk count alone.
    """
    if total_chunks <= 0:
        return []

    timeline_rows: List[Tuple[str, int]] = []
    last_snapshot: Optional[str] = None
    for chunk in sorted(
        db_chunks or [],
        key=lambda item: item.get("chunk_index", -1),
    ):
        status = chunk.get("status")
        if status is not None and status != "completed":
            continue
        if status == "completed" and chunk.get("translated_text") is None:
            continue
        snapshot = (chunk.get("chunk_data") or {}).get("context_snapshot")
        if snapshot:
            last_snapshot = snapshot
        if not last_snapshot:
            continue
        translated_text = str(chunk.get("translated_text") or "")
        source_text = str(chunk.get("original_text") or "")
        weight = max(len(translated_text.strip()), len(source_text.strip()), 1)
        timeline_rows.append((last_snapshot, weight))

    if not timeline_rows:
        return [None] * total_chunks

    def decode(snapshot: str) -> str:
        full_context, _, _ = decode_context_snapshot(
            snapshot,
            fallback_context,
        )
        return full_context

    if len(timeline_rows) == total_chunks:
        return [decode(snapshot) for snapshot, _ in timeline_rows]

    contexts: List[Optional[str]] = []
    if refinement_units and len(refinement_units) == total_chunks:
        target_weights = [max(len(str(unit or "").strip()), 1) for unit in refinement_units]
        source_total = sum(weight for _, weight in timeline_rows)
        target_total = sum(target_weights)
        source_boundaries: List[int] = []
        cumulative_source = 0
        for _, weight in timeline_rows:
            cumulative_source += weight
            source_boundaries.append(cumulative_source)

        cumulative_target = 0
        source_index = 0
        for weight in target_weights:
            target_midpoint = cumulative_target + (weight / 2)
            source_position = (target_midpoint / target_total) * source_total
            while (
                source_index < len(source_boundaries) - 1
                and source_position > source_boundaries[source_index]
            ):
                source_index += 1
            contexts.append(decode(timeline_rows[source_index][0]))
            cumulative_target += weight
        return contexts

    for index in range(total_chunks):
        mapped_index = min(
            int(index * len(timeline_rows) / total_chunks),
            len(timeline_rows) - 1,
        )
        contexts.append(decode(timeline_rows[mapped_index][0]))
    return contexts
def map_dialogue_attributions_for_refinement(
    total_chunks: int,
    db_chunks: List[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    """Reuse dialogue maps only when translation/refinement units align exactly.

    Unlike cumulative lore snapshots, a speaker map belongs to one local source
    unit. Guessing a proportional mapping after re-chunking could attach the
    wrong speaker to unrelated dialogue, so mismatched layouts deliberately
    fall back to fresh monolingual analysis during refinement.
    """
    if total_chunks <= 0:
        return []
    rows = [
        chunk
        for chunk in sorted(
            db_chunks or [],
            key=lambda item: item.get("chunk_index", -1),
        )
        if chunk.get("status") == "completed"
        and chunk.get("translated_text") is not None
    ]
    if len(rows) != total_chunks:
        return [None] * total_chunks
    return [
        (row.get("chunk_data") or {}).get("dialogue_attribution")
        for row in rows
    ]
