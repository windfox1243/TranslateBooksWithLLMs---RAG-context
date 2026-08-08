"""Reading the context a resumed job should start from.

Every format resumes the same way: find the newest already-processed chunk that
carries a context snapshot, take its snapshot and dialogue state, and fall back
to the continuation seed when there is no such chunk. The formats disagree only
on *which* rows are candidates -- `generic_translator` reads the checkpoint
payload it was handed, `plain_text_pipeline` queries the database for rows below
its global offset -- so the selection stays at the call site and only the
reading is shared here.

Keeping it in one place matters because the three fields are read out of
`chunk_data` by string key, and a snapshot that silently resolves to None
restarts the novel context from scratch mid-book rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set


@dataclass(frozen=True)
class ResumeContext:
    """The context state a resumed run starts from. All fields optional."""

    snapshot: Optional[Any] = None
    snapshot_index: Optional[int] = None
    dialogue_state: Optional[Any] = None
    dialogue_scene_key: Optional[Any] = None
    from_continuation_seed: bool = False


EMPTY_RESUME_CONTEXT = ResumeContext()


def resume_context_from_chunk_data(
    chunk_data: Optional[Dict[str, Any]],
    snapshot_index: Optional[int] = None,
) -> ResumeContext:
    """Read snapshot and dialogue state out of one chunk's `chunk_data`."""
    data = chunk_data or {}
    attribution = data.get("dialogue_attribution") or {}
    return ResumeContext(
        snapshot=data.get("context_snapshot"),
        snapshot_index=snapshot_index,
        dialogue_state=attribution.get("state_after"),
        dialogue_scene_key=attribution.get("scene_key"),
    )


def resume_context_from_seed(seed: Optional[Dict[str, Any]]) -> ResumeContext:
    """Read the same state out of a continuation seed.

    The seed is flat and uses its own key names, which is why this cannot just
    be `resume_context_from_chunk_data`.
    """
    data = seed or {}
    return ResumeContext(
        snapshot=data.get("context_snapshot"),
        snapshot_index=data.get("chunk_index"),
        dialogue_state=data.get("dialogue_state"),
        dialogue_scene_key=data.get("dialogue_scene_key"),
        from_continuation_seed=True,
    )


def newest_chunk(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The row with the highest `chunk_index`, or None for an empty sequence."""
    candidates = list(rows)
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.get("chunk_index", -1))


@dataclass(frozen=True)
class CheckpointContext:
    """What a checkpoint payload says about context, for one resumed job."""

    resume: ResumeContext
    # Both are accumulators the caller keeps writing to as the run proceeds,
    # so they are deliberately mutable.
    analyzed_indices: Set[int]
    context_data_by_index: Dict[int, Dict[str, Any]]


_CONTEXT_BEARING_STATUSES = ("completed", "partial", "failed")


def read_checkpoint_context(
    checkpoint_data: Optional[Dict[str, Any]],
    restored_completed: Iterable[int],
    continuation_seed: Optional[Dict[str, Any]],
) -> CheckpointContext:
    """Derive the resume context from a checkpoint payload.

    Three sources, in falling order of preference: the newest chunk that
    actually carries a snapshot, the highest completed chunk index (which can
    name a chunk whose snapshot is missing -- the index is still reported so
    the caller knows where the run left off), and finally the continuation
    seed, which only applies when nothing above produced a snapshot.
    """
    analyzed_indices: Set[int] = set()
    context_data_by_index: Dict[int, Dict[str, Any]] = {}
    resume = EMPTY_RESUME_CONTEXT

    if checkpoint_data:
        chunks = checkpoint_data.get("chunks", [])
        context_rows = []
        for chunk in chunks:
            chunk_data = chunk.get("chunk_data") or {}
            chunk_index = chunk.get("chunk_index")
            if (
                isinstance(chunk_index, int)
                and chunk_data.get("context_snapshot")
                and chunk.get("status") in _CONTEXT_BEARING_STATUSES
            ):
                analyzed_indices.add(chunk_index)
                context_data_by_index[chunk_index] = dict(chunk_data)
                context_rows.append(chunk)

        newest = newest_chunk(context_rows)
        if newest is not None:
            resume = resume_context_from_chunk_data(
                newest.get("chunk_data"), newest.get("chunk_index")
            )
        else:
            restored = list(restored_completed)
            if restored:
                snapshot_index = max(restored)
                # The index stands on its own even when no chunk matches it,
                # which is why it is not read off the row below.
                resume = ResumeContext(snapshot_index=snapshot_index)
                for chunk in chunks:
                    if chunk.get("chunk_index") == snapshot_index:
                        resume = resume_context_from_chunk_data(
                            chunk.get("chunk_data"), snapshot_index
                        )
                        break

    if not resume.snapshot and continuation_seed:
        resume = resume_context_from_seed(continuation_seed)

    return CheckpointContext(resume, analyzed_indices, context_data_by_index)
