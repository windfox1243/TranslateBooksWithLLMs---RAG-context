"""Resume-context selection, including the cases that used to be inline.

The interesting behaviour is the precedence between the three sources and one
asymmetry inside it: the restored-completed fallback reports a chunk index even
when it cannot find a snapshot for it. That was implicit in the original inline
code and is easy to "simplify" away, which would make a resumed run report the
wrong position.
"""

from src.core.common.resume_context import (
    newest_chunk,
    read_checkpoint_context,
    resume_context_from_chunk_data,
    resume_context_from_seed,
)


def _chunk(index, status="completed", snapshot="snap", attribution=None):
    data = {"context_snapshot": snapshot}
    if attribution is not None:
        data["dialogue_attribution"] = attribution
    return {"chunk_index": index, "status": status, "chunk_data": data}


def test_reads_snapshot_and_dialogue_state_from_chunk_data():
    resume = resume_context_from_chunk_data(
        {
            "context_snapshot": "snap",
            "dialogue_attribution": {"state_after": "st", "scene_key": "sc"},
        },
        7,
    )
    assert resume.snapshot == "snap"
    assert resume.snapshot_index == 7
    assert resume.dialogue_state == "st"
    assert resume.dialogue_scene_key == "sc"
    assert resume.from_continuation_seed is False


def test_missing_dialogue_attribution_is_not_an_error():
    resume = resume_context_from_chunk_data({"context_snapshot": "snap"})
    assert resume.snapshot == "snap"
    assert resume.dialogue_state is None
    assert resume.dialogue_scene_key is None


def test_a_null_dialogue_attribution_is_treated_as_absent():
    # The column is nullable, so `or {}` in the reader is load-bearing.
    resume = resume_context_from_chunk_data(
        {"context_snapshot": "snap", "dialogue_attribution": None}
    )
    assert resume.dialogue_state is None


def test_seed_uses_its_own_flat_key_names():
    resume = resume_context_from_seed({
        "context_snapshot": "snap",
        "chunk_index": 3,
        "dialogue_state": "st",
        "dialogue_scene_key": "sc",
    })
    assert (resume.snapshot, resume.snapshot_index) == ("snap", 3)
    assert (resume.dialogue_state, resume.dialogue_scene_key) == ("st", "sc")
    assert resume.from_continuation_seed is True


def test_newest_chunk_of_nothing_is_none():
    assert newest_chunk([]) is None


def test_the_newest_snapshot_bearing_chunk_wins():
    checkpoint = {"chunks": [_chunk(0), _chunk(5), _chunk(2)]}
    result = read_checkpoint_context(checkpoint, [0, 2, 5], None)
    assert result.resume.snapshot_index == 5
    assert result.analyzed_indices == {0, 2, 5}
    assert sorted(result.context_data_by_index) == [0, 2, 5]


def test_chunks_without_a_snapshot_are_not_candidates():
    checkpoint = {"chunks": [_chunk(1), _chunk(9, snapshot=None)]}
    result = read_checkpoint_context(checkpoint, [1, 9], None)
    assert result.resume.snapshot_index == 1
    assert result.analyzed_indices == {1}


def test_chunks_in_an_unfinished_status_are_not_candidates():
    checkpoint = {"chunks": [_chunk(1), _chunk(9, status="pending")]}
    result = read_checkpoint_context(checkpoint, [1], None)
    assert result.resume.snapshot_index == 1


def test_failed_chunks_still_carry_context():
    # A failed unit has already been analysed, so its snapshot is usable.
    checkpoint = {"chunks": [_chunk(4, status="failed")]}
    result = read_checkpoint_context(checkpoint, [], None)
    assert result.resume.snapshot_index == 4


def test_restored_completed_reports_its_index_even_with_no_snapshot():
    # No chunk carries a snapshot, so there is nothing to resume the context
    # from -- but the run still left off at chunk 6 and must say so.
    checkpoint = {"chunks": [_chunk(6, snapshot=None)]}
    result = read_checkpoint_context(checkpoint, [2, 6], None)
    assert result.resume.snapshot is None
    assert result.resume.snapshot_index == 6
    assert result.analyzed_indices == set()


def test_the_seed_only_applies_when_no_snapshot_was_found():
    seed = {"context_snapshot": "seed", "chunk_index": 99}
    with_snapshot = read_checkpoint_context({"chunks": [_chunk(1)]}, [1], seed)
    assert with_snapshot.resume.snapshot == "snap"
    assert with_snapshot.resume.from_continuation_seed is False

    without = read_checkpoint_context({"chunks": []}, [], seed)
    assert without.resume.snapshot == "seed"
    assert without.resume.snapshot_index == 99
    assert without.resume.from_continuation_seed is True


def test_the_seed_overrides_an_index_only_fallback():
    # The index-only fallback leaves snapshot None, so the seed takes over --
    # index included.
    seed = {"context_snapshot": "seed", "chunk_index": 99}
    result = read_checkpoint_context(
        {"chunks": [_chunk(6, snapshot=None)]}, [6], seed
    )
    assert (result.resume.snapshot, result.resume.snapshot_index) == ("seed", 99)


def test_no_checkpoint_and_no_seed_is_empty():
    result = read_checkpoint_context(None, [], None)
    assert result.resume.snapshot is None
    assert result.resume.snapshot_index is None
    assert result.analyzed_indices == set()
    assert result.context_data_by_index == {}


def test_the_accumulators_are_the_callers_to_keep_writing_to():
    result = read_checkpoint_context({"chunks": [_chunk(1)]}, [1], None)
    result.analyzed_indices.add(2)
    result.context_data_by_index[2] = {}
    assert result.analyzed_indices == {1, 2}
