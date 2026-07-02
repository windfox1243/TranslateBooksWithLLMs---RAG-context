import pytest

from src.core.continuation import (
    latest_context_seed,
    matching_prefix_length,
    seed_matching_prefix,
)
from src.core.common.plain_text_pipeline import translate_paragraphs_plain
from src.persistence.checkpoint_manager import CheckpointManager


def test_matching_prefix_stops_at_first_untranslated_old_unit():
    previous_chunks = [
        {
            "chunk_index": 0,
            "original_text": "Chapter 1",
            "translated_text": "Capítulo 1",
            "status": "completed",
        },
        {
            "chunk_index": 1,
            "original_text": "Chapter 2",
            "translated_text": None,
            "status": "failed",
        },
        {
            "chunk_index": 2,
            "original_text": "Chapter 3",
            "translated_text": "Capítulo 3",
            "status": "completed",
        },
    ]

    assert matching_prefix_length(
        previous_chunks,
        ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"],
    ) == 1


def test_matching_prefix_stops_at_first_edited_old_unit():
    previous_chunks = [
        {
            "chunk_index": 0,
            "original_text": "Chapter 1",
            "translated_text": "Capítulo 1",
            "status": "completed",
        },
        {
            "chunk_index": 1,
            "original_text": "Chapter 2 old",
            "translated_text": "Capítulo 2",
            "status": "completed",
        },
    ]

    assert matching_prefix_length(
        previous_chunks,
        ["Chapter 1", "Chapter 2 revised", "Chapter 3"],
    ) == 1


def test_seed_matching_prefix_creates_new_job_prefix_only(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job("continued", "txt", {"file_type": "txt"})
    previous_chunks = [
        {
            "chunk_index": 0,
            "original_text": "Chapter 1",
            "translated_text": "Translated 1",
            "chunk_data": {"context_snapshot": "ctx-1"},
            "status": "completed",
        },
        {
            "chunk_index": 1,
            "original_text": "Chapter 2",
            "translated_text": None,
            "chunk_data": {},
            "status": "failed",
        },
        {
            "chunk_index": 2,
            "original_text": "Chapter 3",
            "translated_text": "Translated 3",
            "chunk_data": {"context_snapshot": "ctx-3"},
            "status": "completed",
        },
    ]

    reused = seed_matching_prefix(
        checkpoint_manager=manager,
        translation_id="continued",
        previous_chunks=previous_chunks,
        new_source_units=["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"],
        total_units=4,
    )

    checkpoint = manager.load_checkpoint("continued")
    chunks = checkpoint["chunks"]

    assert reused == 1
    assert checkpoint["resume_from_index"] == 1
    assert [chunk["chunk_index"] for chunk in chunks] == [0]
    assert chunks[0]["translated_text"] == "Translated 1"
    assert chunks[0]["chunk_data"]["context_snapshot"] == "ctx-1"


def test_seed_matching_prefix_includes_completed_jobs_in_saved_list(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job("complete-base", "txt", {"file": "book.txt"})
    manager.mark_completed("complete-base")

    saved = manager.get_resumable_jobs()

    assert "complete-base" in {job["translation_id"] for job in saved}


def test_latest_context_seed_uses_last_context_bearing_old_chunk():
    seed = latest_context_seed([
        {
            "chunk_index": 1,
            "status": "completed",
            "chunk_data": {
                "context_snapshot": "ctx-1",
                "dialogue_attribution": {
                    "state_after": {"A": "B"},
                    "scene_key": "scene-1",
                },
            },
        },
        {
            "chunk_index": 2,
            "status": "completed",
            "chunk_data": {},
        },
        {
            "chunk_index": 3,
            "status": "partial",
            "chunk_data": {
                "context_snapshot": "ctx-3",
                "dialogue_attribution": {
                    "state_after": {"C": "D"},
                    "scene_key": "scene-3",
                },
            },
        },
    ])

    assert seed == {
        "chunk_index": 3,
        "context_snapshot": "ctx-3",
        "dialogue_state": {"C": "D"},
        "dialogue_scene_key": "scene-3",
        "dialogue_attribution": {
            "state_after": {"C": "D"},
            "scene_key": "scene-3",
        },
    }


@pytest.mark.asyncio
async def test_plain_continuation_without_prefix_starts_from_old_context(
    monkeypatch,
    tmp_path,
):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job("old", "txt", {"file_type": "txt"})
    manager.start_job("continued", "txt", {"file_type": "txt"})
    manager.db.save_chunk(
        translation_id="old",
        chunk_index=4,
        original_text="Old ending",
        translated_text="Translated old ending",
        chunk_data={
            "context_snapshot": "old-final-context",
            "dialogue_attribution": {
                "state_after": {"speaker": "old-state"},
                "scene_key": "old-scene",
            },
        },
        status="completed",
    )
    captured = {}

    def fake_open_session(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "src.utils.novel_context.open_novel_context_session",
        fake_open_session,
    )

    await translate_paragraphs_plain(
        paragraphs=["Brand new chapter only"],
        source_language="Chinese",
        target_language="Vietnamese",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=100,
        prompt_options={"auto_update_context": True},
        checkpoint_manager=manager,
        translation_id="continued",
        continuation_base_id="old",
        check_interruption_callback=lambda: True,
    )

    assert captured["resume_snapshot"] == "old-final-context"
    assert captured["resume_dialogue_state"] == {"speaker": "old-state"}
    assert captured["resume_dialogue_scene_key"] == "old-scene"
