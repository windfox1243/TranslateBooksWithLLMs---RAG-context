import pytest

from src.core.epub import xhtml_translator
from src.core.epub.translation_metrics import TranslationMetrics
from src.persistence.checkpoint_manager import CheckpointManager


def _chunks(count):
    return [
        {
            "text": f"source-{index}",
            "local_tag_map": {},
            "global_indices": [],
        }
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_phase3_fallback_remains_retryable(monkeypatch, tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir()
    manager.start_job("job", "epub", {}, None)

    async def first_pass(**kwargs):
        index = int(kwargs["chunk_text"].rsplit("-", 1)[1])
        return xhtml_translator._ChunkTranslationOutcome(
            f"translated-{index}" if index != 1 else kwargs["chunk_text"],
            succeeded=index != 1,
        )

    monkeypatch.setattr(
        xhtml_translator,
        "translate_chunk_with_fallback",
        first_pass,
    )
    chunks = _chunks(3)
    stats = TranslationMetrics(total_chunks=3)

    translated, stats, interrupted = await (
        xhtml_translator._translate_all_chunks_with_checkpoint(
            chunks=chunks,
            source_language="Korean",
            target_language="English",
            model_name="model",
            llm_client=object(),
            max_retries=1,
            context_manager=None,
            placeholder_format=("[id", "]"),
            checkpoint_manager=manager,
            translation_id="job",
            file_href="chapter.xhtml",
            file_path="chapter.xhtml",
            translated_chunks=[],
            global_tag_map={},
            stats=stats,
        )
    )

    assert interrupted is False
    assert translated == ["translated-0", "source-1", "translated-2"]
    assert stats.failed_chunks == 1

    state = manager.load_xhtml_partial_state("job", "chapter.xhtml")
    assert state.failed_chunk_indices == [1]
    assert state.current_chunk_index == 3
    assert [
        row["status"] for row in manager.db.get_chunks("job")
    ] == ["completed", "failed", "completed"]

    retried_indices = []

    async def resumed_pass(**kwargs):
        index = int(kwargs["chunk_text"].rsplit("-", 1)[1])
        retried_indices.append(index)
        return xhtml_translator._ChunkTranslationOutcome(
            f"translated-{index}",
            succeeded=True,
        )

    monkeypatch.setattr(
        xhtml_translator,
        "translate_chunk_with_fallback",
        resumed_pass,
    )
    resumed_stats = TranslationMetrics.from_dict(state.stats)
    translated, resumed_stats, interrupted = await (
        xhtml_translator._translate_all_chunks_with_checkpoint(
            chunks=state.chunks,
            source_language="Korean",
            target_language="English",
            model_name="model",
            llm_client=object(),
            max_retries=1,
            context_manager=None,
            placeholder_format=state.placeholder_format,
            checkpoint_manager=manager,
            translation_id="job",
            file_href="chapter.xhtml",
            file_path="chapter.xhtml",
            start_chunk_index=state.current_chunk_index,
            translated_chunks=state.translated_chunks.copy(),
            global_tag_map=state.global_tag_map,
            stats=resumed_stats,
            failed_chunk_indices=state.failed_chunk_indices,
        )
    )

    assert interrupted is False
    assert retried_indices == [1]
    assert translated == ["translated-0", "translated-1", "translated-2"]
    assert resumed_stats.failed_chunks == 0
    assert [
        row["status"] for row in manager.db.get_chunks("job")
    ] == ["completed", "completed", "completed"]
