from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.epub import translator


@pytest.mark.asyncio
async def test_resume_uses_source_precount_and_includes_partial_file_progress(
    monkeypatch,
    tmp_path,
):
    async def unexpected_precount(*args, **kwargs):
        raise AssertionError("precomputed source counts should be reused")

    monkeypatch.setattr(translator, "_precount_chunks", unexpected_precount)

    class CheckpointManager:
        def get_job(self, translation_id):
            return {"progress": {}}

        def load_xhtml_partial_state(self, translation_id, file_href):
            assert file_href == "chapter-2.xhtml"
            return SimpleNamespace(
                current_chunk_index=2,
                chunks=[{}, {}, {}, {}, {}, {}],
            )

    emitted_stats = []
    result = await translator._process_all_content_files(
        content_files=["chapter-1.xhtml", "chapter-2.xhtml"],
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="Korean",
        target_language="English",
        model_name="model",
        llm_client=object(),
        max_tokens_per_chunk=2000,
        max_attempts=1,
        context_manager=None,
        translation_id="job",
        resume_from_index=1,
        checkpoint_manager=CheckpointManager(),
        stats_callback=emitted_stats.append,
        check_interruption_callback=lambda: True,
        precomputed_chunk_counts=(10, [4, 6]),
    )

    assert emitted_stats[0]["total_chunks"] == 10
    assert emitted_stats[0]["completed_chunks"] == 6
    assert result["was_interrupted"] is True


def test_epub_source_is_counted_before_translated_files_are_restored():
    import inspect

    source = inspect.getsource(translator.translate_epub_file)

    assert source.index(
        "source_chunk_counts = await _precount_chunks"
    ) < source.index(
        "restored_docs = await _restore_checkpoint_files"
    )
