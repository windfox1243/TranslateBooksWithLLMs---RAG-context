"""Every job runs the single current context contract."""

import pytest


@pytest.mark.asyncio
async def test_output_sync_leaves_non_vietnamese_context_alone(tmp_path):
    from src.utils.novel_context import NovelContextSession

    context_file = tmp_path / "ctx.txt"
    context_file.write_text("=== GLOBAL LORE ===\n", encoding="utf-8")
    session = NovelContextSession(
        path=context_file,
        prompt_options={},
        global_lore="=== GLOBAL LORE ===\n",
        dynamic_state="ORIGINAL",
    )
    # No target language at all used to fall back to Vietnamese rules.
    assert await session.sync_translated_output(translated_chunk="Bonjour.") is False
    assert await session.sync_translated_output(
        translated_chunk="Bonjour.", target_language="French",
    ) is False
    assert session.dynamic_state == "ORIGINAL"
