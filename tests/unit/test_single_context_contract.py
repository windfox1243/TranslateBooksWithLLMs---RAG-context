"""Every job runs the current context contract; older numbers are ignored."""

import pytest

from src.utils.addressing_schema import (
    ADDRESSING_CONTRACT_VERSION,
    context_contract_version,
)


@pytest.mark.parametrize(
    "options",
    [None, {}, {"context_contract_version": 1}, {"context_contract_version": "3"}],
)
def test_any_stored_version_resolves_to_the_current_contract(options):
    assert context_contract_version(options) == ADDRESSING_CONTRACT_VERSION


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
