from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_epub_translation_propagates_failure(monkeypatch, tmp_path):
    from src.core.adapters.translate_file import translate_file
    import src.core.epub.translator as epub_translator

    input_path = tmp_path / "book.epub"
    input_path.write_bytes(b"not-needed-by-mocked-translator")
    mocked_translate = AsyncMock(return_value=False)
    monkeypatch.setattr(
        epub_translator,
        "translate_epub_file",
        mocked_translate,
    )

    success = await translate_file(
        input_filepath=str(input_path),
        output_filepath=str(tmp_path / "translated.epub"),
        source_language="Korean",
        target_language="English",
        model_name="gemini-test",
        llm_provider="gemini",
        checkpoint_manager=object(),
        translation_id="job",
        gemini_api_key="test-placeholder",
    )

    assert success is False
    mocked_translate.assert_awaited_once()
