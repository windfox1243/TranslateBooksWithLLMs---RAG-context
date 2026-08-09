from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_epub_translation_propagates_failure(monkeypatch, tmp_path):
    import src.core.epub.translator as epub_translator
    from src.core.adapters.translate_file import translate_file

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


@pytest.mark.asyncio
async def test_context_window_reaches_the_llm_client(monkeypatch, tmp_path):
    """The caller's context window must survive the trip to the provider.

    It used to be dropped between translate_file and the LLM configuration the
    translator builds its client from, so every job silently ran at
    OLLAMA_NUM_CTX no matter what the caller asked for.
    """
    import importlib

    from src.core.adapters.translate_file import translate_file

    # The package re-exports the function under the module's own name, so ask
    # importlib for the module rather than relying on attribute lookup.
    module = importlib.import_module("src.core.adapters.translate_file")

    input_path = tmp_path / "book.txt"
    input_path.write_text("One line is enough.\n", encoding="utf-8")

    seen = {}

    class RecordingTranslator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def translate(self, **kwargs):
            seen.update(kwargs)
            return True

    monkeypatch.setattr(module, "GenericTranslator", RecordingTranslator)

    await translate_file(
        input_filepath=str(input_path),
        output_filepath=str(tmp_path / "translated.txt"),
        source_language="English",
        target_language="Vietnamese",
        model_name="any-model",
        llm_provider="ollama",
        checkpoint_manager=None,
        translation_id="job",
        context_window=16384,
    )

    assert seen["context_window"] == 16384
