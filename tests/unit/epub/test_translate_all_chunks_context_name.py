"""_translate_all_chunks names a novel context file without a file name.

The shared orchestrator (DOCX and other non-EPUB formats) reaches this loop
with no `file_href` in scope. With auto-update on and no input_filename in
the prompt options, the fallback referenced that undefined name and raised
NameError before translating anything.
"""

import pytest

import src.core.epub.xhtml_translator as xt


@pytest.mark.asyncio
async def test_auto_context_name_without_input_filename(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.NOVEL_CONTEXTS_DIR", tmp_path, raising=False)
    prompt_options = {"auto_update_context": True}

    translated, _stats = await xt._translate_all_chunks(
        chunks=[],
        source_language="English",
        target_language="French",
        model_name="test-model",
        llm_client=None,
        max_retries=1,
        context_manager=None,
        placeholder_format=("[", "]"),
        prompt_options=prompt_options,
    )

    assert translated == []
    assert prompt_options["novel_context_file"]
