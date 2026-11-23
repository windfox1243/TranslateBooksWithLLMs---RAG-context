"""Chunks that keep source-language text are recorded as failed, not translated.

``generate_translation_request`` already retries a chunk once when its answer
still holds source-language text. What is left after that retry is kept, since
it is still the most translated version, but the EPUB placeholder path and the
Plain Text Mode pipeline now record the chunk as failed, so a resume or a retry
pass translates it again. A legitimate echo (a bibliography entry) is never
flagged. (Ported from upstream c2330c52.)
"""
import asyncio

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline
import src.core.epub.xhtml_translator as xt
from src.core.epub.translation_metrics import TranslationMetrics


async def _translate_chunk(monkeypatch, chunk, tag_map, answer,
                           source_language="Korean", target_language="English"):
    async def fake_request(main_content, *args, **kwargs):
        return answer(main_content)

    monkeypatch.setattr(xt, "generate_translation_request", fake_request)
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", False)
    stats = TranslationMetrics()
    result = await xt.translate_chunk_with_fallback(
        chunk_text=chunk,
        local_tag_map=tag_map,
        global_indices=list(range(len(tag_map))),
        source_language=source_language,
        target_language=target_language,
        model_name="fake",
        llm_client=object(),
        stats=stats,
        max_retries=1,
    )
    return result, stats


TWO_PARAGRAPHS = {"[id0]": "<p>", "[id1]": "</p>", "[id2]": "<p>", "[id3]": "</p>"}


@pytest.mark.asyncio
async def test_residue_chunk_is_failed_but_keeps_its_translation(monkeypatch):
    result, stats = await _translate_chunk(
        monkeypatch,
        "[id0]그는 문을 열었다.[id1][id2]바람이 차가웠다.[id3]",
        TWO_PARAGRAPHS,
        lambda text: text.replace("그는 문을 열었다.", "He opened the door."),
    )
    assert result.succeeded is False
    assert stats.fallback_used == 1
    assert "He opened the door." in result


@pytest.mark.asyncio
async def test_echoed_bibliography_stays_translated(monkeypatch):
    entry = "Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997."
    result, stats = await _translate_chunk(
        monkeypatch, f"[id0]{entry}[id1]", {"[id0]": "<p>", "[id1]": "</p>"},
        lambda text: text, source_language="English", target_language="French",
    )
    assert result.succeeded is True
    assert stats.fallback_used == 0


@pytest.mark.asyncio
async def test_clean_chunk_is_still_translated(monkeypatch):
    result, stats = await _translate_chunk(
        monkeypatch, "[id0]그는 문을 열었다.[id1]", {"[id0]": "<p>", "[id1]": "</p>"},
        lambda text: "[id0]He opened the door.[id1]",
    )
    assert result.succeeded is True
    assert stats.fallback_used == 0


@pytest.mark.asyncio
async def test_plain_text_residue_segment_is_failed_but_kept(monkeypatch):
    async def fake(*, main_content, **kwargs):
        await asyncio.sleep(0)
        return main_content.replace("그는 문을 열었다.", "He opened the door.")

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
    translated, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        ["그는 문을 열었다. 바람이 차가웠다."],
        "Korean", "English", "m", object(), max_tokens_per_chunk=2000,
    )
    assert not interrupted
    assert stats.failed_chunks == 1
    assert translated[0].startswith("He opened the door.")


@pytest.mark.asyncio
async def test_plain_text_echo_is_not_a_failed_chunk(monkeypatch):
    async def fake(*, main_content, **kwargs):
        await asyncio.sleep(0)
        return main_content

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
    text = "Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997."
    translated, stats, _ = await plain_pipeline.translate_paragraphs_plain(
        [text], "English", "French", "m", object(), max_tokens_per_chunk=2000)
    assert stats.failed_chunks == 0
    assert translated == [text]
