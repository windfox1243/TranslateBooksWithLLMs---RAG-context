"""Plain Text Mode pipeline: missing paragraphs and <pre> blocks.

- _reconcile_paragraph_counts padded missing paragraphs with "", which
  _reassemble kept as empty paragraphs instead of the source text.
- A <pre> paragraph holding blank lines always broke the paragraph count of
  its segment and had its blank lines collapsed.
(Ported from upstream dde993ee.)
"""

import re

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline

PARAGRAPHS = ["Alpha paragraph.", "Beta paragraph.", "Gamma paragraph."]


def test_missing_paragraphs_keep_their_source_text():
    segments = [{'indices': [0, 1, 2], 'text': "\n\n".join(PARAGRAPHS), 'partial': False}]
    out = plain_pipeline._reassemble(segments, ["T::Alpha paragraph."], PARAGRAPHS)
    assert out == ["T::Alpha paragraph.", "Beta paragraph.", "Gamma paragraph."]


def test_pre_block_gets_its_own_segment():
    pre = "def f():\n    return 1\n\n\ndef g():\n    return 2"
    segments = plain_pipeline.build_plain_segments(["Intro.", pre, "Outro."], 1000)
    pre_segments = [s for s in segments if s['text'] == pre]
    assert len(pre_segments) == 1
    assert pre_segments[0]['indices'] == [1]


@pytest.mark.asyncio
async def test_pre_block_with_blank_lines_keeps_them(monkeypatch):
    pre = "def f():\n    return 1\n\n\ndef g():\n    return 2"
    source = ["Intro paragraph.", pre, "Outro paragraph."]

    async def per_paragraph(*, main_content, **kwargs):
        if main_content == pre:
            return main_content  # code comes back verbatim
        paragraphs = [p for p in re.split(r"\n{2,}", main_content) if p.strip()]
        return "\n\n".join("T::" + p for p in paragraphs)

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", per_paragraph)

    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        source, "English", "French", "m", None, 1000, prompt_options={},
    )

    assert not interrupted
    assert out == ["T::Intro paragraph.", pre, "T::Outro paragraph."]
