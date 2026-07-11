"""Regression tests for issue #205: SRT refine cues mis-assigned.

The refine-only path used to key translations by the cue number printed in
the file while update_translated_subtitles applies them by list position.
Any SRT not numbered exactly 1..N (gaps, restarts, 0-based numbering) got
its refined text assigned to the wrong cue / timestamp.
"""

import re

import pytest

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.srt_processor import SRTProcessor


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    """Echoes back the input block with every subtitle text uppercased.

    Parses the [N]text lines between the SOURCE_TEXT tags of the user
    prompt, so the refined output keeps the exact [N] markers the caller
    sent and only changes the casing of the text.
    """

    def __init__(self):
        self.draft = ""

    async def make_request(self, prompt, model_name, system_prompt=None):
        if "DRAFT TRANSLATION TO AUDIT" in prompt:
            match = re.search(
                r"# DRAFT TRANSLATION TO AUDIT:\n(.*?)\n\nPerform",
                prompt,
                re.DOTALL,
            )
            assert match
            self.draft = match.group(1).strip()
            replacement = self.draft.upper()
            import json
            payload = {
                "status": "needs_repair",
                "issues": [{
                    "category": "style",
                    "severity": "major",
                    "source_quote": "",
                    "draft_quote": self.draft,
                    "instruction": "Use the required uppercase test style.",
                    "draft_replacement": {
                        "draft": self.draft,
                        "replacement": replacement,
                    },
                    "glossary_update": None,
                    "term_replacement": None,
                }],
            }
            return FakeLLMResponse(
                f"<REFLECTION_JSON>{json.dumps(payload)}</REFLECTION_JSON>"
            )
        assert "SENIOR EDITOR CRITIQUE" in prompt
        refined_lines = []
        for line in self.draft.split("\n"):
            marker = re.match(r"^(\[\d+\])(.*)$", line)
            if marker:
                refined_lines.append(f"{marker.group(1)}{marker.group(2).upper()}")
            else:
                refined_lines.append(line.upper())
        body = "\n".join(refined_lines)
        return FakeLLMResponse(f"{TRANSLATE_TAG_IN}\n{body}\n{TRANSLATE_TAG_OUT}")

    def extract_translation(self, content: str):
        match = re.search(
            re.escape(TRANSLATE_TAG_IN) + r"\n(.*?)\n" + re.escape(TRANSLATE_TAG_OUT),
            content,
            re.DOTALL,
        )
        return match.group(1) if match else content

    async def close(self):
        pass


def _build_srt(cues):
    """cues: list of (number, start, end, text) tuples."""
    blocks = []
    for number, start, end, text in cues:
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


_ZERO_WIDTH = re.compile('[\u200b\u200c\u200d\u2060\ufeff]')


def _parse_output(content):
    """Map start_time -> text for every cue of a generated SRT file.

    Zero-width characters are stripped: reconstruct_srt may add invisible
    normalization characters to the first cue, which are irrelevant to the
    cue/text assignment under test.
    """
    processor = SRTProcessor()
    return {
        sub["start_time"]: _ZERO_WIDTH.sub("", sub["text"])
        for sub in processor.parse_srt(content)
    }


async def _refine(tmp_path, monkeypatch, srt_content):
    import src.core.refine.srt_refiner as srt_refiner

    monkeypatch.setattr(
        srt_refiner, "create_llm_client", lambda *args, **kwargs: FakeLLMClient()
    )

    input_file = tmp_path / "input.srt"
    output_file = tmp_path / "output.srt"
    input_file.write_text(srt_content, encoding="utf-8")

    ok = await srt_refiner.refine_srt_file(
        input_filepath=str(input_file),
        output_filepath=str(output_file),
        target_language="French",
    )
    assert ok, "refine_srt_file reported failure"
    return output_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_refine_srt_gapped_numbering(tmp_path, monkeypatch):
    """Cue numbers with a gap (1, 5, 6) must keep text under its own timestamp."""
    srt = _build_srt([
        ("1", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("5", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("6", "00:00:05,000", "00:00:06,000", "charlie three"),
    ])

    output = await _refine(tmp_path, monkeypatch, srt)
    by_start = _parse_output(output)

    assert by_start["00:00:01,000"] == "ALPHA ONE"
    assert by_start["00:00:03,000"] == "BRAVO TWO"
    assert by_start["00:00:05,000"] == "CHARLIE THREE"


@pytest.mark.asyncio
async def test_refine_srt_zero_based_numbering(tmp_path, monkeypatch):
    """0-based cue numbers must not shift texts (cue 0 used to land on the last cue)."""
    srt = _build_srt([
        ("0", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("1", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("2", "00:00:05,000", "00:00:06,000", "charlie three"),
    ])

    output = await _refine(tmp_path, monkeypatch, srt)
    by_start = _parse_output(output)

    assert by_start["00:00:01,000"] == "ALPHA ONE"
    assert by_start["00:00:03,000"] == "BRAVO TWO"
    assert by_start["00:00:05,000"] == "CHARLIE THREE"


@pytest.mark.asyncio
async def test_refine_srt_restarted_numbering(tmp_path, monkeypatch):
    """Duplicate cue numbers (restart at 1) must not collapse or overwrite cues."""
    srt = _build_srt([
        ("1", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("2", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("1", "00:01:01,000", "00:01:02,000", "charlie three"),
        ("2", "00:01:03,000", "00:01:04,000", "delta four"),
    ])

    output = await _refine(tmp_path, monkeypatch, srt)
    by_start = _parse_output(output)

    assert by_start["00:00:01,000"] == "ALPHA ONE"
    assert by_start["00:00:03,000"] == "BRAVO TWO"
    assert by_start["00:01:01,000"] == "CHARLIE THREE"
    assert by_start["00:01:03,000"] == "DELTA FOUR"


def test_update_translated_subtitles_ignores_negative_index():
    """A negative index must not silently write to the end of the list."""
    processor = SRTProcessor()
    subtitles = [
        {"number": "0", "start_time": "00:00:01,000", "end_time": "00:00:02,000",
         "text": "first", "original_text": "first"},
        {"number": "1", "start_time": "00:00:03,000", "end_time": "00:00:04,000",
         "text": "last", "original_text": "last"},
    ]

    updated = processor.update_translated_subtitles(subtitles, {-1: "INTRUDER"})

    assert updated[-1]["text"] == "last"


def test_refine_after_reuses_translation_subtitle_blocks():
    from src.core.refine.srt_refiner import _blocks_from_translation_checkpoint

    subtitles = [
        {"text": f"cue {index}"}
        for index in range(5)
    ]
    rows = [
        {
            "chunk_index": 0,
            "status": "completed",
            "chunk_data": {"block_subtitles": [0, 1, 2]},
        },
        {
            "chunk_index": 1,
            "status": "completed",
            "chunk_data": {"block_subtitles": [3, 4]},
        },
    ]

    blocks = _blocks_from_translation_checkpoint(subtitles, rows)

    assert [[cue["text"] for cue in block] for block in blocks] == [
        ["cue 0", "cue 1", "cue 2"],
        ["cue 3", "cue 4"],
    ]


def test_invalid_checkpoint_blocks_fall_back_to_configured_grouping():
    from src.core.refine.srt_refiner import _blocks_from_translation_checkpoint

    subtitles = [{"text": "one"}, {"text": "two"}]
    rows = [{
        "chunk_index": 0,
        "status": "completed",
        "chunk_data": {"block_subtitles": [0, 0]},
    }]

    assert _blocks_from_translation_checkpoint(subtitles, rows) == []
