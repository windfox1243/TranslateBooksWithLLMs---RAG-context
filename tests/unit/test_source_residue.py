"""Text left in the source language after translation (issue #270).

A chunk used to count as translated as soon as its placeholders survived, so a
model that skipped a paragraph, or echoed the whole chunk back, produced a
"successful" chunk that no retry ever looked at again. These tests pin:

1. the detector itself (what counts as residue, what is tolerated);
2. the one-shot corrective retry inside generate_translation_request;
3. the placeholder pipeline recording a residue chunk as untranslated;
4. the Plain Text Mode pipeline doing the same;
5. the translation of <head><title>, which the body pipeline never sees.

No network: every LLM call is stubbed.
"""

import pytest

import src.config as config_module
import src.core.translator as core_translator
from src.core.common.source_residue import (
    detect_blocking_residue,
    detect_source_residue,
)

ENGLISH = "He opened the door and walked slowly into the cold night air."


# ---------------------------------------------------------------------------
# 1. Detector
# ---------------------------------------------------------------------------

class TestDetector:

    def test_leftover_korean_sentence_is_flagged(self):
        report = detect_source_residue(
            "[id0]그는 문을 열었다.[id1][id2]바람이 차가웠다.[id3]",
            "[id0]He opened the door.[id1][id2]바람이 차가웠다.[id3]",
            "Korean", "English")
        assert report is not None
        assert report.kind == "script"
        assert "바람이" in report.sample

    def test_kept_korean_name_is_tolerated(self):
        assert detect_source_residue(
            "그는 김민준을 만났다.", "He met 김민준.", "Korean", "English") is None

    def test_latin_brand_names_in_japanese_are_tolerated(self):
        assert detect_source_residue(
            "He met Kim at the Starbucks on Main Street yesterday.",
            "彼は昨日Main Streetのスターバックスでキムに会った。",
            "English", "Japanese") is None

    def test_leftover_russian_sentence_is_flagged(self):
        report = detect_source_residue(
            "Он открыл дверь и медленно вышел в холодную ночь.",
            "Il ouvrit la porte. Он медленно вышел в холодную ночь.",
            "Russian", "French")
        assert report is not None and report.kind == "script"

    def test_echo_is_flagged_for_same_script_pairs(self):
        report = detect_source_residue(ENGLISH, ENGLISH, "English", "French")
        assert report is not None and report.kind == "echo"

    def test_echo_and_latin_residue_are_retry_only(self):
        echo = detect_source_residue(ENGLISH, ENGLISH, "English", "French")
        latin = detect_source_residue(
            ENGLISH, "彼は言った：He opened the door and walked slowly into the cold night air.",
            "English", "Japanese")
        korean = detect_source_residue("바람이 차가웠다.", "바람이 차가웠다.", "Korean", "English")
        assert echo.retry_only and latin.retry_only
        assert korean.blocking
        assert detect_blocking_residue(ENGLISH, ENGLISH, "English", "French") is None

    @pytest.mark.parametrize("text", [
        "Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997.",
        "print('hello world')\nfor item in items: process(item)",
        "Characters: Kim Minjun, Park Seoyeon, Lee Jiho",
    ])
    def test_legitimately_identical_text_never_blocks(self, text):
        assert detect_blocking_residue(text, text, "English", "French") is None

    def test_real_translation_is_accepted(self):
        assert detect_source_residue(
            ENGLISH, "Il ouvrit la porte et s'avança lentement dans l'air froid.",
            "English", "French") is None

    def test_short_identical_text_is_not_an_echo(self):
        assert detect_source_residue("Paris", "Paris", "English", "French") is None

    def test_shared_han_script_is_not_residue(self):
        # Kanji are valid Chinese: only kana would be Japanese residue.
        assert detect_source_residue(
            "彼は扉を開けた。", "他打开了门。", "Japanese", "Chinese (Simplified)") is None

    def test_unknown_language_disables_the_check(self):
        assert detect_source_residue("안녕하세요 여러분 반갑습니다", "안녕하세요 여러분 반갑습니다",
                                     "Korean", "Auto") is None

    def test_same_language_disables_the_check(self):
        assert detect_source_residue(ENGLISH, ENGLISH, "English", "English") is None

    def test_config_switch_disables_the_check(self, monkeypatch):
        monkeypatch.setattr(config_module, "SOURCE_RESIDUE_CHECK_ENABLED", False)
        assert detect_source_residue(ENGLISH, ENGLISH, "English", "French") is None


# ---------------------------------------------------------------------------
# 2. Corrective retry in generate_translation_request
# ---------------------------------------------------------------------------

def _stub_adaptive_request(monkeypatch, answers):
    """Answer successive LLM requests from `answers`, recording prompt_options."""
    seen_options = []

    async def fake(**kwargs):
        seen_options.append(kwargs.get("prompt_options") or {})
        return answers[min(len(seen_options) - 1, len(answers) - 1)], "", None

    monkeypatch.setattr(core_translator, "_make_llm_request_with_adaptive_context", fake)
    return seen_options


async def _request(text, **kwargs):
    return await core_translator.generate_translation_request(
        text, "", "", "", source_language="Korean", target_language="English",
        model="m", llm_client=object(), **kwargs)


class TestCorrectiveRetry:

    @pytest.mark.asyncio
    async def test_residue_triggers_one_retry_with_the_hint(self, monkeypatch):
        seen = _stub_adaptive_request(monkeypatch, [
            "He opened the door. 바람이 차가웠다.",
            "He opened the door. The wind was cold.",
        ])
        result = await _request("그는 문을 열었다. 바람이 차가웠다.")

        assert result == "He opened the door. The wind was cold."
        assert len(seen) == 2
        assert not seen[0].get("source_residue_retry")
        assert seen[1]["source_residue_retry"] is True

    @pytest.mark.asyncio
    async def test_clean_answer_costs_no_extra_request(self, monkeypatch):
        seen = _stub_adaptive_request(monkeypatch, ["He opened the door. The wind was cold."])
        await _request("그는 문을 열었다. 바람이 차가웠다.")
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_worse_retry_keeps_the_first_answer(self, monkeypatch):
        first = "He opened the door. 바람이 차가웠다."
        _stub_adaptive_request(monkeypatch, [first, "그는 문을 열었다. 바람이 차가웠다."])
        assert await _request("그는 문을 열었다. 바람이 차가웠다.") == first

    @pytest.mark.asyncio
    async def test_retry_that_breaks_placeholders_is_discarded(self, monkeypatch):
        first = "[id0]He opened the door. 바람이 차가웠다.[id1]"
        _stub_adaptive_request(monkeypatch, [first, "He opened the door. The wind was cold."])
        result = await _request("[id0]그는 문을 열었다. 바람이 차가웠다.[id1]",
                                has_placeholders=True, placeholder_format=("[id", "]"))
        assert result == first

    def test_retry_hint_reaches_the_user_prompt(self):
        from src.prompts.prompts import generate_translation_prompt
        plain = generate_translation_prompt("x", "", "", "", "Korean", "English")
        hinted = generate_translation_prompt("x", "", "", "", "Korean", "English",
                                             prompt_options={"source_residue_retry": True})
        assert "PREVIOUS ATTEMPT INCOMPLETE" in hinted.user
        assert "PREVIOUS ATTEMPT INCOMPLETE" not in plain.user
        # The cacheable system prompt never changes.
        assert hinted.system == plain.system


# The placeholder-pipeline, Plain Text Mode and <head><title> parts of
# upstream #270 are not ported: this fork has no per-chunk outcome tracking
# for them to record into. Only the corrective retry above is.
