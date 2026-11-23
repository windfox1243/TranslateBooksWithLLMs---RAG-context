"""A well-tagged answer that is not a translation is rejected and retried.

The generic translator (TXT, SRT, PGN) used to save any well-tagged answer:
a refusal, a cut-off or runaway fragment, or the source copied back. The
answer is now retried with a reinforced prompt; if every attempt is
rejected, the unit keeps its source text and is counted as failed.
(Ported from upstream fe17ae1c.)
"""
import asyncio
import os

import pytest

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.adapters import translate_file
from src.core.answer_acceptance import (
    languages_differ,
    source_may_stay_unchanged,
    tagged_answer_problem,
)
from src.core.llm.base import LLMResponse
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm

LONG_SOURCE = ("The quick brown fox jumps over the lazy dog while the patient translator "
               "carefully preserves the meaning of every clause in the chapter.")


# -- the rules ------------------------------------------------------------


def test_a_real_translation_is_accepted():
    answer = ("Le renard brun rapide saute par-dessus le chien paresseux pendant que le "
              "traducteur patient préserve le sens de chaque proposition du chapitre.")
    assert tagged_answer_problem(answer, LONG_SOURCE, reject_echo=True) is None


def test_a_tagged_refusal_is_rejected():
    reason = tagged_answer_problem("I'm sorry, I can't help with that request.", LONG_SOURCE,
                                   reject_echo=True)
    assert reason and "refusal" in reason


def test_a_short_dialogue_line_that_reads_like_a_refusal_is_accepted():
    # FR -> EN: the translation of a line of the same length is not a refusal.
    assert tagged_answer_problem("I'm sorry, I can't help you.", "Désolé, je ne peux pas t'aider.",
                                 reject_echo=True) is None


def test_a_cut_off_answer_is_rejected():
    assert tagged_answer_problem("Le renard", LONG_SOURCE, reject_echo=True)


def test_the_source_copied_back_is_rejected_when_languages_differ():
    echoed = "  " + LONG_SOURCE.upper().replace(" ", "  ") + "\n"
    assert tagged_answer_problem(echoed, LONG_SOURCE, reject_echo=True) == \
        "the answer is the source text copied back"
    assert tagged_answer_problem(echoed, LONG_SOURCE, reject_echo=False) is None


@pytest.mark.parametrize("source", [
    "1984",
    "...!?",
    "[0]♪ ♪\n[1]♫",
    "[0][MUSIC]\n[1](laughs)",
    "[0]OK",
    "[0]Sherlock Holmes",
    "[0]12. Nf3 Nc6 13. O-O-O Qxd4+",
    "e4 e5 Nf3",
])
def test_legitimately_unchanged_sources_are_exempt_from_the_echo_rule(source):
    assert source_may_stay_unchanged(source)
    assert tagged_answer_problem(source, source, reject_echo=True) is None


@pytest.mark.parametrize("source", [
    "[0]Where are you going tonight?",
    "[0]Hello there\n[1][MUSIC]",
    LONG_SOURCE,
])
def test_ordinary_text_is_not_exempt(source):
    assert not source_may_stay_unchanged(source)


@pytest.mark.parametrize("src,tgt,expected", [
    ("English", "French", True), ("French", "french", False),
    ("", "French", False), ("auto", "French", False),
])
def test_languages_differ(src, tgt, expected):
    assert languages_differ(src, tgt) is expected


# -- end to end through the generic translator ---------------------------------


class _ScriptedProvider(fake_llm.FakeEchoProvider):
    """Answers each translation request with the next scripted reply."""

    replies = []
    prompts = []

    async def generate(self, prompt, timeout=0, system_prompt=None):
        source = fake_llm._extract_source_block(prompt)
        # Custom instructions travel in the system prompt.
        type(self).prompts.append(f"{system_prompt or ''}\n{prompt}")
        reply = type(self).replies.pop(0) if type(self).replies else "translate"
        text = f"FR: {source}" if reply == "translate" else reply
        content = f"{TRANSLATE_TAG_IN}{text}{TRANSLATE_TAG_OUT}"
        return LLMResponse(content=content, prompt_tokens=10, completion_tokens=10,
                           context_used=20, context_limit=self.context_window)


def _install(monkeypatch, replies):
    fake_llm.install(monkeypatch)
    # The acceptance rules are what is under test here.
    monkeypatch.setattr("src.core.answer_acceptance.languages_differ",
                        languages_differ, raising=True)
    _ScriptedProvider.replies = list(replies)
    _ScriptedProvider.prompts = []
    for target in fake_llm._PATCH_TARGETS:
        monkeypatch.setattr(target, lambda *a, **k: _ScriptedProvider(), raising=True)


def _translate_txt(tmp_path):
    source = tmp_path / "one.txt"
    source.write_text(LONG_SOURCE, encoding="utf-8")
    output = tmp_path / "one_fr.txt"
    stats = {}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ok = asyncio.run(translate_file(
            input_filepath=str(source), output_filepath=str(output),
            source_language="English", target_language="French",
            model_name="fake", llm_provider="poe", poe_api_key="UNUSED_FAKE_KEY",
            max_tokens_per_chunk=400, context_window=4096, auto_adjust_context=False,
            stats_callback=stats.update,
            checkpoint_manager=CheckpointManager(db_path=str(tmp_path / "jobs.db")),
            translation_id="cli_tagged_answer",
        ))
    finally:
        os.chdir(cwd)
    return ok, output.read_text(encoding="utf-8"), stats


def test_a_rejected_answer_is_retried_and_the_retry_is_kept(tmp_path, monkeypatch):
    _install(monkeypatch, ["I'm sorry, I can't help with that request.", "translate"])
    ok, text, stats = _translate_txt(tmp_path)
    assert ok is True
    assert text.startswith("FR: ")
    assert "Your previous response was rejected" in _ScriptedProvider.prompts[-1]


def test_an_answer_rejected_every_time_fails_the_unit(tmp_path, monkeypatch):
    _install(monkeypatch, [LONG_SOURCE] * 10)
    ok, text, stats = _translate_txt(tmp_path)
    assert ok is False
    assert stats.get("failed_chunks") == 1
    assert text.strip() == LONG_SOURCE
