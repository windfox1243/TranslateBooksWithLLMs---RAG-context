"""The consolidation pass may reword characters; it may not quietly lose them.

Consolidation hands the whole Characters section to the model and writes the
reply back wholesale, so a truncated or careless reply used to delete cast
members and rewrite genders with nothing checking either. Both cost the rest of
the book its pronouns.
"""
import asyncio

import pytest

from src.utils.novel_context import consolidation
from src.utils.novel_context.consolidation import consolidate_context_lore

FOUR_CHARACTERS = """# GLOBAL LORE

## CHARACTERS & GENDERS
- Alice: Female, ship's navigator.
- Bob: Male, deckhand.
- Carol: Female, harbourmaster.
- Dan: Male, cook.

## CHARACTER ALIASES

## NAME TRANSLATION MAP

## GLOSSARY & TERMINOLOGY
"""


class ReplyingClient:
    """An LLM client that returns one canned consolidation reply."""

    def __init__(self, content):
        self.content = content

    async def generate(self, **_kwargs):
        return type("Response", (), {"content": self.content})()


def consolidate(reply, lore=FOUR_CHARACTERS):
    return asyncio.run(
        consolidate_context_lore(
            llm_client=ReplyingClient(reply),
            model_name="test-model",
            global_lore=lore,
        )
    )


def test_losing_half_the_cast_rejects_the_whole_pass():
    reply = (
        "[CHARACTERS]\n"
        "- Alice: Female, ship's navigator.\n"
        "- Bob: Male, deckhand.\n"
    )

    updated, logs = consolidate(reply)

    assert updated == FOUR_CHARACTERS
    for name in ("Alice", "Bob", "Carol", "Dan"):
        assert name in updated
    assert any("rejected" in line and "Carol, Dan" in line for line in logs)


def test_a_prune_within_the_ceiling_is_kept_and_named():
    reply = (
        "[CHARACTERS]\n"
        "- Alice: Female, ship's navigator.\n"
        "- Bob: Male, deckhand.\n"
        "- Carol: Female, harbourmaster.\n"
    )

    updated, logs = consolidate(reply)

    assert "Dan" not in updated
    assert any("removed 1 entries" in line and "Dan" in line for line in logs)


def test_a_gender_flip_without_evidence_is_reverted():
    reply = (
        "[CHARACTERS]\n"
        "- Alice: Female, ship's navigator.\n"
        "- Bob: Female, deckhand.\n"
        "- Carol: Female, harbourmaster.\n"
        "- Dan: Male, cook.\n"
    )

    updated, logs = consolidate(reply)

    assert "- Bob: Male, deckhand." in updated
    assert any("Bob's gender" in line and "kept 'Male'" in line for line in logs)


def test_dropping_a_gender_altogether_is_also_reverted():
    reply = (
        "[CHARACTERS]\n"
        "- Alice: Female, ship's navigator.\n"
        "- Bob: Unspecified, deckhand.\n"
        "- Carol: Female, harbourmaster.\n"
        "- Dan: Male, cook.\n"
    )

    updated, _logs = consolidate(reply)

    assert "- Bob: Male, deckhand." in updated


def test_a_reworded_description_still_goes_through():
    reply = (
        "[CHARACTERS]\n"
        "- Alice: Female, navigator who keeps the only working chart.\n"
        "- Bob: Male, deckhand.\n"
        "- Carol: Female, harbourmaster.\n"
        "- Dan: Male, cook.\n"
    )

    updated, logs = consolidate(reply)

    assert "keeps the only working chart" in updated
    assert not any("rejected" in line for line in logs)


def test_a_character_merged_under_an_identity_link_is_not_a_drop():
    lore = FOUR_CHARACTERS.replace(
        "## CHARACTER ALIASES\n",
        "## CHARACTER ALIASES\n- Carol: Alice\n",
    )
    reply = (
        "[CHARACTERS]\n"
        "- Alice: Female, ship's navigator and harbourmaster.\n"
        "- Bob: Male, deckhand.\n"
        "- Dan: Male, cook.\n"
        "[IDENTITY_LINKS]\n"
        "- Carol: Alice\n"
    )

    updated, logs = consolidate(reply, lore=lore)

    assert "harbourmaster" in updated
    assert not any("rejected" in line for line in logs)


def test_the_ceiling_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(
        consolidation,
        "_consolidation_max_drop_percent",
        lambda: 0,
    )
    reply = "[CHARACTERS]\n- Alice: Female, ship's navigator.\n"

    updated, _logs = consolidate(reply)

    assert "Carol" not in updated


@pytest.mark.parametrize(
    "dropped,original,expected",
    [
        (0, 4, False),
        (1, 4, False),
        (2, 4, True),
        (6, 20, False),
        (7, 20, True),
        # One removal always passes, however short the list.
        (1, 2, False),
        (2, 2, True),
    ],
)
def test_the_ceiling_is_a_percentage_of_the_original_list(
    dropped, original, expected
):
    assert consolidation._exceeds_drop_ceiling(dropped, original) is expected
