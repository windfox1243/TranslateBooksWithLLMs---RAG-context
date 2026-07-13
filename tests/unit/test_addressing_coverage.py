"""Regression tests for high-confidence dialogue addressing coverage."""

from types import SimpleNamespace

import pytest

from src.utils.novel_context import (
    _missing_addressing_requirements,
    _retry_missing_addressing_candidates,
)


def _attribution(cue="Thanks, Apollo-chan!"):
    return {
        "turns": [
            {
                "id": "dlg-1",
                "cue": cue,
                "speaker": "Guri",
                "addressee": "Apollo Rainbow",
                "confidence": 0.96,
            }
        ]
    }


def test_addressing_coverage_requires_named_high_confidence_dialogue_pair():
    missing = _missing_addressing_requirements(
        _attribution(), [], "", "English"
    )
    assert missing == [
        {
            "dialogue_turn_id": "dlg-1",
            "speaker": "Guri",
            "addressee": "Apollo Rainbow",
            "evidence_quote": "Thanks, Apollo-chan!",
            "confidence": 0.96,
        }
    ]


def test_addressing_coverage_ignores_uncertain_or_non_directed_relationships():
    uncertain = _attribution("Apollo entered the room.")
    uncertain["turns"][0]["confidence"] = 0.79
    assert not _missing_addressing_requirements(uncertain, [], "", "English")

    dialogue_without_direct_address = _attribution("Thanks again!")
    assert not _missing_addressing_requirements(
        dialogue_without_direct_address, [], "", "English"
    )


def test_addressing_coverage_accepts_existing_candidate():
    candidate = SimpleNamespace(speaker="Guri", addressee="Apollo Rainbow")
    assert not _missing_addressing_requirements(
        _attribution(), [candidate], "", "English"
    )


@pytest.mark.asyncio
async def test_addressing_coverage_retry_parses_only_missing_pair():
    class Client:
        async def generate(self, **kwargs):
            assert "Guri" in kwargs["prompt"]
            return SimpleNamespace(
                content='{"updates":[{'
                '"speaker":"Guri","addressee":"Apollo Rainbow",'
                '"source_forms":[{"text":"Apollo-chan",'
                '"usage":"direct_address",'
                '"evidence_quote":"Thanks, Apollo-chan!"}],'
                '"target_form":{"self_reference":"mình",'
                '"second_person":"cậu","vocative":"Apollo"},'
                '"evidence_quote":"Thanks, Apollo-chan!",'
                '"dialogue_turn_id":"dlg-1","confidence":0.96}]}'
            )

    candidates, status = await _retry_missing_addressing_candidates(
        Client(),
        _missing_addressing_requirements(_attribution(), [], "", "English"),
        "English",
        "Vietnamese",
    )
    assert status == "json"
    assert [(item.speaker, item.addressee) for item in candidates] == [
        ("Guri", "Apollo Rainbow")
    ]
