"""The Vietnamese language pack is the one home of Vietnamese addressing data."""

import pytest

from src.languages.vietnamese import addressing, prompting
from src.prompts import prompts
from src.utils import universal_addressing_engine as engine


@pytest.mark.parametrize(
    "hierarchy, source_gender, target_gender, expected",
    [
        ("source_junior", "", "female", ("em", "chị")),
        ("source_junior", "", "male", ("em", "anh")),
        ("source_senior", "female", "", ("chị", "em")),
        ("source_senior", "male", "", ("anh", "em")),
        ("peer", "male", "female", None),
        ("source_junior", "", "", None),
    ],
)
def test_default_pair(hierarchy, source_gender, target_gender, expected):
    assert addressing.default_pair(hierarchy, source_gender, target_gender) == expected


@pytest.mark.parametrize(
    "self_ref, second, expected",
    [
        ("em", "anh", "junior_to_senior"),
        ("Chị", " em ", "senior_to_junior"),
        ("tớ", "cậu", "peer"),
        ("tôi", "cậu", "peer"),
        ("anh", "anh", "same"),
        ("", "", ""),
    ],
)
def test_pair_direction(self_ref, second, expected):
    assert addressing.pair_direction(self_ref, second) == expected


def test_engine_reads_its_vietnamese_tables_from_the_pack():
    rules = engine.UniversalAddressingEngine("vi").rules
    assert rules.senior_pronouns == addressing.SENIOR_PRONOUNS
    assert rules.harmonious_pairs == addressing.HARMONIOUS_PAIRS


def test_prompt_builders_use_the_pack_text():
    assert prompts._build_target_language_style_section("vi") == prompting.STYLE_GUARDRAILS


def test_style_guardrails_name_no_characters_from_a_specific_book():
    for name in ("Pascal", "Philly", "Frondier"):
        assert name not in prompting.STYLE_GUARDRAILS
