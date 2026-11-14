"""UniversalAddressingEngine applies only the target language's own rules."""

import pytest

from src.languages.vietnamese.engine_rules import VietnameseAddressingRules
from src.utils.universal_addressing_engine import UniversalAddressingEngine

_VIETNAMESE_PRONOUNS = {"anh", "chị", "em", "thầy", "cô", "cha", "mẹ", "tôi", "con"}


@pytest.mark.parametrize("language", ["Japanese", "Korean"])
@pytest.mark.parametrize(
    "self_ref, second, context",
    [
        ("", "omae", "trainee to trainer"),   # junior addressing senior
        ("", "neo", "trainer to trainee"),    # senior addressing junior
        ("kohai", "sensei", "trainer to trainee"),
    ],
)
def test_no_vietnamese_pronoun_reaches_a_japanese_or_korean_pair(
    language, self_ref, second, context
):
    repaired = UniversalAddressingEngine(language).validate_and_repair_pair(
        self_ref, second, speaker="A", addressee="B", details_context=context,
    )
    assert not set(repaired[:2]) & _VIETNAMESE_PRONOUNS


def test_another_language_repair_table_is_not_applied():
    # ("em", "em") is a Vietnamese pair repair; French must not apply it.
    assert UniversalAddressingEngine("French").validate_and_repair_pair(
        "em", "em"
    ) == ("em", "em", "")
    # ("watakushi", "omae") is a Japanese repair; Vietnamese must not apply it.
    assert UniversalAddressingEngine("Vietnamese").validate_and_repair_pair(
        "watakushi", "omae"
    ) == ("watakushi", "omae", "")


def test_vietnamese_cues_do_not_set_seniority_for_other_languages():
    assert UniversalAddressingEngine("French").resolve_seniority_hierarchy(
        "A", "B", "", vocative="Cha"
    ) == "PEER"
    assert UniversalAddressingEngine("Vietnamese").resolve_seniority_hierarchy(
        "A", "B", "", vocative="Cha"
    ) == "JUNIOR_TO_SENIOR"


@pytest.mark.parametrize(
    "gender, expected",
    [("female", "chị"), ("male", "anh"), ("", "chị")],
)
def test_vietnamese_em_em_pair_is_resolved_by_gender(gender, expected):
    # The repair table maps em-em to em-anh; the identity guard reads the
    # pair from before the table, so the addressee's gender decides.
    engine = UniversalAddressingEngine("Vietnamese")
    genders = {"B": gender} if gender else {}
    assert engine.validate_and_repair_pair(
        "em", "em", speaker="A", addressee="B", character_genders=genders,
    )[:2] == ("em", expected)


def test_vietnamese_job_title_moves_to_the_vocative():
    engine = UniversalAddressingEngine("Vietnamese")
    assert engine.validate_and_repair_pair(
        "em", "Trainer", speaker="A", addressee="B",
        character_genders={"B": "male"},
    ) == ("em", "anh", "Trainer")


def test_forbidden_pronouns_come_from_the_language_rules():
    assert UniversalAddressingEngine("Vietnamese").get_forbidden_pronouns(
        "em", "anh"
    ) == VietnameseAddressingRules().forbidden_pronouns("em", "anh")
    assert UniversalAddressingEngine("Japanese").get_forbidden_pronouns(
        "kohai", "senpai"
    ) == (set(), set())
