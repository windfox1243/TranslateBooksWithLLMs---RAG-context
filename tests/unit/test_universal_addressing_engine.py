"""
Unit tests for Universal Addressing Constraint Engine across multiple languages.
"""

import pytest
from src.utils.universal_addressing_engine import (
    UniversalAddressingEngine,
    normalize_language_code,
)


def test_normalize_language_code():
    assert normalize_language_code("Vietnamese") == "vi"
    assert normalize_language_code("tiếng việt") == "vi"
    assert normalize_language_code("Japanese") == "ja"
    assert normalize_language_code("Korean") == "ko"
    assert normalize_language_code("French") == "fr"
    assert normalize_language_code("Unknown") == "vi"


def test_vietnamese_intra_pair_repairs():
    engine = UniversalAddressingEngine(language="vi")

    # Identical pronouns: em - em -> em - anh
    s, t, v = engine.validate_and_repair_pair("em", "em", details_context="male speaker")
    assert s == "em"
    assert t == "anh"

    # Identical pronouns: chị - chị -> chị - em
    s, t, v = engine.validate_and_repair_pair("chị", "chị")
    assert s == "chị"
    assert t == "em"

    # Register clash: tớ - mày -> tao - mày
    s, t, v = engine.validate_and_repair_pair("tớ", "mày")
    assert s == "tao"
    assert t == "mày"

    # Register clash: mình - mày -> tao - mày
    s, t, v = engine.validate_and_repair_pair("mình", "mày")
    assert s == "tao"
    assert t == "mày"


def test_vietnamese_trainee_to_trainer_repairs():
    engine = UniversalAddressingEngine(language="vi")

    # Trainee addressing Trainer with peer 'cậu' -> repaired to 'Trainer'
    s, t, v = engine.validate_and_repair_pair(
        "tôi",
        "cậu",
        speaker="Apollo Rainbow",
        addressee="Tomio Momozawa",
        vocative="Trainer",
        details_context="trainer/student hierarchy",
    )
    assert s == "tôi"
    assert t == "Trainer"


def test_japanese_intra_pair_repairs():
    engine = UniversalAddressingEngine(language="ja")

    # Watakushi + Omae -> Ore + Omae
    s, t, v = engine.validate_and_repair_pair("Watakushi", "Omae")
    assert s == "ore"
    assert t == "omae"


def test_korean_intra_pair_repairs():
    engine = UniversalAddressingEngine(language="ko")

    # Jeu + Neo -> Na + Neo
    s, t, v = engine.validate_and_repair_pair("Jeu", "Neo")
    assert s == "na"
    assert t == "neo"
