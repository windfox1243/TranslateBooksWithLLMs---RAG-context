"""
Unit tests for Lean Universal Addressing Constraint Engine.
"""

import pytest
from src.utils.universal_addressing_engine import UniversalAddressingEngine


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


def test_clean_separation_of_vocative_and_pronoun():
    engine = UniversalAddressingEngine(language="vi")

    s, t, v = engine.validate_and_repair_pair(
        "tôi",
        "Trainer",
        speaker="Apollo Rainbow",
        addressee="Tomio Momozawa",
        vocative="Trainer",
    )
    assert s == "tôi"
    assert v == "Trainer"
