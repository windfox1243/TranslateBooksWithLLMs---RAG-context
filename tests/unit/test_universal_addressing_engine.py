"""
Unit tests for SOTA Formality Distance Arithmetic Addressing Engine.
"""

import pytest
from src.utils.universal_addressing_engine import UniversalAddressingEngine


def test_formality_score_calculation():
    engine = UniversalAddressingEngine(language="vi")

    assert engine.get_formality_score("ngài") == 2
    assert engine.get_formality_score("tôi") == 1
    assert engine.get_formality_score("tớ") == 0
    assert engine.get_formality_score("mày") == -2

    # Distance calculation: |F(tớ) - F(mày)| = |0 - (-2)| = 2
    assert engine.calculate_formality_distance("tớ", "mày") == 2

    # Distance calculation: |F(tôi) - F(anh)| = |1 - 1| = 0
    assert engine.calculate_formality_distance("tôi", "anh") == 0


def test_vietnamese_formality_distance_repairs():
    engine = UniversalAddressingEngine(language="vi")

    # Identical pronouns: em - em -> em - anh
    s, t, v = engine.validate_and_repair_pair("em", "em", details_context="male speaker")
    assert s == "em"
    assert t == "anh"

    # Register clash (Distance 2): tớ - mày -> tao - mày
    s, t, v = engine.validate_and_repair_pair("tớ", "mày")
    assert s == "tao"
    assert t == "mày"

    # Register clash (Distance 4): tao - ngài -> tôi - ngài
    s, t, v = engine.validate_and_repair_pair("tao", "ngài")
    assert s == "tôi"
    assert t == "ngài"


def test_japanese_formality_distance_repairs():
    engine = UniversalAddressingEngine(language="ja")

    # Watakushi + Omae -> Ore + Omae (Formality distance |2 - (-2)| = 4)
    s, t, v = engine.validate_and_repair_pair("Watakushi", "Omae")
    assert s == "ore"
    assert t == "omae"


def test_korean_formality_distance_repairs():
    engine = UniversalAddressingEngine(language="ko")

    # Jeu + Neo -> Na + Neo
    s, t, v = engine.validate_and_repair_pair("Jeu", "Neo")
    assert s == "na"
    assert t == "neo"
