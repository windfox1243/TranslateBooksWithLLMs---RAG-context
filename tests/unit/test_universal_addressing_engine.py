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


def test_vietnamese_toi_nguoi_context_aware_repairs():
    engine = UniversalAddressingEngine(language="vi")

    # Register mismatch: tôi - ngươi is preserved as tôi - ngươi so novel_context filters it out
    s, t, v = engine.validate_and_repair_pair("tôi", "ngươi", details_context="hostile contempt, enemies")
    assert s == "tôi"
    assert t == "ngươi"


def test_vietnamese_comprehensive_hierarchy_repairs():
    engine = UniversalAddressingEngine(language="vi")

    # 1. Trainee addressing Trainer with peer 'cậu' -> repaired to 'Trainer'
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

    # 2. Student addressing Teacher with peer 'cậu' -> repaired to 'thầy'
    s, t, v = engine.validate_and_repair_pair(
        "tôi",
        "cậu",
        speaker="Student A",
        addressee="Thầy Nam",
        details_context="teacher-student relationship",
    )
    assert t == "thầy"

    # 3. Employee addressing Boss with peer 'cậu' -> repaired to 'sếp'
    s, t, v = engine.validate_and_repair_pair(
        "tôi",
        "cậu",
        speaker="Nhân viên B",
        addressee="Giám đốc C",
        details_context="sếp-nhân viên",
    )
    assert t == "sếp"

    # 4. Senior addressing Junior with senior pronoun 'anh' -> repaired to 'em'
    s, t, v = engine.validate_and_repair_pair(
        "tôi",
        "anh",
        speaker="Thầy Nam",
        addressee="Học sinh A",
        details_context="teacher-student relationship",
    )
    assert t == "em"


def test_japanese_intra_pair_and_hierarchy_repairs():
    engine = UniversalAddressingEngine(language="ja")

    # Watakushi + Omae -> Ore + Omae
    s, t, v = engine.validate_and_repair_pair("Watakushi", "Omae")
    assert s == "ore"
    assert t == "omae"

    # Kohai addressing Senpai with 'Omae' -> repaired to 'Senpai'
    s, t, v = engine.validate_and_repair_pair(
        "Boku",
        "Omae",
        speaker="Kohai A",
        addressee="Senpai B",
        details_context="senpai-kohai relationship",
    )
    assert t == "Senpai"


def test_korean_intra_pair_and_hierarchy_repairs():
    engine = UniversalAddressingEngine(language="ko")

    # Jeu + Neo -> Na + Neo
    s, t, v = engine.validate_and_repair_pair("Jeu", "Neo")
    assert s == "na"
    assert t == "neo"

    # Junior addressing Senior with 'Neo' -> repaired to 'Sunbae-nim'
    s, t, v = engine.validate_and_repair_pair(
        "Jeu",
        "Neo",
        speaker="Ho-bae A",
        addressee="Sunbae B",
        details_context="sunbae-hobae relationship",
    )
    assert t == "Sunbae-nim"
