"""
Unit tests for 2D Formality + Seniority Hierarchy Matrix Addressing Engine.
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


def test_vietnamese_2d_seniority_hierarchy_repairs():
    engine = UniversalAddressingEngine(language="vi")

    # 1. Trainee calling Trainer with peer 'cậu' -> Repaired to genuine pronoun 'anh', vocative 'Trainer'
    s, t, v = engine.validate_and_repair_pair(
        self_pronoun="tôi",
        target_pronoun="cậu",
        speaker="Apollo Rainbow",
        addressee="Tomio Momozawa",
        vocative="Trainer",
        details_context="trainer/student hierarchy, trainee",
    )
    assert s == "tôi"
    assert t == "anh"
    assert v == "Trainer"

    # 2. Raw English title 'Trainer' in target_pronoun normalized to genuine pronoun 'anh', vocative 'Trainer'
    s, t, v = engine.validate_and_repair_pair(
        self_pronoun="tôi",
        target_pronoun="Trainer",
        speaker="Apollo Rainbow",
        addressee="Tomio Momozawa",
        vocative="",
    )
    assert t == "anh"
    assert v == "Trainer"

    # 3. Student calling Teacher with peer 'cậu' -> Repaired to 'thầy'
    s, t, v = engine.validate_and_repair_pair(
        self_pronoun="tôi",
        target_pronoun="cậu",
        speaker="Student A",
        addressee="Teacher B",
        details_context="teacher-student relationship",
    )
    assert t == "thầy"

    # 4. Senior (Trainer) calling Junior (Trainee) with Senior pronoun 'anh' -> Repaired to 'em'
    s, t, v = engine.validate_and_repair_pair(
        self_pronoun="tôi",
        target_pronoun="anh",
        speaker="Tomio Momozawa",
        addressee="Apollo Rainbow",
        details_context="trainer to trainee relationship",
    )
    assert s == "tôi"
    assert t == "em"


def test_japanese_2d_seniority_hierarchy_repairs():
    engine = UniversalAddressingEngine(language="ja")

    # Watakushi + Omae -> Ore + Omae
    s, t, v = engine.validate_and_repair_pair("Watakushi", "Omae")
    assert s == "ore"
    assert t == "omae"


def test_korean_2d_seniority_hierarchy_repairs():
    engine = UniversalAddressingEngine(language="ko")

    # Jeu + Neo -> Na + Neo
    s, t, v = engine.validate_and_repair_pair("Jeu", "Neo")
    assert s == "na"
    assert t == "neo"


def test_forbidden_pronouns_and_auditing():
    engine = UniversalAddressingEngine(language="vi")

    # Junior -> Senior (em -> anh) forbidden self includes 'tôi', 'tao'; target forbidden includes 'cậu', 'mày'
    f_self, f_target = engine.get_forbidden_pronouns("em", "anh")
    assert "tôi" in f_self
    assert "mày" in f_target

    rules = [
        {
            "speaker_name": "Aster",
            "addressee_name": "Apollo",
            "self_pronoun": "em",
            "target_pronoun": "anh",
        }
    ]

    sample_text = 'Aster cười nói: “Này cậu, tôi không biết việc này đâu.”'
    violations = engine.audit_addressing_violations(sample_text, rules)

    assert len(violations) > 0
    assert violations[0]["speaker"] == "Aster"
    assert violations[0]["forbidden_found"] in ("cậu", "tôi")

