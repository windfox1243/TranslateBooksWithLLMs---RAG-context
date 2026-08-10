"""The quality harness has to be trustworthy before its numbers mean anything."""

import json
from pathlib import Path

import pytest

from src.core.quality.chrf import chrf_score, corpus_chrf
from src.core.quality.regression import (
    RegressionCase,
    Scorecard,
    compare_to_baseline,
    load_cases,
    run_cases,
    score_case,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "quality_regression"
    / "en_vi_baseline.jsonl"
)


def _case(**overrides):
    values = {
        "case_id": "c1",
        "source": "He waited.",
        "reference": "Anh đã đợi.",
    }
    values.update(overrides)
    return RegressionCase(**values)


def test_a_perfect_match_scores_one():
    assert chrf_score("Anh đã đợi.", "Anh đã đợi.") == pytest.approx(1.0)


def test_unrelated_text_scores_near_zero():
    assert chrf_score("Trời mưa", "Kobayashi") < 0.1


def test_two_empty_strings_are_a_match():
    """Nothing was asked for and nothing was lost."""

    assert chrf_score("", "   ") == pytest.approx(1.0)


def test_a_missing_translation_scores_zero():
    assert chrf_score("", "Anh đã đợi.") == pytest.approx(0.0)


def test_whitespace_is_not_part_of_the_score():
    """Line wrapping must not register as a quality change."""

    assert chrf_score("Anh  đã\nđợi.", "Anh đã đợi.") == pytest.approx(1.0)


def test_a_near_miss_beats_an_unrelated_answer():
    reference = "Cô chưa từng một lần nhờ anh giúp đỡ."
    near = "Cô chưa từng nhờ anh giúp đỡ."
    far = "Trời mưa như trút nước khi chúng tôi đến cổng."
    assert chrf_score(near, reference) > chrf_score(far, reference)


def test_corpus_score_averages_the_segments():
    pairs = [("Anh đã đợi.", "Anh đã đợi."), ("", "Anh đã đợi.")]
    assert corpus_chrf(pairs) == pytest.approx(0.5)


def test_a_required_term_must_be_a_whole_word():
    """`em` hides inside `xem`, so substring matching would pass a broken case."""

    case = _case(reference="Anh xem đi.", must_contain=("em",))
    assert "missing required term" in score_case(case, "Anh xem đi.").failures[0]


def test_a_forbidden_term_inside_a_longer_word_is_not_a_violation():
    case = _case(reference="Anh nhanh lên.", must_not_contain=("anh",))
    result = score_case(case, "Nhanh lên.")
    assert result.passed


def test_a_forbidden_term_is_reported():
    case = _case(must_not_contain=("chị",))
    result = score_case(case, "Chị đã đợi.")
    assert not result.passed
    assert "forbidden term" in result.failures[0]


def test_an_empty_translation_fails_regardless_of_score():
    assert not score_case(_case(), "   ").passed


def test_multi_word_terms_are_matched():
    case = _case(reference="Bác sĩ Phạm đến.", must_contain=("Bác sĩ",))
    assert score_case(case, "Bác sĩ Phạm đến.").passed


def test_the_shipped_corpus_agrees_with_itself():
    """Each reference must satisfy its own rules, or the case is mis-specified."""

    cases = load_cases(FIXTURE)
    assert len(cases) >= 20
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        result = score_case(case, case.reference)
        assert result.passed, f"{case.case_id}: {result.failures}"
        assert result.chrf == pytest.approx(1.0)


def test_comments_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '# a note\n\n{"case_id": "one", "source": "a", "reference": "b"}\n',
        encoding="utf-8",
    )
    assert [case.case_id for case in load_cases(path)] == ["one"]


def test_a_case_without_an_id_is_rejected(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source": "a", "reference": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="case_id"):
        load_cases(path)


def test_a_malformed_line_names_its_line_number(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"case_id": "one"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_cases(path)


def test_the_runner_scores_whatever_the_translator_returns():
    cases = [_case(case_id="a"), _case(case_id="b", reference="Cô đã đợi.")]
    scorecard = run_cases(cases, lambda case: case.reference)
    assert [result.case_id for result in scorecard.results] == ["a", "b"]
    assert scorecard.mean_chrf == pytest.approx(1.0)
    assert scorecard.failures == []


def test_a_first_run_without_a_baseline_reports_only_hard_failures():
    scorecard = run_cases(
        [_case(must_contain=("anh",))], lambda case: "Cô đã đợi."
    )
    assert compare_to_baseline(scorecard, None) == ["c1: missing required term: 'anh'"]


def test_a_small_drift_is_not_a_regression():
    scorecard = Scorecard(results=run_cases([_case()], lambda c: c.reference).results)
    baseline = {"mean_chrf": 1.01, "cases": {"c1": 1.01}}
    assert compare_to_baseline(scorecard, baseline) == []


def test_a_fallen_mean_is_reported():
    scorecard = run_cases([_case()], lambda case: "Trời mưa như trút nước.")
    regressions = compare_to_baseline(scorecard, {"mean_chrf": 1.0, "cases": {}})
    assert any("mean chrF fell" in line for line in regressions)


def test_a_collapsed_case_is_named():
    scorecard = run_cases([_case()], lambda case: "Trời mưa như trút nước.")
    regressions = compare_to_baseline(
        scorecard, {"mean_chrf": 0.0, "cases": {"c1": 1.0}}
    )
    assert any(line.startswith("c1: chrF fell") for line in regressions)


def test_the_baseline_shape_round_trips():
    scorecard = run_cases([_case()], lambda case: case.reference)
    stored = json.loads(json.dumps(scorecard.to_dict()))
    assert stored["case_count"] == 1
    assert compare_to_baseline(scorecard, stored) == []
