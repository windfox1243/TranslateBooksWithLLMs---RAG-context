"""salvage_partial_contract keeps the usable part of an invalid editor answer."""

from src.core.editor.contract_retry import salvage_partial_contract
from src.core.translator import ReflectionResult


def _issue(issue_id, replacement="fixed", deterministic=False):
    return {
        "issue_id": issue_id,
        "repair_kind": "local_replace",
        "draft_quote": "old",
        "draft_replacement": replacement,
        "deterministic": deterministic,
    }


def test_valid_issues_are_kept_and_incomplete_ones_counted():
    result = ReflectionResult(
        "needs_repair",
        [_issue("a"), _issue("b", replacement=None), _issue("c", None, deterministic=True)],
        "raw",
        "json",
    )
    salvage = salvage_partial_contract(result)
    assert [i["issue_id"] for i in salvage.reflection_result.issues] == ["a"]
    assert salvage.review_required is True
    assert salvage.review_increment == 1
    assert salvage.warning_increment == 1


def test_only_incomplete_issues_become_review_evidence():
    result = ReflectionResult(
        "needs_repair", [_issue("b", replacement=None)], "raw", "json",
    )
    salvage = salvage_partial_contract(result)
    [kept] = salvage.reflection_result.issues
    assert kept["repair_kind"] == "review_only"
    assert kept["draft_replacement"] is None
    assert salvage.review_required is False
    assert salvage.warning_increment == 0


def test_unparsed_answer_is_left_alone():
    result = ReflectionResult("needs_repair", [_issue("a")], "raw", "invalid_json")
    salvage = salvage_partial_contract(result)
    assert salvage.reflection_result is result
    assert salvage.review_required is None
    assert (salvage.review_increment, salvage.warning_increment) == (0, 0)
