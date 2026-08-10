"""A small, fixed corpus that answers "did this change make translation worse?".

The pipeline has plenty of checks that a translation is well-formed and none
that it is good. That gap is why a prompt change, a chunking change, or an
editor downgrade can ship without anyone noticing the prose degraded: the tests
stay green because nothing was ever measuring the output.

This module measures two things that behave very differently:

* similarity to a human reference (`chrf`), which moves gradually and is only
  meaningful compared against an earlier run of the same cases;
* deterministic assertions, which are binary and are where addressing lives --
  if a case says the junior must call the senior `anh`, that is not a matter of
  degree and a near-miss is still a defect.

A case never fails on chrF alone. Similarity drifts for harmless reasons, so a
drop is reported as a regression against a stored baseline rather than as a
verdict on the translation in isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.core.quality.chrf import chrf_score

# How far the mean may fall below the baseline before it is called a
# regression. chrF on short segments is noisy; below this, a difference says
# more about sampling than about the change under test.
DEFAULT_TOLERANCE = 0.02

_WORDLIKE = re.compile(r"^\w[\w\s'’-]*$", re.UNICODE)


@dataclass(frozen=True)
class RegressionCase:
    """One source segment with a reference translation and its hard rules."""

    case_id: str
    source: str
    reference: str
    target_language: str = ""
    must_contain: Sequence[str] = ()
    must_not_contain: Sequence[str] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RegressionCase":
        return cls(
            case_id=str(payload.get("case_id") or payload.get("id") or ""),
            source=str(payload.get("source") or ""),
            reference=str(payload.get("reference") or ""),
            target_language=str(payload.get("target_language") or ""),
            must_contain=tuple(payload.get("must_contain") or ()),
            must_not_contain=tuple(payload.get("must_not_contain") or ()),
            notes=str(payload.get("notes") or ""),
        )


@dataclass(frozen=True)
class CaseResult:
    """What one case scored and which of its hard rules it broke."""

    case_id: str
    chrf: float
    failures: Sequence[str] = ()
    hypothesis: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class Scorecard:
    """The outcome of running the whole corpus once."""

    results: Sequence[CaseResult] = field(default_factory=tuple)

    @property
    def mean_chrf(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.chrf for result in self.results) / len(self.results)

    @property
    def failures(self) -> List[CaseResult]:
        return [result for result in self.results if not result.passed]

    def to_dict(self) -> Dict[str, Any]:
        """Render the shape that gets stored as a baseline."""

        return {
            "mean_chrf": round(self.mean_chrf, 6),
            "case_count": len(self.results),
            "cases": {
                result.case_id: round(result.chrf, 6)
                for result in self.results
            },
        }


def _contains_term(haystack: str, term: str) -> bool:
    """Look for `term`, respecting word edges when the term is a word.

    Vietnamese address terms are short and hide inside longer words -- `em`
    sits in `xem`, `anh` in `nhanh` -- so a naive substring test would report
    an addressing rule as satisfied by an unrelated word.
    """

    if not term:
        return False
    if _WORDLIKE.match(term):
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        return re.search(pattern, haystack, re.IGNORECASE | re.UNICODE) is not None
    return term.lower() in haystack.lower()


def score_case(case: RegressionCase, hypothesis: str) -> CaseResult:
    """Score one produced translation against its case."""

    failures: List[str] = []
    if not hypothesis.strip():
        failures.append("empty translation")
    for term in case.must_contain:
        if not _contains_term(hypothesis, term):
            failures.append(f"missing required term: {term!r}")
    for term in case.must_not_contain:
        if _contains_term(hypothesis, term):
            failures.append(f"forbidden term present: {term!r}")
    return CaseResult(
        case_id=case.case_id,
        chrf=chrf_score(hypothesis, case.reference),
        failures=tuple(failures),
        hypothesis=hypothesis,
    )


def load_cases(path: Any) -> List[RegressionCase]:
    """Read a JSONL corpus, ignoring blank lines and `#` comments.

    JSONL rather than one big JSON document so a corpus can be appended to,
    diffed line by line, and kept outside the repo when the source text is
    copyrighted.
    """

    cases: List[RegressionCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        case = RegressionCase.from_dict(payload)
        if not case.case_id:
            raise ValueError(f"{path}:{number}: case is missing 'case_id'")
        cases.append(case)
    return cases


def run_cases(
    cases: Iterable[RegressionCase],
    translate: Callable[[RegressionCase], str],
) -> Scorecard:
    """Translate every case with `translate` and score the results.

    `translate` is injected rather than imported so the harness can be run
    against a live provider, a recorded set of outputs, or a stub, without the
    scoring code knowing the difference.
    """

    return Scorecard(
        results=tuple(score_case(case, translate(case)) for case in cases)
    )


def compare_to_baseline(
    scorecard: Scorecard,
    baseline: Optional[Dict[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[str]:
    """Report every way this run is worse than the stored baseline.

    Returns human-readable lines; an empty list means nothing regressed. With
    no baseline there is nothing to compare against, which is not a failure --
    the first run is what creates one.
    """

    regressions: List[str] = []
    for result in scorecard.failures:
        for failure in result.failures:
            regressions.append(f"{result.case_id}: {failure}")

    if not baseline:
        return regressions

    previous_mean = float(baseline.get("mean_chrf") or 0.0)
    if scorecard.mean_chrf < previous_mean - tolerance:
        regressions.append(
            f"mean chrF fell from {previous_mean:.4f} to "
            f"{scorecard.mean_chrf:.4f} (tolerance {tolerance:.4f})"
        )

    previous_cases = dict(baseline.get("cases") or {})
    for result in scorecard.results:
        if result.case_id not in previous_cases:
            continue
        before = float(previous_cases[result.case_id])
        # Per-case tolerance is deliberately looser than the corpus one: a
        # single segment swings far more than the mean, and the point of the
        # per-case check is to catch a case that collapsed, not one that drifted.
        if result.chrf < before - (tolerance * 3):
            regressions.append(
                f"{result.case_id}: chrF fell from {before:.4f} to "
                f"{result.chrf:.4f}"
            )
    return regressions
