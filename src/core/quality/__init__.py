"""Offline measurement of translation quality.

Nothing in this package calls a model or touches a job. It turns a set of
(source, reference, produced translation) triples into numbers that can be
compared between two versions of the prompt, the pipeline, or the editor --
which is the one thing the translation path has never been able to answer.
"""

from src.core.quality.chrf import chrf_score, corpus_chrf
from src.core.quality.regression import (
    CaseResult,
    RegressionCase,
    Scorecard,
    compare_to_baseline,
    load_cases,
    run_cases,
    score_case,
)

__all__ = [
    "CaseResult",
    "RegressionCase",
    "Scorecard",
    "chrf_score",
    "compare_to_baseline",
    "corpus_chrf",
    "load_cases",
    "run_cases",
    "score_case",
]
