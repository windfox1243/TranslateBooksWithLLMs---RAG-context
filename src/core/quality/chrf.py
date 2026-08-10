"""chrF: a character n-gram F-score for comparing a translation to a reference.

Chosen over word-level metrics because the target languages this project cares
about most -- Vietnamese, Chinese, Japanese, Korean -- either tokenize badly or
carry meaning in morphology that whole-word matching throws away. Character
n-grams need no tokenizer and no model, so a score is reproducible offline and
costs nothing to compute.

The implementation follows the standard formulation (Popovic, 2015): whitespace
is discarded, precision and recall are averaged arithmetically over n-gram
orders 1..N, and recall is weighted beta times as heavily as precision.

A chrF score is only meaningful *relative to another run of the same cases*. It
says a change moved the output closer to or further from the reference; it does
not say the output is good.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

DEFAULT_ORDER = 6
DEFAULT_BETA = 2.0


def _char_ngrams(text: str, order: int) -> Counter:
    """Count the character n-grams of one order, ignoring whitespace."""

    stripped = "".join(text.split())
    if len(stripped) < order:
        return Counter()
    return Counter(
        stripped[index:index + order]
        for index in range(len(stripped) - order + 1)
    )


def _overlap(hypothesis: Counter, reference: Counter) -> int:
    """Count n-grams present in both, respecting multiplicity."""

    return sum((hypothesis & reference).values())


def chrf_score(
    hypothesis: str,
    reference: str,
    *,
    order: int = DEFAULT_ORDER,
    beta: float = DEFAULT_BETA,
) -> float:
    """Score one hypothesis against one reference, from 0.0 to 1.0.

    Two empty strings score 1.0: nothing was asked for and nothing was lost.
    One empty string scores 0.0.
    """

    if not "".join(hypothesis.split()) and not "".join(reference.split()):
        return 1.0

    precisions = []
    recalls = []
    for n in range(1, max(1, order) + 1):
        hypothesis_grams = _char_ngrams(hypothesis, n)
        reference_grams = _char_ngrams(reference, n)
        hypothesis_total = sum(hypothesis_grams.values())
        reference_total = sum(reference_grams.values())
        # An order with nothing to match on either side carries no evidence and
        # is skipped, rather than being scored zero and dragging the average
        # down for every short segment.
        if not hypothesis_total and not reference_total:
            continue
        matched = _overlap(hypothesis_grams, reference_grams)
        precisions.append(matched / hypothesis_total if hypothesis_total else 0.0)
        recalls.append(matched / reference_total if reference_total else 0.0)

    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_squared = beta * beta
    return (
        (1.0 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
    )


def corpus_chrf(
    pairs: Iterable[tuple[str, str]],
    *,
    order: int = DEFAULT_ORDER,
    beta: float = DEFAULT_BETA,
) -> float:
    """Average the per-segment scores of (hypothesis, reference) pairs.

    Segment-averaged rather than corpus-aggregated so that one long chapter
    cannot drown out the short exchanges where addressing defects live.
    """

    scores = [
        chrf_score(hypothesis, reference, order=order, beta=beta)
        for hypothesis, reference in pairs
    ]
    return sum(scores) / len(scores) if scores else 0.0
