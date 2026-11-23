"""
Completion classifier for finished translation jobs.

A translation loop that ran to the end is not the same thing as a translation
that succeeded. Chunks can fall back to their source text, and the output file
can be missing or empty even though nothing raised. This module turns the raw
job stats plus the state of the output file into a single verdict, so job
finalization has exactly one place that decides between 'completed', 'partial'
and 'error' (a missing or empty output, or a run in which no unit at all got a
translation).

The function is pure apart from two ``os.path`` lookups: no logging, no state
manager, no socket, no checkpoint manager. It is meant to be callable with a
plain dict and a path string from a unit test.
"""
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CompletionVerdict:
    """Outcome of classifying a finished translation job.

    Attributes:
        status: Exactly one of 'completed', 'partial' or 'error'.
        failed_chunks: Number of chunks that failed outright (>= 0).
        fallback_chunks: Number of chunks left as source text (>= 0).
        error: Human-readable reason, non-None if and only if status is 'error'.
        nothing_translated: The job ran to the end but no unit got a
            translation (rule 4). The output exists and the checkpoint is
            kept, so the job is resumable.
    """

    status: str
    failed_chunks: int
    fallback_chunks: int
    error: Optional[str]
    nothing_translated: bool = False


def _nothing_translated(stats: Mapping[str, Any], total: int, failed: int,
                        fallback: int) -> bool:
    """True when not a single unit of the job got a translation.

    Every unit is either still owed (failed, or fallen back to its source
    text) or needed no translation at all (``skipped_chunks``: a text-free
    EPUB chunk passed through without an LLM call). A failed chunk is usually
    also a fallback chunk, so the larger of the two counts is what is owed. A
    job with nothing owed is never "nothing translated", even when every unit
    was skipped: an empty or text-free book is a legitimate success.
    (Ported from upstream bec5fe93.)
    """
    owed = max(failed, fallback)
    if total <= 0 or owed <= 0:
        return False
    skipped = int(stats.get('skipped_chunks') or 0)
    return owed + skipped >= total


def classify_completion(
    stats: Mapping[str, Any],
    output_path: Optional[str],
) -> CompletionVerdict:
    """Classify a finished translation job as completed, partial or errored.

    The rules below are exhaustive and evaluated in order:

    1. Read ``failed_chunks``, ``fallback_used`` and ``total_chunks`` from the
       stats, coercing a missing or ``None`` value to 0.
    2. Missing output: ``output_path`` is falsy, or the file does not exist
       → 'error'.
    3. Empty output: the file exists, is 0 bytes, and ``total_chunks`` > 0
       → 'error'. The ``total_chunks`` > 0 condition is deliberate: an empty
       source file legitimately produces an empty output and stays 'completed'.
    4. Nothing translated: work is owed and not a single unit got a
       translation (every unit failed or fell back to its source text, apart
       from the ``skipped_chunks`` that needed none) → 'error'. The output
       file is 100% source text. The checkpoint is kept, so the job stays
       resumable exactly like a partial one.
    5. Unfinished work: any failed chunk or any fallback chunk → 'partial'.
    6. Otherwise → 'completed'.

    IMPORTANT: ``token_alignment_used`` and ``placeholder_errors`` NEVER
    contribute to the verdict, and must never be added to the rules above.
    Those chunks *are* translated; only their placeholder positions were
    approximated. Counting them would report healthy jobs as degraded.

    ``failed_chunks`` and ``fallback_chunks`` are reported separately and are
    never summed into a single number: "failed outright" and "left as source
    text" are different outcomes for the user.

    Args:
        stats: The job stats payload (any mapping; missing keys are tolerated).
        output_path: Path the job was supposed to write, or None.

    Returns:
        A :class:`CompletionVerdict`.
    """
    failed = int(stats.get('failed_chunks') or 0)
    fallback = int(stats.get('fallback_used') or 0)
    total = int(stats.get('total_chunks') or 0)

    # Rule 2 — the job claims to be done but there is nothing on disk.
    if not output_path or not os.path.exists(output_path):
        return CompletionVerdict(
            status='error',
            failed_chunks=failed,
            fallback_chunks=fallback,
            error=f"Output file was not written: {output_path or '(none)'}",
        )

    # Rule 3 — a 0-byte output is only an error when there was work to do.
    if os.path.getsize(output_path) == 0 and total > 0:
        return CompletionVerdict(
            status='error',
            failed_chunks=failed,
            fallback_chunks=fallback,
            error=f"Output file is empty (0 bytes): {output_path}",
        )

    # Rule 4 — the file is there, but it holds nothing but source text.
    if _nothing_translated(stats, total, failed, fallback):
        return CompletionVerdict(
            status='error',
            failed_chunks=failed,
            fallback_chunks=fallback,
            error=(f"No unit could be translated ({max(failed, fallback)}/{total} failed); "
                   f"the output holds only source text."),
            nothing_translated=True,
        )

    # Rule 5 — the file is there, but part of the content is not translated.
    if failed > 0 or fallback > 0:
        return CompletionVerdict(
            status='partial',
            failed_chunks=failed,
            fallback_chunks=fallback,
            error=None,
        )

    # Rule 6 — nothing left to flag.
    return CompletionVerdict(
        status='completed',
        failed_chunks=failed,
        fallback_chunks=fallback,
        error=None,
    )
