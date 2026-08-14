"""Aligned source/draft units for the unit-rewrite editor.

The current editor contract asks the model for an exact substring of the draft
and the text to put in its place. That makes the model do two jobs at once:
judge the translation, and extract a span character for character. Only the
first is something a model is needed for -- the second is arithmetic we can do
ourselves, and doing it badly is where the editor loses most of what it finds.
Measured over 31 runs of one book, 14 findings were discarded for
`locator_missing`, `locator_ambiguous`, an unusable replacement or a retry that
answered the wrong question, and not one of them was a disagreement about the
translation.

This module holds the deterministic half of the replacement: split both texts
into units, align them, render them side by side, and turn a rewritten unit
back into exact draft spans by diffing it. A unit id is issued by us and
carries its own offsets, so a locator cannot be wrong; the span the old
contract begs the model for falls out of the rewrite for free.

Nothing here calls a model or reads configuration. The alignment was checked
against one real book: 3,355 units over 28 chunks, 98.6% of them one to one,
none unalignable, and 1,299 rewrites round-tripped exactly through the spans
computed for them.

The aligner reads lengths, not meaning, and that bounds what it can claim. A
draft that drops a short paragraph and a draft that merges two paragraphs
produce the same evidence -- one draft block of about the right length -- so an
unpaired unit is a reliable omission while a merged unit may be one. This is
not the hole it looks like: a merge keeps both source paragraphs on the source
side of the same unit, so an editor reading that line sees the untranslated
sentence sitting beside its draft with nothing corresponding to it. What the
aligner cannot decide, it hands over intact rather than hiding.
"""

from __future__ import annotations

import difflib
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Spread of the length ratio between paragraphs that really are translations
# of each other, in log space. Measured at 0.16 over 3,339 aligned pairs of one
# English-to-Vietnamese book; carried at roughly twice that so a translator who
# expands or condenses a paragraph on purpose is still read as a pairing rather
# than an omission. Pairing cost is the deviation in units of this spread,
# squared -- a plausible pair stays nearly free while an implausible one is
# expensive, which is the whole discrimination the aligner has to make. A
# linear cost cannot make it: a pairing 6 standard deviations out then reads as
# ordinary noise, and a paragraph the draft never produced gets quietly folded
# into its neighbour instead of being reported.
_RATIO_SPREAD = 0.30

# Cost of leaving a unit unpaired. Sits above what an unusual but genuine
# pairing costs (a paragraph translated to twice its length is about 2.6) and
# below what an implausible one costs, so an omission is found without inventing
# omissions out of stylistic variation.
_DROP_PENALTY = 4.5

# Merges and splits are real and common enough to allow -- a translator joining
# two short paragraphs, or breaking a long one -- but they should lose to a
# plain pairing whenever a plain pairing is available at all.
_MERGE_PENALTY = 1.0

# How far off the diagonal the search may wander before the band is abandoned.
# Alignment is overwhelmingly one to one, so the path hugs the diagonal and a
# band makes long chunks cheap; when the band cannot reach the end the whole
# matrix is searched instead, so this costs speed and never correctness.
_MIN_BAND = 48
_BAND_FRACTION = 0.15

# An equal run shorter than this between two edits is not a boundary worth
# keeping. Diffing a rewritten sentence otherwise yields a patch per changed
# word, which is unreadable in the diagnostics and gives the overlap rules a
# dozen adjacent spans to reason about instead of one.
_COALESCE_GAP = 12

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_LINE_BREAK = re.compile(r"\n")


@dataclass(frozen=True)
class TextBlock:
    """One paragraph, with the offsets it occupies in the text it came from."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Unit:
    """One aligned source/draft pair, addressed by an id we issued.

    `draft_start` and `draft_end` are offsets into the whole draft, so a
    rewrite of this unit becomes an exact draft span without anything having to
    be quoted or searched for.
    """

    unit_id: str
    source: str
    draft: str
    source_start: int
    source_end: int
    draft_start: int
    draft_end: int
    source_blocks: int = 1
    draft_blocks: int = 1

    @property
    def is_omission(self) -> bool:
        """Report whether the draft has nothing for this source unit."""

        return self.draft_blocks == 0

    @property
    def is_addition(self) -> bool:
        """Report whether the draft carries a unit the source does not."""

        return self.source_blocks == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source": self.source,
            "draft": self.draft,
            "draft_start": self.draft_start,
            "draft_end": self.draft_end,
        }


def split_blocks(text: str) -> List[TextBlock]:
    """Split text into offset-preserving paragraphs.

    Blank lines are the paragraph boundary that survives translation; sentence
    boundaries do not, because languages disagree about where a sentence ends.
    Text written with no blank line at all -- subtitles, verse, a stripped
    export -- would collapse into a single unit, so it falls back to lines.
    """

    raw = str(text or "")
    blocks = _split_on(raw, _PARAGRAPH_BREAK)
    if len(blocks) <= 1 and "\n" in raw:
        blocks = _split_on(raw, _LINE_BREAK)
    return blocks


def _split_on(raw: str, separator: re.Pattern) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    cursor = 0
    for match in separator.finditer(raw):
        blocks.append(_block(raw, cursor, match.start()))
        cursor = match.end()
    blocks.append(_block(raw, cursor, len(raw)))
    return [block for block in blocks if block is not None]


def _block(raw: str, start: int, end: int) -> Optional[TextBlock]:
    """Trim surrounding whitespace off a block while keeping true offsets."""

    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return TextBlock(start=start, end=end, text=raw[start:end])


def _length_ratio(source: str, draft: str) -> float:
    """Characters of draft per character of source, kept away from zero.

    Languages differ by a lot -- the measured book runs 1.13 -- and the ratio is
    only used to say which pairing is more plausible than another, so a rough
    figure from the texts themselves beats any constant.
    """

    source_length = len(str(source or "").strip())
    draft_length = len(str(draft or "").strip())
    if source_length <= 0 or draft_length <= 0:
        return 1.0
    return max(0.1, min(10.0, draft_length / source_length))


def _pair_cost(source_chars: int, draft_chars: int, ratio: float) -> float:
    if source_chars <= 0 or draft_chars <= 0:
        return _DROP_PENALTY
    deviation = math.log(draft_chars / (source_chars * ratio)) / _RATIO_SPREAD
    return 0.5 * deviation * deviation


def _align_blocks(
    source_blocks: Sequence[TextBlock],
    draft_blocks: Sequence[TextBlock],
    ratio: float,
    band: Optional[int],
) -> Optional[List[Tuple[int, int]]]:
    """Return the cheapest sequence of (source_count, draft_count) moves."""

    rows, columns = len(source_blocks), len(draft_blocks)
    infinity = float("inf")
    cost = [[infinity] * (columns + 1) for _ in range(rows + 1)]
    back: List[List[Optional[Tuple[int, int, int, int]]]] = [
        [None] * (columns + 1) for _ in range(rows + 1)
    ]
    cost[0][0] = 0.0
    slope = (columns / rows) if rows else 0.0
    moves = (
        (1, 1, 0.0),
        (1, 2, _MERGE_PENALTY),
        (2, 1, _MERGE_PENALTY),
        (1, 0, 0.0),
        (0, 1, 0.0),
    )
    for row in range(rows + 1):
        for column in range(columns + 1):
            if cost[row][column] == infinity:
                continue
            if band is not None and abs(column - row * slope) > band:
                continue
            for source_step, draft_step, extra in moves:
                next_row, next_column = row + source_step, column + draft_step
                if next_row > rows or next_column > columns:
                    continue
                source_chars = sum(
                    len(source_blocks[row + offset].text)
                    for offset in range(source_step)
                )
                draft_chars = sum(
                    len(draft_blocks[column + offset].text)
                    for offset in range(draft_step)
                )
                candidate = (
                    cost[row][column]
                    + _pair_cost(source_chars, draft_chars, ratio)
                    + extra
                )
                if candidate < cost[next_row][next_column]:
                    cost[next_row][next_column] = candidate
                    back[next_row][next_column] = (
                        row, column, source_step, draft_step,
                    )

    if cost[rows][columns] == infinity:
        return None
    path: List[Tuple[int, int]] = []
    row, column = rows, columns
    while (row, column) != (0, 0):
        step = back[row][column]
        if step is None:
            return None
        previous_row, previous_column, source_step, draft_step = step
        path.append((source_step, draft_step))
        row, column = previous_row, previous_column
    return list(reversed(path))


def _refined_ratio(
    source_blocks: Sequence[TextBlock],
    draft_blocks: Sequence[TextBlock],
    path: Sequence[Tuple[int, int]],
) -> Optional[float]:
    """Re-estimate the length ratio from the pairs a first pass agreed on.

    The median, not the mean: one paragraph paired badly should not move the
    estimate that decides every other pairing.
    """

    paired: List[float] = []
    merged: List[float] = []
    source_index = draft_index = 0
    for source_step, draft_step in path:
        if source_step and draft_step:
            source_length = sum(
                len(source_blocks[source_index + offset].text)
                for offset in range(source_step)
            )
            draft_length = sum(
                len(draft_blocks[draft_index + offset].text)
                for offset in range(draft_step)
            )
            if source_length and draft_length:
                target = paired if (source_step, draft_step) == (1, 1) else merged
                target.append(draft_length / source_length)
        source_index += source_step
        draft_index += draft_step
    # A one-to-one pair is the aligner saying it was sure. A merge is the
    # aligner having settled for something, and letting the unsure pairings vote
    # drags the estimate toward whichever reading the corrupted first ratio
    # already favoured -- which is how a single omitted paragraph stays hidden.
    # Merges are consulted only when nothing was paired outright.
    ratios = paired or merged
    if not ratios:
        return None
    return max(0.1, min(10.0, statistics.median(ratios)))


def align_units(source_text: str, draft_text: str) -> List[Unit]:
    """Pair source and draft paragraphs into addressable units.

    One-to-one is the overwhelming case, so the aligner exists for the rest:
    merges, splits, and paragraphs the draft never produced. A unit with no
    draft side is an omission found without asking a model anything; a merged
    unit may also be one, and keeps both source paragraphs so the question
    reaches the editor rather than being decided here on evidence that cannot
    settle it.
    """

    source_blocks = split_blocks(source_text)
    draft_blocks = split_blocks(draft_text)
    if not source_blocks and not draft_blocks:
        return []

    band = max(_MIN_BAND, math.ceil(
        _BAND_FRACTION * max(len(source_blocks), len(draft_blocks))
    ))

    def solve(ratio: float) -> Optional[List[Tuple[int, int]]]:
        found = _align_blocks(source_blocks, draft_blocks, ratio, band)
        if found is None:
            # The draft moved further from the source than the band allows,
            # which is exactly the case worth aligning properly rather than
            # giving up on.
            found = _align_blocks(source_blocks, draft_blocks, ratio, None)
        return found

    path = solve(_length_ratio(source_text, draft_text))
    if path is None:
        return []
    # The ratio taken from the two texts whole is wrong in precisely the case
    # the aligner exists for: a draft missing a paragraph looks like a draft
    # that condensed everything, which then makes the omission read as ordinary
    # compression and hides it. Re-measuring on the pairs the first pass is
    # confident about breaks that circle.
    refined = _refined_ratio(source_blocks, draft_blocks, path)
    if refined is not None:
        path = solve(refined) or path

    units: List[Unit] = []
    source_index = draft_index = 0
    for source_step, draft_step in path:
        source_slice = source_blocks[source_index:source_index + source_step]
        draft_slice = draft_blocks[draft_index:draft_index + draft_step]
        source_index += source_step
        draft_index += draft_step
        units.append(Unit(
            unit_id=f"U-{len(units) + 1:04d}",
            source="\n\n".join(block.text for block in source_slice),
            draft="\n\n".join(block.text for block in draft_slice),
            source_start=source_slice[0].start if source_slice else -1,
            source_end=source_slice[-1].end if source_slice else -1,
            draft_start=draft_slice[0].start if draft_slice else -1,
            draft_end=draft_slice[-1].end if draft_slice else -1,
            source_blocks=len(source_slice),
            draft_blocks=len(draft_slice),
        ))
    return units


def format_bitext(units: Sequence[Unit], *, source_available: bool = True) -> str:
    """Render aligned units as the side-by-side view the editor reads.

    The editor is the only pass that reads the source beside the draft, and
    until now it was handed two separate walls of text and left to align them
    in its head before it could judge anything. That is the hardest part of the
    task and the part we can simply do for it.
    """

    lines: List[str] = []
    for unit in units:
        lines.append(f"[{unit.unit_id}]")
        if source_available:
            lines.append(f"  SRC   {unit.source or '(missing)'}")
        lines.append(f"  DRAFT {unit.draft or '(missing)'}")
    return "\n".join(lines)


def _coalesce(opcodes: Sequence[Tuple[str, int, int, int, int]]) -> List[
    Tuple[int, int, int, int]
]:
    """Merge edits separated by only a few unchanged characters."""

    merged: List[List[int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if merged and i1 - merged[-1][1] <= _COALESCE_GAP:
            merged[-1][1] = i2
            merged[-1][3] = j2
            continue
        merged.append([i1, i2, j1, j2])
    return [tuple(item) for item in merged]


def unit_rewrite_patches(
    unit: Unit,
    rewritten: str,
    issue_id: str = "issue",
) -> List[Tuple[int, int, str, str]]:
    """Turn a rewritten unit into exact draft spans.

    Returns the `(start, end, replacement, issue_id)` tuples the existing patch
    application already speaks, so the overlap, conflict and no-op rules are
    unchanged -- only where the spans come from changes. They are computed from
    the rewrite rather than quoted by the model, so they cannot fail to be
    found in the draft.
    """

    original = str(unit.draft or "")
    replacement_text = str(rewritten or "")
    if not original or original == replacement_text:
        return []
    matcher = difflib.SequenceMatcher(
        None, original, replacement_text, autojunk=False,
    )
    return [
        (
            unit.draft_start + start,
            unit.draft_start + end,
            replacement_text[new_start:new_end],
            issue_id,
        )
        for start, end, new_start, new_end in _coalesce(matcher.get_opcodes())
    ]


def rewrite_change_ratio(original: str, rewritten: str) -> float:
    """Report how much of a unit a rewrite touched, from 0.0 to 1.0.

    Handed a whole unit, a model will improve what was not being asked about,
    and a rewrite that rephrases four sentences to repair one is not the repair
    that was reported. The old contract could not measure this at all, because
    a span edit only ever showed the span it admitted to.
    """

    source = str(original or "")
    target = str(rewritten or "")
    if not source:
        return 1.0 if target else 0.0
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    changed = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    return min(1.0, changed / len(source))
