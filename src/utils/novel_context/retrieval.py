"""Rank and budget what the prompt renderer injects.

Selection answers "is this entry relevant to this chunk?". That is enough while
everything relevant fits, and stops being enough on a long book: a chapter can
name thirty characters, and the gender roster names every character in the book
on every chunk. Something then has to give way.

Left to itself the renderer gave way in file order and in section order, which
is close to the worst possible policy. File order means the cast discovered
first wins regardless of who this chunk is about. Section order means addressing
-- rendered last, and the reason the whole context system exists for Vietnamese,
Japanese and Korean -- is starved by a roster of characters who are not even in
the scene.

So this module does two things. It scores entries by how strongly this chunk
actually points at them, and it splits the budget so the dynamic state has a
reserve the lore cannot spend.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, TypeVar

from .characters import (
    _entry_mentions_reference,
    _plain_key,
    _reference_mentions_latin_name_part,
    _text_mentions,
)

T = TypeVar("T")

# Scores are ordinal, not measurements: only their order matters.
SCORE_NAMED = 100  # the chunk writes this character's name
SCORE_LATIN_NAME_PART = 80  # a Latin part of the name appears
SCORE_DESCRIBED = 60  # the chunk matches the entry's description
SCORE_ON_STAGE = 40  # named by a dynamic line already selected
SCORE_POV = 30  # first-person narration and this is the viewpoint
SCORE_NONE = 0


def _gender_roster_max() -> int:
    """How many characters the pinned gender roster may name (0 = no cap)."""
    try:
        from src import config as _config

        return max(0, int(getattr(_config, "NOVEL_CONTEXT_GENDER_ROSTER_MAX", 40)))
    except Exception:
        return 40


def _dynamic_budget_percent() -> int:
    """Share of the prompt budget reserved for addressing and relationships."""
    try:
        from src import config as _config

        return min(
            90,
            max(0, int(getattr(_config, "NOVEL_CONTEXT_DYNAMIC_BUDGET_PERCENT", 40))),
        )
    except Exception:
        return 40


def score_entry(
    name: str,
    value: str,
    reference_text: str,
    *,
    on_stage_keys: Optional[set] = None,
    is_pov: bool = False,
) -> int:
    """How strongly this chunk points at one lore entry.

    Checked strongest first so a directly named character never loses to one
    that merely matches a word in its description.
    """
    if _text_mentions(name, reference_text):
        return SCORE_NAMED
    if _reference_mentions_latin_name_part(name, reference_text):
        return SCORE_LATIN_NAME_PART
    if _entry_mentions_reference(name, value, reference_text):
        return SCORE_DESCRIBED
    if on_stage_keys and _plain_key(name) in on_stage_keys:
        return SCORE_ON_STAGE
    if is_pov:
        return SCORE_POV
    return SCORE_NONE


def rank(items: List[T], score_of: Callable[[T], int]) -> List[T]:
    """Order by score, highest first, keeping file order within a score.

    Stable on purpose: entries the chunk points at equally hard stay in the
    order the context file records them, which is roughly discovery order, so
    the ranking never shuffles a prompt for no reason.
    """
    return [
        item
        for _score, _index, item in sorted(
            ((-score_of(item), index, item) for index, item in enumerate(items)),
            key=lambda triple: (triple[0], triple[1]),
        )
    ]


def split_budget(max_chars: int) -> Tuple[int, int]:
    """Return (lore_budget, dynamic_reserve) for one prompt render.

    The reserve is a floor, not a quota: whatever the lore does not spend rolls
    over, so a chunk with two characters in it still gets every addressing rule
    it can use. It only bites when the lore would otherwise take everything.
    """
    if max_chars <= 0:
        return 0, 0
    reserve = max_chars * _dynamic_budget_percent() // 100
    return max(0, max_chars - reserve), reserve


def cap_roster(lines: List[str]) -> Tuple[List[str], int]:
    """Bound the pinned gender roster, returning the kept lines and the cut.

    The roster names every character in the book with a recorded gender, so it
    is the one part of the prompt that grows with the book rather than with the
    chunk. Ranked before it gets here, so the cut falls on the least relevant.
    """
    limit = _gender_roster_max()
    if limit <= 0 or len(lines) <= limit:
        return lines, 0
    return lines[:limit], len(lines) - limit
