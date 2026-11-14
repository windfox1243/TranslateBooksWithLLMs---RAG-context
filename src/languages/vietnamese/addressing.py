"""Vietnamese addressing data: pronoun classes, pair defaults and repairs.

Vietnamese address is a paired social system: the speaker's self-reference
and the form used for the addressee are chosen together from age, kinship
and seniority. The tables below are the vocabulary the context and
addressing layers reason with.

Two families of pronoun classes live here side by side, because two layers
were written against slightly different lists and unifying them changes
behavior:

- ``SENIOR_PRONOUNS`` / ``JUNIOR_PRONOUNS`` / ``PEER_PRONOUNS`` are what
  UniversalAddressingEngine validates and repairs pairs with. The senior
  class includes parental terms (cha, bố, mẹ, ba, má) and "tiền bối"; the
  peer class does not include "tao".
- ``DIRECTION_*_FORMS`` are what the context merge engine uses to classify
  the direction of a stored pair. Its senior class has no parental terms,
  and its peer class includes "tao".
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

# --- UniversalAddressingEngine pronoun classes -----------------------------

SENIOR_PRONOUNS: FrozenSet[str] = frozenset({
    "anh", "chị", "thầy", "cô", "sếp", "bác", "chú", "ông", "bà",
    "cha", "bố", "mẹ", "ba", "má", "tiền bối", "ngài",
})
JUNIOR_PRONOUNS: FrozenSet[str] = frozenset({"em", "cháu", "con", "hậu bối"})
PEER_PRONOUNS: FrozenSet[str] = frozenset({"cậu", "bạn", "tớ", "mình", "mày"})

# --- Context merge engine pair-direction classes ---------------------------

DIRECTION_SENIOR_FORMS: FrozenSet[str] = frozenset({
    "anh", "chị", "thầy", "cô", "sếp", "bác", "chú", "ông", "bà", "ngài",
})
DIRECTION_JUNIOR_FORMS: FrozenSet[str] = frozenset({"em", "cháu", "con", "hậu bối"})
DIRECTION_PEER_FORMS: FrozenSet[str] = frozenset({
    "cậu", "bạn", "tớ", "mình", "tao", "mày",
})
DIRECTION_NEUTRAL_SELF_FORMS: FrozenSet[str] = frozenset({"tôi"})

# --- Pair repairs ----------------------------------------------------------

# (self-reference, second person) -> the coherent pair to use instead.
HARMONIOUS_PAIRS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("em", "em"): ("em", "anh"),
    ("chị", "chị"): ("chị", "em"),
    ("anh", "anh"): ("anh", "em"),
    # Calling someone "anh"/"chị"/"thầy"/"cô" already places the speaker below
    # them, so the peer self-reference contradicts the pair's own second person.
    # This holds whatever the social basis says, and the basis frequently names
    # both roles at once ("trainer-trainee") without saying who is speaking.
    ("tớ", "anh"): ("em", "anh"),
    ("mình", "anh"): ("em", "anh"),
    ("tớ", "chị"): ("em", "chị"),
    ("mình", "chị"): ("em", "chị"),
    ("tớ", "thầy"): ("em", "thầy"),
    ("mình", "thầy"): ("em", "thầy"),
    ("tớ", "cô"): ("em", "cô"),
    ("mình", "cô"): ("em", "cô"),
    ("tớ", "mày"): ("tao", "mày"),
    ("mình", "mày"): ("tao", "mày"),
    ("tôi", "ngươi"): ("ta", "ngươi"),
    ("tớ", "ngươi"): ("ta", "ngươi"),
    ("mình", "ngươi"): ("ta", "ngươi"),
    ("tao", "ngài"): ("tôi", "ngài"),
    ("ta", "cậu"): ("ta", "ngươi"),
    ("ta", "anh"): ("tôi", "anh"),
}


def default_pair(
    hierarchy: str,
    source_gender: str,
    target_gender: str,
) -> Optional[Tuple[str, str]]:
    """Return the (self-reference, second person) pair a known hierarchy implies.

    Only the two cases the evidence can settle are answered: a junior
    addressing a senior uses "em" and the senior's gendered form, and a
    senior addressing a junior uses their own gendered form and "em".
    Anything else returns None so the caller keeps what it has.
    """
    if hierarchy == "source_junior":
        if target_gender == "female":
            return "em", "chị"
        if target_gender == "male":
            return "em", "anh"
    if hierarchy == "source_senior":
        if source_gender == "female":
            return "chị", "em"
        if source_gender == "male":
            return "anh", "em"
    return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def pair_direction(self_pronoun: str, target_pronoun: str) -> str:
    """Classify a stored pair as junior_to_senior, senior_to_junior, peer, same or ""."""
    self_key = _norm(self_pronoun)
    target_key = _norm(target_pronoun)
    if self_key in DIRECTION_JUNIOR_FORMS and target_key in DIRECTION_SENIOR_FORMS:
        return "junior_to_senior"
    if self_key in DIRECTION_SENIOR_FORMS and target_key in DIRECTION_JUNIOR_FORMS:
        return "senior_to_junior"
    if self_key in DIRECTION_PEER_FORMS and target_key in DIRECTION_PEER_FORMS:
        return "peer"
    if self_key in DIRECTION_NEUTRAL_SELF_FORMS and target_key in DIRECTION_PEER_FORMS:
        return "peer"
    if self_key and self_key == target_key:
        return "same"
    return ""
