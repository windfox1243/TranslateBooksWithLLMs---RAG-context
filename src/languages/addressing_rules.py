"""Per-language hooks for UniversalAddressingEngine.

The engine owns the language-neutral part of validating a directed address
pair (speaker -> addressee): read the genders, resolve who is senior, and
apply a table of known pair repairs. Everything that only makes sense in
one target language lives in an AddressingRules subclass in that
language's pack, and the engine calls it at fixed points:

1. before_hierarchy: rewrites that do not depend on seniority
   (Vietnamese: a job title used as a pronoun, parent-child forms);
2. junior_to_senior / senior_to_junior: what a pair must look like once the
   direction is known;
3. after_hierarchy: checks against the addressee's gender;
4. after_table: guards that run after the pair-repair table.

A language with no rules here only gets the pronoun classes and the repair
table; nothing is rewritten into another language's pronouns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set, Tuple


@dataclass
class PairState:
    """A pair being repaired, plus what the engine already read about it.

    `self_form` / `target_form` / `vocative` are the values being repaired;
    the boolean gender flags and `context` (casefolded details) are inputs.
    """

    self_form: str
    target_form: str
    vocative: str
    speaker: str = ""
    addressee: str = ""
    context: str = ""
    speaker_female: bool = False
    speaker_male: bool = False
    addressee_female: bool = False
    addressee_male: bool = False

    @property
    def self_key(self) -> str:
        return self.self_form.casefold()

    @property
    def target_key(self) -> str:
        return self.target_form.casefold()


@dataclass(frozen=True)
class SeniorityCues:
    """Words and phrases that reveal seniority in a pair's free-text context."""

    senior: Tuple[str, ...] = ()
    junior: Tuple[str, ...] = ()
    senior_to_junior_phrases: Tuple[str, ...] = ()
    junior_to_senior_phrases: Tuple[str, ...] = ()


@dataclass
class AddressingRules:
    """Pronoun classes, repair table and hooks for one target language."""

    senior_pronouns: FrozenSet[str] = frozenset()
    junior_pronouns: FrozenSet[str] = frozenset()
    peer_pronouns: FrozenSet[str] = frozenset()
    harmonious_pairs: Dict[Tuple[str, str], Tuple[str, str]] = field(
        default_factory=dict
    )
    seniority_cues: SeniorityCues = field(default_factory=SeniorityCues)

    def before_hierarchy(self, pair: PairState) -> None:
        """Rewrite the pair before seniority is resolved."""

    def junior_to_senior(self, pair: PairState) -> None:
        """Make a junior-to-senior pair coherent."""

    def senior_to_junior(self, pair: PairState) -> None:
        """Make a senior-to-junior pair coherent."""

    def after_hierarchy(self, pair: PairState) -> None:
        """Check the pair against the addressee's gender."""

    def after_table(self, pair: PairState, target_key_before_table: str) -> None:
        """Final guards; receives the target key as it was before the table."""

    def forbidden_pronouns(
        self, self_key: str, target_key: str
    ) -> Tuple[Set[str], Set[str]]:
        """Pronouns a pair rules out, for negative prompt constraints."""
        return set(), set()
