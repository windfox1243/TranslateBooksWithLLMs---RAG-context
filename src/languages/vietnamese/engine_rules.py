"""Vietnamese hooks for UniversalAddressingEngine.

Each method is one step of the engine's pair repair (see
src/languages/addressing_rules.py). The conditions are the ones the engine
used to run inline under `if lang_code == "vi"`.
"""

from __future__ import annotations

from typing import FrozenSet, Set, Tuple

from src.languages.addressing_rules import AddressingRules, PairState, SeniorityCues
from src.languages.vietnamese.addressing import (
    HARMONIOUS_PAIRS,
    JUNIOR_PRONOUNS,
    PEER_PRONOUNS,
    SENIOR_PRONOUNS,
)

# Job titles and nouns that must never stand in the second-person slot.
JOB_TITLE_NOUNS: FrozenSet[str] = frozenset({
    "trainer", "huấn luyện viên", "giám đốc", "bác sĩ", "luật sư", "manager",
    "doctor", "giam doc", "huan luyen vien", "bac si", "luat su", "chủ tịch",
    "chu tich",
})

_PARENT_CHILD_CUES = (
    "father-son", "father-daughter", "mother-son", "mother-daughter",
    "parent-child", "con với cha", "con với mẹ",
)
_FATHER_FORMS = frozenset({"cha", "bố", "ba"})
_MOTHER_FORMS = frozenset({"mẹ", "má"})
_PARENT_FORMS = _FATHER_FORMS | _MOTHER_FORMS
_SENIORITY_CONTEXT_CUES = (
    "sister", "brother", "senior", "tiền bối", "sư tỷ", "sư huynh", "kinship",
)


def _is_teacher_context(context: str) -> bool:
    return "teacher" in context or "giáo viên" in context


class VietnameseAddressingRules(AddressingRules):
    """Vietnamese pair repairs: paired self-reference and address forms."""

    def __init__(self) -> None:
        super().__init__(
            senior_pronouns=SENIOR_PRONOUNS,
            junior_pronouns=JUNIOR_PRONOUNS,
            peer_pronouns=PEER_PRONOUNS,
            harmonious_pairs=HARMONIOUS_PAIRS,
            seniority_cues=SeniorityCues(
                senior=(
                    "huấn luyện viên", "thầy", "sếp", "giám đốc",
                    "cha", "bố", "ba", "mẹ", "má",
                ),
                junior=("học sinh", "hậu bối", "con"),
                senior_to_junior_phrases=("sếp đến nhân viên", "thầy đến trò"),
                junior_to_senior_phrases=(
                    "trò đến thầy", "con với cha", "con với mẹ",
                ),
            ),
        )

    def before_hierarchy(self, pair: PairState) -> None:
        ctx = pair.context

        # A job title in the pronoun slot moves to the vocative, and a genuine
        # pronoun takes its place.
        title_key = pair.target_key
        if title_key in JOB_TITLE_NOUNS or any(
            title_key.startswith(job) for job in JOB_TITLE_NOUNS
        ):
            if not pair.vocative:
                pair.vocative = pair.target_form
            if _is_teacher_context(ctx):
                pair.target_form = "thầy"
            elif pair.addressee_female:
                pair.target_form = "chị"
            elif "giám đốc" in title_key or "manager" in ctx or "boss" in ctx:
                pair.target_form = "sếp"
            elif pair.addressee_male:
                pair.target_form = "anh"
            else:
                pair.target_form = "anh"

        # A child addressing a parent uses the parent's form, and the child
        # calls themself "con".
        self_key = pair.self_key
        if self_key == "con" or any(cue in ctx for cue in _PARENT_CHILD_CUES):
            if pair.target_key in {"anh", "chị", "cậu", "bạn", "mày", "ngươi"}:
                voc = pair.vocative.casefold()
                if voc in _FATHER_FORMS or any(
                    k in ctx for k in ("father", "cha", "bố", "ba")
                ):
                    pair.target_form = (
                        pair.vocative.lower() if voc in _FATHER_FORMS else "cha"
                    )
                elif voc in _MOTHER_FORMS or any(
                    k in ctx for k in ("mother", "mẹ", "má")
                ):
                    pair.target_form = (
                        pair.vocative.lower() if voc in _MOTHER_FORMS else "mẹ"
                    )
                elif voc == "father":
                    pair.target_form = "cha"
                elif voc == "mother":
                    pair.target_form = "mẹ"
                elif pair.addressee_female:
                    pair.target_form = "mẹ"
                else:
                    pair.target_form = "cha"

        if (
            pair.target_key in _PARENT_FORMS
            or pair.vocative.casefold() in (_PARENT_FORMS | {"father", "mother"})
        ):
            if self_key in {"anh", "chị", "tớ", "mày"}:
                pair.self_form = "con"

    def junior_to_senior(self, pair: PairState) -> None:
        # A junior cannot call a senior by a peer pronoun.
        if pair.target_key in self.peer_pronouns:
            if _is_teacher_context(pair.context):
                pair.target_form = "thầy"
            elif pair.addressee_female:
                pair.target_form = "chị"
            else:
                pair.target_form = "anh"
        # Nor call themself by the peer "tớ"/"mình" when the pair or the
        # context is about seniority or kinship.
        if pair.self_key in {"tớ", "mình"}:
            if pair.target_key in self.senior_pronouns or any(
                k in pair.context for k in _SENIORITY_CONTEXT_CUES
            ):
                pair.self_form = "em"

    def senior_to_junior(self, pair: PairState) -> None:
        # A senior cannot address a junior as a senior ("anh"/"chị"/"thầy"),
        # nor as a peer ("cậu"): the peer form denies the seniority just as
        # plainly as the senior form inverts it.
        if (
            pair.target_key in self.senior_pronouns
            or pair.target_key in self.peer_pronouns
        ):
            pair.target_form = "em"
        # The senior's own reference cannot be the junior "em", and a peer
        # "tớ"/"mình" is the same mistake from the other side. The gendered
        # senior form keeps the closeness; "tôi" is the fallback.
        if pair.self_key in self.junior_pronouns:
            pair.self_form = "tôi"
        elif pair.self_key in {"tớ", "mình"}:
            if pair.speaker_female:
                pair.self_form = "chị"
            elif pair.speaker_male:
                pair.self_form = "anh"
            else:
                pair.self_form = "tôi"

    def after_hierarchy(self, pair: PairState) -> None:
        ctx = pair.context
        if (
            pair.addressee_female
            and pair.target_key == "anh"
            and not _is_teacher_context(ctx)
            and "thầy" not in ctx
        ):
            pair.target_form = "chị"
        elif pair.addressee_male and pair.target_key in {"chị", "cô", "bà"}:
            pair.target_form = "anh"

    def after_table(self, pair: PairState, target_key_before_table: str) -> None:
        # Identical self and target forms ("em-em", "chị-chị") are never a
        # pair. The check reads the target as it was before the repair table:
        # the table turns "em-em" into "em-anh" whatever the addressee's
        # gender, and this guard is what chooses "chị" for a woman.
        self_key = pair.self_key
        if not self_key or self_key != target_key_before_table:
            return
        if self_key == "em":
            if pair.addressee_female:
                pair.target_form = "chị"
            elif pair.addressee_male or "male" in pair.context:
                pair.target_form = "anh"
            else:
                pair.target_form = "chị"
        elif self_key in {"chị", "anh"}:
            pair.target_form = "em"
        else:
            pair.target_form = pair.vocative or pair.addressee or pair.target_form

    def forbidden_pronouns(
        self, self_key: str, target_key: str
    ) -> Tuple[Set[str], Set[str]]:
        senior, junior = self.senior_pronouns, self.junior_pronouns
        if self_key in junior and target_key in senior:
            return (
                {"tôi", "tao", "tớ", "mình", "ta"},
                {"cậu", "mày", "bạn", "ngươi", "em"},
            )
        if self_key in senior and target_key in junior:
            return (
                {"em", "tớ", "mình", "cháu"},
                {"anh", "chị", "thầy", "cô", "sếp", "bác", "chú", "ông", "bà",
                 "tiền bối", "ngài"},
            )
        if self_key in {"tớ", "mình"} and target_key == "cậu":
            return (
                {"em", "anh", "chị", "tôi", "tao"},
                {"mày", "anh", "chị", "em", "ngươi"},
            )
        if self_key == "tao" and target_key == "mày":
            return (
                {"tôi", "tớ", "em", "mình", "cháu"},
                {"cậu", "anh", "chị", "bạn", "em"},
            )
        if self_key == "tôi" and target_key in senior:
            return {"tao", "tớ", "mày"}, {"cậu", "mày", "ngươi"}
        return set(), set()


RULES = VietnameseAddressingRules()
