"""
Universal Multi-Language Addressing Constraint Engine using 2D Formality + Seniority Matrix.

Enforces strict linguistic separation:
1. self_pronoun MUST be a genuine first-person pronoun (tôi, em, tớ, tao, ta, Watashi, Ore, Boku, Na).
2. target_pronoun MUST be a genuine second-person pronoun (anh, chị, em, thầy, cô, cậu, mày, ngươi, Anata, Omae, Neo).
3. Nouns, job roles, and titles (Trainer, huấn luyện viên, giám đốc, bác sĩ, sensei, sunbae) MUST reside in vocative ONLY.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.languages.addressing_rules import AddressingRules, PairState, SeniorityCues
from src.languages.vietnamese.engine_rules import RULES as VIETNAMESE_RULES
from src.utils.language_profiles import get_language_profile


def _has_cue_word(text: str, cue: str) -> bool:
    """Check if the cue is present in the text, ensuring word boundaries for short/single words to avoid false positive substring matches."""
    if len(cue) <= 4 or " " not in cue:
        return bool(re.search(rf"\b{re.escape(cue)}\b", text, re.IGNORECASE))
    return cue.casefold() in text.casefold()


# Seniority cues every language shares: English role words and the romanized
# Japanese/Korean ones. Each language's rules add their own.
_BASE_SENIORITY_CUES = SeniorityCues(
    senior=(
        "trainer", "coach", "mentor", "teacher", "sensei", "sunbae", "senpai",
        "father", "mother", "parent",
    ),
    junior=(
        "trainee", "student", "junior", "kohai", "hobae", "son", "daughter",
        "child",
    ),
    senior_to_junior_phrases=(
        "trainer to trainee", "senior to junior", "teacher to student",
        "parent to child", "father to son", "father to daughter",
        "mother to son", "mother to daughter",
    ),
    junior_to_senior_phrases=(
        "trainee to trainer", "junior to senior", "student to teacher",
        "child to parent", "father-son", "father-daughter", "mother-son",
        "mother-daughter", "parent-child",
    ),
)

# Languages without hooks only get their pronoun classes and repair table.
_JAPANESE_RULES = AddressingRules(
    senior_pronouns=frozenset({"senpai", "sensei", "sama"}),
    junior_pronouns=frozenset({"kohai"}),
    peer_pronouns=frozenset({"omae", "kimi", "anta"}),
    harmonious_pairs={
        ("watakushi", "omae"): ("ore", "omae"),
        ("watakushi", "kisama"): ("ore", "kisama"),
        ("boku", "kisama"): ("ore", "kisama"),
        ("ore", "anata"): ("boku", "anata"),
    },
)
_KOREAN_RULES = AddressingRules(
    senior_pronouns=frozenset({"sunbae-nim", "sunbae"}),
    junior_pronouns=frozenset({"hobae"}),
    peer_pronouns=frozenset({"neo", "inoma"}),
    harmonious_pairs={
        ("jeu", "neo"): ("na", "neo"),
        ("jeu", "inoma"): ("na", "inoma"),
        ("na", "dang-sin"): ("jeu", "dang-sin"),
    },
)

_RULES_BY_LANGUAGE: Dict[str, AddressingRules] = {
    "vi": VIETNAMESE_RULES,
    "ja": _JAPANESE_RULES,
    "ko": _KOREAN_RULES,
    # French and Spanish are recognized but carry no pair rules yet.
    "fr": AddressingRules(),
    "es": AddressingRules(),
}
_GENERIC_RULES = AddressingRules()


def _merge_cues(base: SeniorityCues, extra: SeniorityCues) -> SeniorityCues:
    return SeniorityCues(
        senior=base.senior + extra.senior,
        junior=base.junior + extra.junior,
        senior_to_junior_phrases=(
            base.senior_to_junior_phrases + extra.senior_to_junior_phrases
        ),
        junior_to_senior_phrases=(
            base.junior_to_senior_phrases + extra.junior_to_senior_phrases
        ),
    )


def _read_gender(genders: Dict[str, str], name: str) -> str:
    return (genders.get(name, "") or genders.get((name or "").casefold(), "")).casefold()


class UniversalAddressingEngine:
    """
    Validate and repair directed addressing pairs (speaker -> addressee).

    The engine reads genders, resolves seniority and applies the target
    language's repair table; everything specific to one language comes from
    that language's AddressingRules (see src/languages/addressing_rules.py).
    """

    def __init__(self, language: str = "generic"):
        code = get_language_profile(language).code
        self.rules = _RULES_BY_LANGUAGE.get(code, _GENERIC_RULES)
        self.lang_code = code if code in _RULES_BY_LANGUAGE else "generic"
        self._cues = _merge_cues(_BASE_SENIORITY_CUES, self.rules.seniority_cues)

    def resolve_seniority_hierarchy(
        self, speaker: str, addressee: str, context: str, vocative: str = "",
    ) -> str:
        """
        Determine relative seniority: 'JUNIOR_TO_SENIOR', 'SENIOR_TO_JUNIOR', or 'PEER'.

        The vocative is the strongest directional evidence available here: it is
        what the speaker calls the addressee, so a role word in it describes the
        addressee's standing and nobody else's.
        """
        spk = (speaker or "").casefold()
        adr = (addressee or "").casefold()
        ctx = (context or "").casefold()
        voc = (vocative or "").casefold()
        cues = self._cues

        # Explicit directional context cues (e.g. "trainer to trainee")
        if any(c in ctx for c in cues.senior_to_junior_phrases):
            return "SENIOR_TO_JUNIOR"
        if any(c in ctx for c in cues.junior_to_senior_phrases):
            return "JUNIOR_TO_SENIOR"

        # The vocative names the addressee, so its role word settles the direction
        if any(_has_cue_word(voc, k) for k in cues.senior):
            return "JUNIOR_TO_SENIOR"
        if any(_has_cue_word(voc, k) for k in cues.junior):
            return "SENIOR_TO_JUNIOR"

        # Direct explicit role cues in context or addressee
        speaker_is_senior = any(_has_cue_word(spk, k) for k in cues.senior)
        if any(_has_cue_word(adr, k) for k in cues.senior) and not speaker_is_senior:
            return "JUNIOR_TO_SENIOR"

        if speaker_is_senior and any(
            _has_cue_word(adr, k) or _has_cue_word(ctx, k) for k in cues.junior
        ):
            return "SENIOR_TO_JUNIOR"

        # A social basis such as "student-trainer" names both roles without
        # saying which end is speaking. Reading the senior cue alone made every
        # such pair junior-to-senior, which pointed the repair the wrong way for
        # the trainer's own lines. Direction has to come from the graph or the
        # vocative; free text that carries both roles cannot supply it.
        if (
            not speaker_is_senior
            and any(_has_cue_word(ctx, k) for k in cues.senior)
            and not any(_has_cue_word(ctx, k) for k in cues.junior)
        ):
            return "JUNIOR_TO_SENIOR"

        return "PEER"

    def validate_and_repair_pair(
        self,
        self_pronoun: str,
        target_pronoun: str,
        speaker: str = "",
        addressee: str = "",
        vocative: str = "",
        register: str = "",
        details_context: str = "",
        character_genders: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str, str]:
        """
        Validate and repair a directional addressing pair.
        Returns (repaired_self, repaired_target, repaired_vocative).
        """
        s_clean = (self_pronoun or "").strip()
        t_clean = (target_pronoun or "").strip()
        v_clean = (vocative or "").strip()
        if not s_clean and not t_clean:
            return s_clean, t_clean, v_clean

        context = (details_context or "").casefold()
        genders = character_genders or {}
        speaker_g = _read_gender(genders, speaker)
        addressee_g = _read_gender(genders, addressee)
        pair = PairState(
            self_form=s_clean,
            target_form=t_clean,
            vocative=v_clean,
            speaker=speaker,
            addressee=addressee,
            context=context,
            speaker_female="female" in speaker_g or "nữ" in speaker_g,
            speaker_male=(
                ("male" in speaker_g and "female" not in speaker_g)
                or ("nam" in speaker_g and "nữ" not in speaker_g)
            ),
            addressee_female=(
                "female" in addressee_g
                or "nữ" in addressee_g
                or "gender: female" in context
                or "giới tính: nữ" in context
            ),
            addressee_male=(
                ("male" in addressee_g and "female" not in addressee_g)
                or ("nam" in addressee_g and "nữ" not in addressee_g)
                or "gender: male" in context
                or "giới tính: nam" in context
            ),
        )
        rules = self.rules

        rules.before_hierarchy(pair)

        hierarchy = self.resolve_seniority_hierarchy(
            speaker, addressee, details_context, vocative=pair.vocative,
        )
        if hierarchy == "JUNIOR_TO_SENIOR":
            rules.junior_to_senior(pair)
        elif hierarchy == "SENIOR_TO_JUNIOR":
            rules.senior_to_junior(pair)

        rules.after_hierarchy(pair)

        target_key_before_table = pair.target_key
        repaired = rules.harmonious_pairs.get((pair.self_key, pair.target_key))
        if repaired:
            pair.self_form, pair.target_form = repaired

        rules.after_table(pair, target_key_before_table)

        return pair.self_form, pair.target_form, pair.vocative

    def get_forbidden_pronouns(self, self_pronoun: str, target_pronoun: str) -> Tuple[Set[str], Set[str]]:
        """
        Derive forbidden self and target pronouns based on active pair rules to enforce negative constraints.
        Returns tuple of (forbidden_self_set, forbidden_target_set).
        """
        return self.rules.forbidden_pronouns(
            (self_pronoun or "").strip().casefold(),
            (target_pronoun or "").strip().casefold(),
        )

    def audit_addressing_violations(
        self,
        text: str,
        rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Audit translated text against active addressing rules for negative constraint violations.
        Returns a list of violation descriptors.
        """
        import re
        if not text or not rules:
            return []

        try:
            from src.utils.dialogue_attribution import detect_dialogue_turns
            turns = detect_dialogue_turns(text)
        except ImportError:
            turns = []

        violations: List[Dict[str, Any]] = []

        for r in rules:
            speaker = r.get("speaker_name", "")
            addressee = r.get("addressee_name", "")
            self_p = r.get("self_pronoun", "")
            target_p = r.get("target_pronoun", "")

            if not self_p or not target_p:
                continue

            f_self, f_target = self.get_forbidden_pronouns(self_p, target_p)
            if not f_self and not f_target:
                continue

            for turn in turns:
                cue = turn.get("cue", "")
                cue_lower = cue.lower()

                for forbidden_p in f_self | f_target:
                    pattern = r'\b' + re.escape(forbidden_p) + r'\b'
                    if re.search(pattern, cue_lower):
                        violations.append({
                            "speaker": speaker,
                            "addressee": addressee,
                            "expected": f"self='{self_p}', target='{target_p}'",
                            "forbidden_found": forbidden_p,
                            "cue": cue,
                        })
                        break

        return violations
