"""
Universal Multi-Language Addressing Constraint Engine using 2D Formality + Seniority Matrix.

Enforces strict linguistic separation:
1. self_pronoun MUST be a genuine first-person pronoun (tôi, em, tớ, tao, ta, Watashi, Ore, Boku, Na).
2. target_pronoun MUST be a genuine second-person pronoun (anh, chị, em, thầy, cô, cậu, mày, ngươi, Anata, Omae, Neo).
3. Nouns, job roles, and titles (Trainer, huấn luyện viên, giám đốc, bác sĩ, sensei, sunbae) MUST reside in vocative ONLY.
"""

from typing import Dict, Tuple, Optional, Set, List, Any

# Genuine Seniority Pronoun Sets (Strictly exclude Job Titles / Nouns)
_SENIOR_PRONOUN_SETS: Dict[str, Set[str]] = {
    "vi": {"anh", "chị", "thầy", "cô", "sếp", "bác", "chú", "ông", "bà", "tiền bối", "ngài"},
    "ja": {"senpai", "sensei", "sama"},
    "ko": {"sunbae-nim", "sunbae"},
}

_JUNIOR_PRONOUN_SETS: Dict[str, Set[str]] = {
    "vi": {"em", "cháu", "con", "hậu bối"},
    "ja": {"kohai"},
    "ko": {"hobae"},
}

_PEER_PRONOUN_SETS: Dict[str, Set[str]] = {
    "vi": {"cậu", "bạn", "tớ", "mình", "mày"},
    "ja": {"omae", "kimi", "anta"},
    "ko": {"neo", "inoma"},
}

# Non-pronoun job titles / nouns that MUST be converted to genuine pronouns
_JOB_TITLE_NOUNA: Set[str] = {
    "trainer", "huấn luyện viên", "giám đốc", "bác sĩ", "luật sư", "manager", "doctor",
    "giam doc", "huan luyen vien", "bac si", "luat su", "chủ tịch", "chu tich",
}

# Fast Harmonious Alignment Maps
_HARMONIOUS_ALIGNMENT_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # Vietnamese
    ("em", "em"): ("em", "anh"),
    ("chị", "chị"): ("chị", "em"),
    ("anh", "anh"): ("anh", "em"),
    ("tớ", "mày"): ("tao", "mày"),
    ("mình", "mày"): ("tao", "mày"),
    ("tôi", "ngươi"): ("ta", "ngươi"),
    ("tớ", "ngươi"): ("ta", "ngươi"),
    ("mình", "ngươi"): ("ta", "ngươi"),
    ("tao", "ngài"): ("tôi", "ngài"),
    ("ta", "cậu"): ("ta", "ngươi"),
    ("ta", "anh"): ("tôi", "anh"),
    # Japanese
    ("watakushi", "omae"): ("ore", "omae"),
    ("watakushi", "kisama"): ("ore", "kisama"),
    ("boku", "kisama"): ("ore", "kisama"),
    ("ore", "anata"): ("boku", "anata"),
    # Korean
    ("jeu", "neo"): ("na", "neo"),
    ("jeu", "inoma"): ("na", "inoma"),
    ("na", "dang-sin"): ("jeu", "dang-sin"),
}


class UniversalAddressingEngine:
    """
    2D Formality + Seniority Matrix Engine to validate and repair addressing rules.
    Enforces strict separation between pronouns and job titles/vocatives.
    """

    def __init__(self, language: str = "vi"):
        lang_clean = (language or "vi").strip().casefold()
        if lang_clean in {"vietnamese", "tiếng việt", "vi"}:
            self.lang_code = "vi"
        elif lang_clean in {"japanese", "tiếng nhật", "ja"}:
            self.lang_code = "ja"
        elif lang_clean in {"korean", "tiếng hàn", "ko"}:
            self.lang_code = "ko"
        elif lang_clean in {"french", "français", "fr"}:
            self.lang_code = "fr"
        elif lang_clean in {"spanish", "es"}:
            self.lang_code = "es"
        else:
            self.lang_code = "vi"

    def resolve_seniority_hierarchy(self, speaker: str, addressee: str, context: str) -> str:
        """
        Determine relative seniority: 'JUNIOR_TO_SENIOR', 'SENIOR_TO_JUNIOR', or 'PEER'.
        """
        spk = (speaker or "").casefold()
        adr = (addressee or "").casefold()
        ctx = (context or "").casefold()

        senior_cues = ("trainer", "coach", "mentor", "huấn luyện viên", "teacher", "thầy", "sensei", "sunbae", "sếp", "giám đốc", "senpai")
        junior_cues = ("trainee", "student", "học sinh", "hậu bối", "junior", "kohai", "hobae")

        # Explicit directional context cues (e.g. "trainer to trainee")
        if any(c in ctx for c in ("trainer to trainee", "senior to junior", "teacher to student", "sếp đến nhân viên", "thầy đến trò")):
            return "SENIOR_TO_JUNIOR"
        if any(c in ctx for c in ("trainee to trainer", "junior to senior", "student to teacher", "trò đến thầy")):
            return "JUNIOR_TO_SENIOR"

        # Direct explicit role cues in context or addressee
        if any(k in adr for k in senior_cues) or (any(k in ctx for k in senior_cues) and not any(k in spk for k in senior_cues)):
            if not any(k in spk for k in senior_cues):
                return "JUNIOR_TO_SENIOR"

        if any(k in spk for k in senior_cues) and any(k in adr or k in ctx for k in junior_cues):
            return "SENIOR_TO_JUNIOR"

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
        Validate and repair a directional addressing pair (self_pronoun, target_pronoun, vocative).
        Ensures target_pronoun is strictly a GENUINE PRONOUN (never a job title or noun).
        Cross-validates target pronoun against character_genders metadata.
        Returns (repaired_self, repaired_target, repaired_vocative).
        """
        s_clean = (self_pronoun or "").strip()
        t_clean = (target_pronoun or "").strip()
        v_clean = (vocative or "").strip()

        if not s_clean and not t_clean:
            return s_clean, t_clean, v_clean

        s_key = s_clean.casefold()
        t_key = t_clean.casefold()
        c_clean = (details_context or "").casefold()

        genders = character_genders or {}
        addressee_g_raw = genders.get(addressee, "") or genders.get((addressee or "").casefold(), "")
        is_addressee_female = (
            "female" in addressee_g_raw.casefold()
            or "nữ" in addressee_g_raw.casefold()
            or "gender: female" in c_clean
            or "giới tính: nữ" in c_clean
        )
        is_addressee_male = (
            ("male" in addressee_g_raw.casefold() and "female" not in addressee_g_raw.casefold())
            or ("nam" in addressee_g_raw.casefold() and "nữ" not in addressee_g_raw.casefold())
            or "gender: male" in c_clean
            or "giới tính: nam" in c_clean
        )

        # Strict Rule: Job titles/nouns in target_pronoun MUST be moved to vocative & converted to genuine pronouns
        if t_key in _JOB_TITLE_NOUNA or any(t_key.startswith(job) for job in _JOB_TITLE_NOUNA):
            if not v_clean:
                v_clean = t_clean
            # Determine genuine pronoun replacement
            if "teacher" in c_clean or "giáo viên" in c_clean:
                t_clean = "thầy"
            elif is_addressee_female:
                t_clean = "chị"
            elif "giám đốc" in t_key or "manager" in c_clean or "boss" in c_clean:
                t_clean = "sếp"
            elif is_addressee_male:
                t_clean = "anh"
            else:
                t_clean = "chị" if is_addressee_female else "anh"
            t_key = t_clean.casefold()

        # 1. Resolve 2D Seniority Hierarchy (JUNIOR_TO_SENIOR, SENIOR_TO_JUNIOR, PEER)
        hierarchy = self.resolve_seniority_hierarchy(speaker, addressee, details_context)

        peer_set = _PEER_PRONOUN_SETS.get(self.lang_code, _PEER_PRONOUN_SETS["vi"])
        senior_set = _SENIOR_PRONOUN_SETS.get(self.lang_code, _SENIOR_PRONOUN_SETS["vi"])
        junior_set = _JUNIOR_PRONOUN_SETS.get(self.lang_code, _JUNIOR_PRONOUN_SETS["vi"])

        # 2. Hierarchy Constraint Solver
        if hierarchy == "JUNIOR_TO_SENIOR":
            # Junior calling Senior cannot use peer pronoun 'cậu', 'mày', 'omae', 'neo'
            if t_key in peer_set:
                if "teacher" in c_clean or "giáo viên" in c_clean:
                    t_clean = "thầy"
                elif is_addressee_female:
                    t_clean = "chị"
                else:
                    t_clean = "anh"
                t_key = t_clean.casefold()

            # Junior self-reference when addressing senior must be 'em' (not peer 'tớ'/'mình') in Vietnamese kinship/seniority
            if self.lang_code == "vi" and s_key in {"tớ", "mình"}:
                if t_key in senior_set or any(k in c_clean for k in ("sister", "brother", "senior", "tiền bối", "sư tỷ", "sư huynh", "kinship")):
                    s_clean = "em"
                    s_key = "em"

        elif hierarchy == "SENIOR_TO_JUNIOR":
            # Senior calling Junior cannot address Junior as Senior 'anh'/'chị'/'thầy'
            if t_key in senior_set:
                t_clean = "em"
                t_key = t_clean.casefold()
            # Senior self-reference cannot be junior 'em'
            if s_key in junior_set:
                s_clean = "tôi"
                s_key = s_clean.casefold()

        # Cross-validation: enforce gender alignment of target_pronoun with character_genders
        if is_addressee_female and t_key == "anh" and "teacher" not in c_clean and "giáo viên" not in c_clean and "thầy" not in c_clean:
            t_clean = "chị"
            t_key = "chị"
        elif is_addressee_male and t_key in {"chị", "cô", "bà"}:
            t_clean = "anh"
            t_key = "anh"

        # 3. Check Fast Harmonious Alignment Table
        if (s_key, t_key) in _HARMONIOUS_ALIGNMENT_MAP:
            repaired_s, repaired_t = _HARMONIOUS_ALIGNMENT_MAP[(s_key, t_key)]
            s_clean, t_clean = repaired_s, repaired_t
            s_key, t_key = s_clean.casefold(), t_key.casefold()

        # 4. Self-Consistency Guard: Prevent identical self & target pronouns (e.g., em - em, chị - chị)
        if s_key and s_key == t_key:
            if s_key == "em":
                t_clean = "chị" if is_addressee_female else ("anh" if is_addressee_male or "male" in c_clean else "chị")
            elif s_key in {"chị", "anh"}:
                t_clean = "em"
            else:
                t_clean = v_clean or addressee or t_clean
            t_key = t_clean.casefold()

        return s_clean, t_clean, v_clean

    def get_forbidden_pronouns(self, self_pronoun: str, target_pronoun: str) -> Tuple[Set[str], Set[str]]:
        """
        Derive forbidden self and target pronouns based on active pair rules to enforce negative constraints.
        Returns tuple of (forbidden_self_set, forbidden_target_set).
        """
        s_clean = (self_pronoun or "").strip().casefold()
        t_clean = (target_pronoun or "").strip().casefold()

        forbidden_self: Set[str] = set()
        forbidden_target: Set[str] = set()

        if self.lang_code == "vi":
            junior_set = _JUNIOR_PRONOUN_SETS["vi"]
            senior_set = _SENIOR_PRONOUN_SETS["vi"]

            if s_clean in junior_set and t_clean in senior_set:
                forbidden_self = {"tôi", "tao", "tớ", "mình", "ta"}
                forbidden_target = {"cậu", "mày", "bạn", "ngươi", "em"}
            elif s_clean in senior_set and t_clean in junior_set:
                forbidden_self = {"em", "tớ", "mình", "cháu"}
                forbidden_target = {"anh", "chị", "thầy", "cô", "sếp", "bác", "chú", "ông", "bà", "tiền bối", "ngài"}
            elif s_clean in {"tớ", "mình"} and t_clean == "cậu":
                forbidden_self = {"em", "anh", "chị", "tôi", "tao"}
                forbidden_target = {"mày", "anh", "chị", "em", "ngươi"}
            elif s_clean == "tao" and t_clean == "mày":
                forbidden_self = {"tôi", "tớ", "em", "mình", "cháu"}
                forbidden_target = {"cậu", "anh", "chị", "bạn", "em"}
            elif s_clean == "tôi" and t_clean in senior_set:
                forbidden_self = {"tao", "tớ", "mày"}
                forbidden_target = {"cậu", "mày", "ngươi"}

        return forbidden_self, forbidden_target

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

