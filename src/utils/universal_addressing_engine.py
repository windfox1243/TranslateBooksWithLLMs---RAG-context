"""
Universal Multi-Language Addressing Constraint Engine using 2D Formality + Seniority Matrix.

Enforces strict linguistic separation:
1. self_pronoun MUST be a genuine first-person pronoun (tôi, em, tớ, tao, ta, Watashi, Ore, Boku, Na).
2. target_pronoun MUST be a genuine second-person pronoun (anh, chị, em, thầy, cô, cậu, mày, ngươi, Anata, Omae, Neo).
3. Nouns, job roles, and titles (Trainer, huấn luyện viên, giám đốc, bác sĩ, sensei, sunbae) MUST reside in vocative ONLY.
"""

from typing import Dict, Tuple, Optional, Set

# Formality Index Mapping F(p) in [-2, +2] across languages
_PRONOUN_FORMALITY_MAP: Dict[str, Dict[str, int]] = {
    "vi": {
        # Honorific / Extremely Formal (+2)
        "ngài": 2, "quý khách": 2, "bệ hạ": 2, "điện hạ": 2, "quý vị": 2, "tiền bối": 2,
        # Polite / Neutral Formal (+1)
        "tôi": 1, "anh": 1, "chị": 1, "ông": 1, "bà": 1, "bác": 1, "chú": 1, "cô": 1, "thầy": 1, "sếp": 1,
        # Youthful / Friendly Peer / Intimate (0)
        "tớ": 0, "cậu": 0, "mình": 0, "bạn": 0, "em": 0, "cháu": 0, "con": 0,
        # Vulgar / Hostile / Contemptuous (-2)
        "tao": -2, "mày": -2, "ngươi": -2, "hắn": -2, "nó": -2,
    },
    "ja": {
        "watakushi": 2, "kochira": 2, "sama": 2, "senpai": 2, "sensei": 2,
        "watashi": 1, "anata": 1, "san": 1,
        "boku": 0, "uchi": 0, "kimi": 0, "kun": 0,
        "ore": -2, "omae": -2, "kisama": -2, "temee": -2,
    },
    "ko": {
        "jeu": 2, "dang-sin": 2, "nim": 2, "sunbae": 2,
        "na": 0, "cheing-gu": 0,
        "neo": -2, "inoma": -2, "gisa-ma": -2,
    },
    "fr": {
        "vous": 1,
        "tu": 0,
    },
    "es": {
        "usted": 1,
        "tú": 0,
    },
}

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

    def get_formality_score(self, pronoun: str) -> int:
        """Get Formality Index F(p) in range [-2, +2]. Defaults to 0 (neutral)."""
        lang_dict = _PRONOUN_FORMALITY_MAP.get(self.lang_code, _PRONOUN_FORMALITY_MAP["vi"])
        return lang_dict.get((pronoun or "").strip().casefold(), 0)

    def calculate_formality_distance(self, self_pronoun: str, target_pronoun: str) -> int:
        """Arithmetic Formality Distance |F(self) - F(target)|."""
        f_s = self.get_formality_score(self_pronoun)
        f_t = self.get_formality_score(target_pronoun)
        return abs(f_s - f_t)

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
    ) -> Tuple[str, str, str]:
        """
        Validate and repair a directional addressing pair (self_pronoun, target_pronoun, vocative).
        Ensures target_pronoun is strictly a GENUINE PRONOUN (never a job title or noun).
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

        # Strict Rule: Job titles/nouns in target_pronoun MUST be moved to vocative & converted to genuine pronouns
        if t_key in _JOB_TITLE_NOUNA or any(t_key.startswith(job) for job in _JOB_TITLE_NOUNA):
            if not v_clean:
                v_clean = t_clean
            # Determine genuine pronoun replacement
            if "teacher" in c_clean or "giáo viên" in c_clean:
                t_clean = "thầy"
            elif "female" in c_clean or "nữ" in c_clean:
                t_clean = "chị"
            elif "giám đốc" in t_key or "manager" in c_clean or "boss" in c_clean:
                t_clean = "sếp"
            else:
                t_clean = "anh"
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
                t_clean = "thầy" if "teacher" in c_clean else ("chị" if "female" in c_clean else "anh")
                t_key = t_clean.casefold()

        elif hierarchy == "SENIOR_TO_JUNIOR":
            # Senior calling Junior cannot address Junior as Senior 'anh'/'chị'/'thầy'
            if t_key in senior_set:
                t_clean = "em"
                t_key = t_clean.casefold()
            # Senior self-reference cannot be junior 'em'
            if s_key in junior_set:
                s_clean = "tôi"
                s_key = s_clean.casefold()

        # 3. Check Fast Harmonious Alignment Table
        if (s_key, t_key) in _HARMONIOUS_ALIGNMENT_MAP:
            repaired_s, repaired_t = _HARMONIOUS_ALIGNMENT_MAP[(s_key, t_key)]
            s_clean, t_clean = repaired_s, repaired_t
            s_key, t_key = s_clean.casefold(), t_key.casefold()

        # 4. Check Arithmetic Formality Distance |F(self) - F(target)|
        else:
            distance = self.calculate_formality_distance(s_clean, t_clean)
            if distance >= 3:
                f_s = self.get_formality_score(s_clean)
                f_t = self.get_formality_score(t_clean)
                if f_s < f_t:
                    s_clean = "tôi" if self.lang_code == "vi" else ("watashi" if self.lang_code == "ja" else "jeu")
                    s_key = s_clean.casefold()

        # 5. Self-Consistency Guard: Prevent identical self & target pronouns (e.g., em - em, chị - chị)
        if s_key and s_key == t_key:
            if s_key == "em":
                t_clean = "anh" if "male" in c_clean else "chị"
            elif s_key in {"chị", "anh"}:
                t_clean = "em"
            else:
                t_clean = v_clean or addressee or t_clean
            t_key = t_clean.casefold()

        return s_clean, t_clean, v_clean
