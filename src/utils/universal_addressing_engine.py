"""
Universal Multi-Language Addressing Constraint Engine.

Provides O(1) table-driven intra-pair incompatibility filtering, register alignment,
and social hierarchy constraint solving across Vietnamese, Japanese, Korean, Chinese, French, and English.
"""

from typing import Dict, Tuple, Optional, Set, Any
import re

# Canonical language normalizer
_LANGUAGE_MAP: Dict[str, str] = {
    "vietnamese": "vi",
    "vietnamien": "vi",
    "viet": "vi",
    "tiếng việt": "vi",
    "tieng viet": "vi",
    "vi": "vi",
    "japanese": "ja",
    "japonais": "ja",
    "tiếng nhật": "ja",
    "ja": "ja",
    "korean": "ko",
    "coréen": "ko",
    "tiếng hàn": "ko",
    "ko": "ko",
    "chinese": "zh",
    "chinois": "zh",
    "tiếng trung": "zh",
    "zh": "zh",
    "french": "fr",
    "français": "fr",
    "tiếng pháp": "fr",
    "fr": "fr",
    "english": "en",
    "anglais": "en",
    "tiếng anh": "en",
    "en": "en",
}

# VIETNAMESE INTRA-PAIR INCOMPATIBILITY TABLE (Self, Target) -> (Repaired Self, Repaired Target)
_VI_INTRA_INCOMPATIBLE_PAIRS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("em", "em"): ("em", "anh"),
    ("chị", "chị"): ("chị", "em"),
    ("anh", "anh"): ("anh", "em"),
    ("tớ", "mày"): ("tao", "mày"),
    ("mình", "mày"): ("tao", "mày"),
    ("tao", "ngài"): ("tôi", "ngài"),
    ("ta", "cậu"): ("ta", "ngươi"),
    ("ta", "anh"): ("tôi", "anh"),
}

# JAPANESE INTRA-PAIR INCOMPATIBILITY TABLE
_JA_INTRA_INCOMPATIBLE_PAIRS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("watakushi", "omae"): ("ore", "omae"),
    ("watakushi", "kisama"): ("ore", "kisama"),
    ("boku", "kisama"): ("ore", "kisama"),
    ("ore", "anata"): ("boku", "anata"),
}

# KOREAN INTRA-PAIR INCOMPATIBILITY TABLE
_KO_INTRA_INCOMPATIBLE_PAIRS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("jeu", "neo"): ("na", "neo"),
    ("jeu", "inoma"): ("na", "inoma"),
    ("na", "dang-sin"): ("jeu", "dang-sin"),
}

# SENIORITY HIERARCHY CUES (Parsed from relationship details / role names)
_SENIOR_ROLE_KEYWORDS: Set[str] = {
    # Academic / Mentorship
    "trainer", "coach", "mentor", "huấn luyện viên", "thầy", "thầy giáo", "cô giáo", "giáo viên", "sensei", "kantoku", "gwang-su", "professeur",
    # Work / Rank / Hierarchy
    "sếp", "giám đốc", "trưởng phòng", "tiền bối", "senior", "boss", "manager", "master", "chủ nhân", "senpai", "sunbae", "chủ gia tộc",
    # Family / Senior Kinship
    "bố", "cha", "mẹ", "ông", "bà", "bác", "chú", "cô", "dì", "anh", "chị", "older brother", "older sister", "father", "mother"
}

_JUNIOR_ROLE_KEYWORDS: Set[str] = {
    "trainee", "student", "học sinh", "sinh viên", "hậu bối", "junior", "kohai", "ho-bae", "người hầu", "servant", "slave",
    "con", "cháu", "em", "em gái", "em trai", "younger brother", "younger sister", "son", "daughter"
}


def normalize_language_code(language: Optional[str]) -> str:
    """Normalize language name or code to ISO 2-letter code."""
    if not language:
        return "vi"
    clean = str(language).strip().casefold()
    return _LANGUAGE_MAP.get(clean, "vi")


class UniversalAddressingEngine:
    """
    Language-Agnostic Engine to validate, repair, and solve addressing constraints.
    """

    def __init__(self, language: str = "vi"):
        self.lang_code = normalize_language_code(language)

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
        Returns (repaired_self, repaired_target, repaired_vocative).
        """
        s_clean = self_pronoun.strip()
        t_clean = target_pronoun.strip()
        v_clean = vocative.strip()

        if not s_clean and not t_clean:
            return s_clean, t_clean, v_clean

        if self.lang_code == "vi":
            return self._repair_vietnamese(s_clean, t_clean, speaker, addressee, v_clean, register, details_context)
        elif self.lang_code == "ja":
            return self._repair_japanese(s_clean, t_clean, speaker, addressee, v_clean, register, details_context)
        elif self.lang_code == "ko":
            return self._repair_korean(s_clean, t_clean, speaker, addressee, v_clean, register, details_context)
        else:
            if s_clean.casefold() == t_clean.casefold() and s_clean:
                t_clean = v_clean or addressee or t_clean
            return s_clean, t_clean, v_clean

    def _resolve_seniority(self, speaker: str, addressee: str, context: str) -> str:
        """
        Determine relative seniority: 'JUNIOR_TO_SENIOR', 'SENIOR_TO_JUNIOR', or 'PEER'.
        """
        spk = speaker.casefold()
        adr = addressee.casefold()
        ctx = context.casefold()

        # Specific trainer/mentor/senior cues in addressee or context
        senior_cues = ("trainer", "coach", "mentor", "huấn luyện viên", "teacher", "thầy", "sensei", "sunbae", "sếp", "giám đốc", "senpai")
        if any(k in adr or k in ctx for k in senior_cues):
            if not any(k in spk for k in senior_cues):
                return "JUNIOR_TO_SENIOR"

        spk_senior = any(k in spk for k in _SENIOR_ROLE_KEYWORDS)
        adr_senior = any(k in adr for k in _SENIOR_ROLE_KEYWORDS)
        spk_junior = any(k in spk for k in _JUNIOR_ROLE_KEYWORDS)
        adr_junior = any(k in adr for k in _JUNIOR_ROLE_KEYWORDS)

        if adr_senior and not spk_senior:
            return "JUNIOR_TO_SENIOR"
        if spk_senior and not adr_senior:
            return "SENIOR_TO_JUNIOR"
        if spk_junior and adr_senior:
            return "JUNIOR_TO_SENIOR"
        if spk_senior and adr_junior:
            return "SENIOR_TO_JUNIOR"

        return "PEER"

    def _repair_vietnamese(
        self,
        self_p: str,
        target_p: str,
        speaker: str,
        addressee: str,
        vocative: str,
        register: str,
        context: str,
    ) -> Tuple[str, str, str]:
        s_key = self_p.casefold()
        t_key = target_p.casefold()
        v_res = vocative
        c_clean = context.casefold()

        # 1. Check O(1) intra-pair incompatibility table
        if (s_key, t_key) in _VI_INTRA_INCOMPATIBLE_PAIRS:
            repaired_s, repaired_t = _VI_INTRA_INCOMPATIBLE_PAIRS[(s_key, t_key)]
            self_p, target_p = repaired_s, repaired_t
            s_key, t_key = self_p.casefold(), target_p.casefold()

        # 2. Check identical self/target pronouns (e.g., em-em, chị-chị)
        if s_key and s_key == t_key:
            if s_key == "em":
                target_p = "anh" if "male" in c_clean else "chị"
            elif s_key in {"chị", "anh"}:
                target_p = "em"
            else:
                target_p = vocative or addressee or target_p
            t_key = target_p.casefold()

        # 4. Comprehensive Hierarchy Solver (Junior -> Senior and Senior -> Junior)
        seniority = self._resolve_seniority(speaker, addressee, context)

        if seniority == "JUNIOR_TO_SENIOR":
            # Junior calling Senior cannot use peer/junior pronouns ('cậu', 'tớ', 'bạn', 'mày')
            if t_key in {"cậu", "tớ", "bạn", "mày"}:
                # Infer appropriate senior target pronoun based on role/vocative
                if any(k in addressee.casefold() or k in c_clean for k in ("trainer", "coach", "mentor", "huấn luyện viên")):
                    target_p = vocative or "Trainer"
                elif any(k in addressee.casefold() or k in c_clean for k in ("thầy", "giáo viên", "teacher", "professor", "sensei")):
                    target_p = "thầy" if "female" not in c_clean else "cô"
                elif any(k in addressee.casefold() or k in c_clean for k in ("sếp", "giám đốc", "boss", "manager")):
                    target_p = "sếp"
                else:
                    target_p = vocative or "anh" if "male" in c_clean else "chị"
        elif seniority == "SENIOR_TO_JUNIOR":
            # Senior calling Junior cannot use senior pronouns ('anh', 'chị', 'bác', 'chú', 'ông', 'bà')
            if t_key in {"anh", "chị", "bác", "chú", "ông", "bà"}:
                target_p = "em" if "child" not in c_clean else "con"

        return self_p, target_p, v_res

    def _repair_japanese(
        self,
        self_p: str,
        target_p: str,
        speaker: str,
        addressee: str,
        vocative: str,
        register: str,
        context: str,
    ) -> Tuple[str, str, str]:
        s_key = self_p.casefold()
        t_key = target_p.casefold()

        if (s_key, t_key) in _JA_INTRA_INCOMPATIBLE_PAIRS:
            self_p, target_p = _JA_INTRA_INCOMPATIBLE_PAIRS[(s_key, t_key)]
            s_key, t_key = self_p.casefold(), target_p.casefold()

        # Hierarchy repair for Japanese (Junior calling Senior cannot use 'Omae' or 'Kimi')
        seniority = self._resolve_seniority(speaker, addressee, context)
        if seniority == "JUNIOR_TO_SENIOR" and t_key in {"omae", "kimi", "anta"}:
            target_p = vocative or ("Senpai" if "senpai" in context.casefold() else "Sensei")

        return self_p, target_p, vocative

    def _repair_korean(
        self,
        self_p: str,
        target_p: str,
        speaker: str,
        addressee: str,
        vocative: str,
        register: str,
        context: str,
    ) -> Tuple[str, str, str]:
        s_key = self_p.casefold()
        t_key = target_p.casefold()

        if (s_key, t_key) in _KO_INTRA_INCOMPATIBLE_PAIRS:
            self_p, target_p = _KO_INTRA_INCOMPATIBLE_PAIRS[(s_key, t_key)]
            s_key, t_key = self_p.casefold(), target_p.casefold()

        # Hierarchy repair for Korean (Junior calling Senior cannot use 'Neo')
        seniority = self._resolve_seniority(speaker, addressee, context)
        if seniority == "JUNIOR_TO_SENIOR" and t_key in {"neo", "inoma"}:
            target_p = vocative or "Sunbae-nim"

        return self_p, target_p, vocative
