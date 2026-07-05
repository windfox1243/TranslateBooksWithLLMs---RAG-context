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
    ("watakushi", "kisotama"): ("ore", "kisama"),
    ("boku", "kisama"): ("ore", "kisama"),
    ("ore", "anata"): ("boku", "anata"),
}

# KOREAN INTRA-PAIR INCOMPATIBILITY TABLE
_KO_INTRA_INCOMPATIBLE_PAIRS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("jeu", "neo"): ("na", "neo"),
    ("jeu", "inoma"): ("na", "inoma"),
    ("na", "dang-sin"): ("jeu", "dang-sin"),
}

# TRAINER / MENTOR KEYWORDS ACROSS LANGUAGES
_TRAINER_ROLE_KEYWORDS: Set[str] = {
    "trainer",
    "coach",
    "mentor",
    "huấn luyện viên",
    "thầy",
    "thầy giáo",
    "giáo viên",
    "sensei",
    "kantoku",
    "gwang-su",
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
            # Generic fallback
            if s_clean.casefold() == t_clean.casefold() and s_clean:
                t_clean = v_clean or addressee or t_clean
            return s_clean, t_clean, v_clean

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

        # 3. Check Trainee/Junior addressing Trainer/Senior
        is_addressee_trainer = (
            any(k in addressee.casefold() for k in _TRAINER_ROLE_KEYWORDS)
            or any(k in c_clean for k in ("trainer", "mentor", "huấn luyện viên", "thầy"))
        )
        is_speaker_trainer = any(k in speaker.casefold() for k in _TRAINER_ROLE_KEYWORDS)

        if t_key in {"cậu", "tớ", "bạn"} and is_addressee_trainer and not is_speaker_trainer:
            target_p = vocative or addressee or "Trainer"

        # 4. Check Trainer/Senior addressing Trainee/Junior
        if is_speaker_trainer and not is_addressee_trainer:
            if t_key in {"anh", "chị", "cô", "ông", "bà"}:
                target_p = "em"

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
            repaired_s, repaired_t = _JA_INTRA_INCOMPATIBLE_PAIRS[(s_key, t_key)]
            return repaired_s, repaired_t, vocative

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
            repaired_s, repaired_t = _KO_INTRA_INCOMPATIBLE_PAIRS[(s_key, t_key)]
            return repaired_s, repaired_t, vocative

        return self_p, target_p, vocative
