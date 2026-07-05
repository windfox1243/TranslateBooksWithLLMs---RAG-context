"""
Universal Multi-Language Addressing Constraint Engine using SOTA Formality Distance Arithmetic.

Calculates arithmetic formality distance |F(self) - F(target)| across languages
and dynamically resolves register incompatibilities and pronoun clashes in O(1) time.
"""

from typing import Dict, Tuple, Optional, Set

# Formality Index Mapping F(p) in [-2, +2] across languages
_PRONOUN_FORMALITY_MAP: Dict[str, Dict[str, int]] = {
    "vi": {
        # Honorific / Extremely Formal (+2)
        "ngài": 2, "quý khách": 2, "bệ hạ": 2, "điện hạ": 2, "quý vị": 2, "tiền bối": 2,
        # Polite / Neutral Formal (+1)
        "tôi": 1, "anh": 1, "chị": 1, "ông": 1, "bà": 1, "bác": 1, "chú": 1, "cô": 1,
        # Youthful / Friendly Peer / Intimate (0)
        "tớ": 0, "cậu": 0, "mình": 0, "bạn": 0, "em": 0, "cháu": 0, "con": 0,
        # Vulgar / Hostile / Contemptuous (-2)
        "tao": -2, "mày": -2, "ngươi": -2, "hắn": -2, "nó": -2,
    },
    "ja": {
        "watakushi": 2, "kochira": 2, "sama": 2,
        "watashi": 1, "anata": 1, "san": 1,
        "boku": 0, "uchi": 0, "kimi": 0, "kun": 0,
        "ore": -2, "omae": -2, "kisama": -2, "temee": -2,
    },
    "ko": {
        "jeu": 2, "dang-sin": 2, "nim": 2,
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
    Formality Distance Arithmetic Engine to validate and repair addressing rules.
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
        s_clean = (self_pronoun or "").strip()
        t_clean = (target_pronoun or "").strip()
        v_clean = (vocative or "").strip()

        if not s_clean and not t_clean:
            return s_clean, t_clean, v_clean

        s_key = s_clean.casefold()
        t_key = t_clean.casefold()
        c_clean = (details_context or "").casefold()

        # 1. Check Fast Harmonious Alignment Table
        if (s_key, t_key) in _HARMONIOUS_ALIGNMENT_MAP:
            repaired_s, repaired_t = _HARMONIOUS_ALIGNMENT_MAP[(s_key, t_key)]
            s_clean, t_clean = repaired_s, repaired_t
            s_key, t_key = s_clean.casefold(), t_clean.casefold()

        # 2. Check Arithmetic Formality Distance |F(self) - F(target)|
        else:
            distance = self.calculate_formality_distance(s_clean, t_clean)
            if distance >= 3:
                f_s = self.get_formality_score(s_clean)
                f_t = self.get_formality_score(t_clean)
                if f_s < f_t:
                    # Target is extremely formal (e.g. tao vs ngài)
                    s_clean = "tôi" if self.lang_code == "vi" else ("watashi" if self.lang_code == "ja" else "jeu")
                    s_key = s_clean.casefold()

        # 3. Self-Consistency Guard: Prevent identical self & target pronouns (e.g., em - em, chị - chị)
        if s_key and s_key == t_key:
            if s_key == "em":
                t_clean = "anh" if "male" in c_clean else "chị"
            elif s_key in {"chị", "anh"}:
                t_clean = "em"
            else:
                t_clean = v_clean or addressee or t_clean
            t_key = t_clean.casefold()

        return s_clean, t_clean, v_clean
