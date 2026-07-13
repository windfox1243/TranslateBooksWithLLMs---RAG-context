"""Shared language capability profiles for translation pipeline decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class TextMatchPolicy:
    """Script-aware matching capabilities used by names, aliases, and glossary terms."""

    latin_word_boundaries: bool = False
    cjk_exact_labels: bool = False
    hangul_exact_labels: bool = False
    rtl_word_boundaries: bool = False
    no_space_script: bool = False
    casefold: bool = True
    accent_sensitive: bool = False


@dataclass(frozen=True)
class LanguageProfile:
    """Internal language profile for prompt, matching, and addressing behavior."""

    code: str
    name: str
    aliases: FrozenSet[str] = field(default_factory=frozenset)
    script: str = "latin"
    match_policy: TextMatchPolicy = field(default_factory=TextMatchPolicy)
    rtl: bool = False
    addressing_family: str = "generic"
    has_formality_register: bool = False
    has_grammatical_gender: bool = False
    prompt_style: str = "generic"
    neutral_fallback: bool = False
    residue_social_terms: FrozenSet[str] = field(default_factory=frozenset)
    narrator_voice_dimensions: Tuple[str, ...] = (
        "point_of_view", "number", "tense", "style",
    )


@dataclass(frozen=True)
class LayerCapabilityReport:
    """Debug metadata describing which language-aware layer behavior was active."""

    layer: str
    language_profile: str
    capability: str
    status: str
    fallback_reason: str = ""


@dataclass(frozen=True)
class ContextSelectionReason:
    """Why a context entry was selected for prompt injection."""

    entry_type: str
    label: str
    reason: str


@dataclass(frozen=True)
class AdapterRepairValidationResult:
    """Adapter validation result for LLM repairs."""

    accepted: bool
    reason: str = ""


LATIN_POLICY = TextMatchPolicy(latin_word_boundaries=True)
CJK_POLICY = TextMatchPolicy(cjk_exact_labels=True, no_space_script=True)
HANGUL_POLICY = TextMatchPolicy(hangul_exact_labels=True)
RTL_POLICY = TextMatchPolicy(rtl_word_boundaries=True)
THAI_POLICY = TextMatchPolicy(no_space_script=True)


_PROFILES: Dict[str, LanguageProfile] = {
    "en": LanguageProfile("en", "English", frozenset({"english", "en"}), script="latin", match_policy=LATIN_POLICY, residue_social_terms=frozenset({"aunt", "brother", "captain", "commander", "dad", "daughter", "elder", "father", "lord", "master", "mother", "professor", "senior", "sister", "son", "teacher", "uncle", "younger"})),
    "fr": LanguageProfile("fr", "French", frozenset({"french", "fr", "français", "francais"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_formality_register=True, has_grammatical_gender=True, prompt_style="romance", residue_social_terms=frozenset({"frère", "soeur", "père", "mère", "oncle", "tante", "maître", "professeur"})),
    "es": LanguageProfile("es", "Spanish", frozenset({"spanish", "es", "español", "espanol"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_formality_register=True, has_grammatical_gender=True, prompt_style="romance", residue_social_terms=frozenset({"hermano", "hermana", "padre", "madre", "tío", "tía", "maestro", "profesor"})),
    "de": LanguageProfile("de", "German", frozenset({"german", "de", "deutsch"}), script="latin", match_policy=LATIN_POLICY, addressing_family="germanic", has_formality_register=True, has_grammatical_gender=True, prompt_style="germanic", residue_social_terms=frozenset({"bruder", "schwester", "vater", "mutter", "onkel", "tante", "meister", "lehrer"})),
    "vi": LanguageProfile("vi", "Vietnamese", frozenset({"vietnamese", "vietnamien", "viet", "vi", "tiếng việt", "tieng viet"}), script="latin", match_policy=LATIN_POLICY, addressing_family="vietnamese", has_formality_register=True, prompt_style="vietnamese", residue_social_terms=frozenset({"anh", "chị", "em", "ông", "bà", "cha", "mẹ", "thầy", "cô"})),
    "zh": LanguageProfile("zh", "Chinese", frozenset({"chinese", "zh", "zh-cn", "zh-tw", "中文", "汉语", "漢語"}), script="cjk", match_policy=CJK_POLICY, addressing_family="chinese", prompt_style="chinese", residue_social_terms=frozenset({"哥哥", "姐姐", "父亲", "母亲", "老师", "主人"})),
    "ja": LanguageProfile("ja", "Japanese", frozenset({"japanese", "ja", "nihongo", "日本語"}), script="cjk", match_policy=CJK_POLICY, addressing_family="japanese", has_formality_register=True, prompt_style="japanese", residue_social_terms=frozenset({"兄", "姉", "父", "母", "先生", "先輩"})),
    "ko": LanguageProfile("ko", "Korean", frozenset({"korean", "ko", "hangul", "한국어", "tiếng hàn"}), script="hangul", match_policy=HANGUL_POLICY, addressing_family="korean", has_formality_register=True, prompt_style="korean", residue_social_terms=frozenset({"형", "오빠", "누나", "언니", "아버지", "어머니", "선생님"})),
    "ar": LanguageProfile("ar", "Arabic", frozenset({"arabic", "ar", "العربية"}), script="rtl", match_policy=RTL_POLICY, rtl=True, prompt_style="rtl", residue_social_terms=frozenset({"أخي", "أختي", "أبي", "أمي", "سيدي", "معلم"})),
    "ru": LanguageProfile("ru", "Russian", frozenset({"russian", "ru", "русский"}), script="cyrillic", match_policy=LATIN_POLICY, has_grammatical_gender=True, residue_social_terms=frozenset({"брат", "сестра", "отец", "мать", "учитель", "господин"})),
    "hi": LanguageProfile("hi", "Hindi", frozenset({"hindi", "hi", "हिन्दी", "हिंदी"}), script="indic", match_policy=TextMatchPolicy(no_space_script=False), has_grammatical_gender=True, residue_social_terms=frozenset({"भाई", "बहन", "पिता", "माता", "गुरु"})),
    "th": LanguageProfile("th", "Thai", frozenset({"thai", "th", "ไทย"}), script="thai", match_policy=THAI_POLICY, residue_social_terms=frozenset({"พี่", "น้อง", "พ่อ", "แม่", "ครู", "ท่าน"})),
    "it": LanguageProfile("it", "Italian", frozenset({"italian", "it", "italiano"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_grammatical_gender=True, prompt_style="romance"),
    "pt": LanguageProfile("pt", "Portuguese", frozenset({"portuguese", "pt", "português", "portugues"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_grammatical_gender=True, prompt_style="romance"),
    "nl": LanguageProfile("nl", "Dutch", frozenset({"dutch", "nl", "nederlands"}), script="latin", match_policy=LATIN_POLICY, addressing_family="germanic", has_formality_register=True),
    "pl": LanguageProfile("pl", "Polish", frozenset({"polish", "pl", "polski"}), script="latin", match_policy=LATIN_POLICY, has_grammatical_gender=True),
    "tr": LanguageProfile("tr", "Turkish", frozenset({"turkish", "tr", "türkçe", "turkce"}), script="latin", match_policy=LATIN_POLICY),
}

# Narrator dimensions are deliberately explicit per supported language.  They
# describe what an observation may assert; they are not lexical heuristics.
_VOICE_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "en": ("point_of_view", "number", "gender", "formality", "tense", "style"),
    "fr": ("point_of_view", "number", "gender", "formality", "regional_register", "tense", "style"),
    "es": ("point_of_view", "number", "gender", "formality", "regional_register", "tense", "style"),
    "de": ("point_of_view", "number", "gender", "formality", "regional_register", "tense", "style"),
    "vi": ("point_of_view", "self_reference", "politeness", "dialogue_isolation", "style"),
    "zh": ("point_of_view", "self_reference", "pronoun_omission", "persona", "speech_level", "style"),
    "ja": ("point_of_view", "self_reference", "pronoun_omission", "persona", "speech_level", "style"),
    "ko": ("point_of_view", "self_reference", "pronoun_omission", "persona", "speech_level", "style"),
    "ar": ("point_of_view", "number", "gender", "formality", "tense", "style"),
    "ru": ("point_of_view", "number", "gender", "formality", "tense", "style"),
    "hi": ("point_of_view", "number", "gender", "formality", "tense", "style"),
    "th": ("point_of_view", "self_reference", "politeness", "dialogue_isolation", "style"),
    "it": ("point_of_view", "number", "gender", "formality", "regional_register", "tense", "style"),
    "pt": ("point_of_view", "number", "gender", "formality", "regional_register", "tense", "style"),
    "nl": ("point_of_view", "number", "gender", "formality", "regional_register", "tense", "style"),
    "pl": ("point_of_view", "number", "gender", "formality", "tense", "style"),
    "tr": ("point_of_view", "number", "gender", "formality", "tense", "style"),
}

_ALIASES = {
    alias.casefold(): code
    for code, profile in _PROFILES.items()
    for alias in profile.aliases | {profile.name}
}

GENERIC_PROFILE = LanguageProfile(
    code="generic",
    name="Generic",
    aliases=frozenset({"generic", "unknown", "custom"}),
    script="mixed",
    match_policy=TextMatchPolicy(latin_word_boundaries=True, cjk_exact_labels=True, hangul_exact_labels=True, rtl_word_boundaries=True),
    addressing_family="generic",
    prompt_style="generic",
    neutral_fallback=True,
)


def get_language_profile(language: str | None) -> LanguageProfile:
    """Return a known language profile, or a neutral generic fallback."""

    key = str(language or "").strip().casefold()
    if not key:
        return GENERIC_PROFILE
    profile = _PROFILES.get(_ALIASES.get(key, ""), GENERIC_PROFILE)
    dimensions = _VOICE_DIMENSIONS.get(profile.code)
    if not dimensions or profile.narrator_voice_dimensions == dimensions:
        return profile
    return LanguageProfile(**{
        **profile.__dict__, "narrator_voice_dimensions": dimensions,
    })


def supported_translation_languages() -> tuple[str, ...]:
    """Return canonical supported language names used by UI/backend contracts."""

    return tuple(profile.name for profile in _PROFILES.values())
