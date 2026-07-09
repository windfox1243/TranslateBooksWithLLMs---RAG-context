"""Shared language capability profiles for translation pipeline decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet


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
    "en": LanguageProfile("en", "English", frozenset({"english", "en"}), script="latin", match_policy=LATIN_POLICY),
    "fr": LanguageProfile("fr", "French", frozenset({"french", "fr", "français", "francais"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_formality_register=True, has_grammatical_gender=True, prompt_style="romance"),
    "es": LanguageProfile("es", "Spanish", frozenset({"spanish", "es", "español", "espanol"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_formality_register=True, has_grammatical_gender=True, prompt_style="romance"),
    "de": LanguageProfile("de", "German", frozenset({"german", "de", "deutsch"}), script="latin", match_policy=LATIN_POLICY, addressing_family="germanic", has_formality_register=True, has_grammatical_gender=True, prompt_style="germanic"),
    "vi": LanguageProfile("vi", "Vietnamese", frozenset({"vietnamese", "vietnamien", "viet", "vi", "tiếng việt", "tieng viet"}), script="latin", match_policy=LATIN_POLICY, addressing_family="vietnamese", has_formality_register=True, prompt_style="vietnamese"),
    "zh": LanguageProfile("zh", "Chinese", frozenset({"chinese", "zh", "zh-cn", "zh-tw", "中文", "汉语", "漢語"}), script="cjk", match_policy=CJK_POLICY, addressing_family="chinese", prompt_style="chinese"),
    "ja": LanguageProfile("ja", "Japanese", frozenset({"japanese", "ja", "nihongo", "日本語"}), script="cjk", match_policy=CJK_POLICY, addressing_family="japanese", has_formality_register=True, prompt_style="japanese"),
    "ko": LanguageProfile("ko", "Korean", frozenset({"korean", "ko", "hangul", "한국어", "tiếng hàn"}), script="hangul", match_policy=HANGUL_POLICY, addressing_family="korean", has_formality_register=True, prompt_style="korean"),
    "ar": LanguageProfile("ar", "Arabic", frozenset({"arabic", "ar", "العربية"}), script="rtl", match_policy=RTL_POLICY, rtl=True, prompt_style="rtl"),
    "ru": LanguageProfile("ru", "Russian", frozenset({"russian", "ru", "русский"}), script="cyrillic", match_policy=LATIN_POLICY, has_grammatical_gender=True),
    "hi": LanguageProfile("hi", "Hindi", frozenset({"hindi", "hi", "हिन्दी", "हिंदी"}), script="indic", match_policy=TextMatchPolicy(no_space_script=False), has_grammatical_gender=True),
    "th": LanguageProfile("th", "Thai", frozenset({"thai", "th", "ไทย"}), script="thai", match_policy=THAI_POLICY),
    "it": LanguageProfile("it", "Italian", frozenset({"italian", "it", "italiano"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_grammatical_gender=True, prompt_style="romance"),
    "pt": LanguageProfile("pt", "Portuguese", frozenset({"portuguese", "pt", "português", "portugues"}), script="latin", match_policy=LATIN_POLICY, addressing_family="romance", has_grammatical_gender=True, prompt_style="romance"),
    "nl": LanguageProfile("nl", "Dutch", frozenset({"dutch", "nl", "nederlands"}), script="latin", match_policy=LATIN_POLICY, addressing_family="germanic", has_formality_register=True),
    "pl": LanguageProfile("pl", "Polish", frozenset({"polish", "pl", "polski"}), script="latin", match_policy=LATIN_POLICY, has_grammatical_gender=True),
    "tr": LanguageProfile("tr", "Turkish", frozenset({"turkish", "tr", "türkçe", "turkce"}), script="latin", match_policy=LATIN_POLICY),
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
    return _PROFILES.get(_ALIASES.get(key, ""), GENERIC_PROFILE)


def supported_translation_languages() -> tuple[str, ...]:
    """Return canonical supported language names used by UI/backend contracts."""

    return tuple(profile.name for profile in _PROFILES.values())
