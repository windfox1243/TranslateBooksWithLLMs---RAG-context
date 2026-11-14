"""Language-name resolution shared by every "which target language?" check.

Before this, each module carried its own alias set and they disagreed: the
novel-context layer did not accept "vi", the reconciler accepted anything
containing "viet", and the addressing engine knew neither "vi-VN" nor a
qualified name. They now all go through get_language_profile().
"""

import pytest

from src.languages.vietnamese.context_rules import _is_vietnamese_target_language
from src.utils.language_profiles import get_language_profile, is_target_language
from src.utils.universal_addressing_engine import UniversalAddressingEngine


@pytest.mark.parametrize(
    "name, code",
    [
        ("Vietnamese", "vi"),
        ("vi", "vi"),
        ("vi-VN", "vi"),
        ("Vietnamese (Southern)", "vi"),
        ("tiếng việt", "vi"),
        ("Tieng Viet", "vi"),
        ("pt_BR", "pt"),
        ("Portuguese - Brazil", "pt"),
        ("zh-TW", "zh"),
        ("tiếng nhật", "ja"),
        ("English", "en"),
    ],
)
def test_known_names_resolve(name, code):
    assert get_language_profile(name).code == code


@pytest.mark.parametrize("name", ["", None, "Klingon", "eng", "the Vietnamese way"])
def test_unknown_names_fall_back_to_generic(name):
    assert get_language_profile(name).code == "generic"


def test_is_target_language():
    assert is_target_language("vi-VN", "vi")
    assert not is_target_language("English", "vi")


def test_novel_context_check_accepts_the_language_code():
    assert _is_vietnamese_target_language("vi")
    assert _is_vietnamese_target_language("Vietnamese")
    assert not _is_vietnamese_target_language("Japanese")


@pytest.mark.parametrize(
    "name, code",
    [("vi-VN", "vi"), ("Japanese", "ja"), ("한국어", "ko"), ("German", "generic")],
)
def test_addressing_engine_uses_the_shared_resolution(name, code):
    assert UniversalAddressingEngine(name).lang_code == code
