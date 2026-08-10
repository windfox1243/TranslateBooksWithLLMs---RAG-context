"""Vietnamese-specific addressing inference, repair and mismatch detection."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import (
    _canonical_display_name,
    _canonical_gender,
    _clean_inline_text,
    _is_invalid_context_key,
    _plain_key,
)
from .dynamic_state import (
    _DYNAMIC_DELETE_VALUES,
    _DYNAMIC_RELATION_PATTERN,
    _canonical_relationship_party,
    _parse_dynamic_relation,
)


def _is_vietnamese_target_language(target_language: Optional[str]) -> bool:
    target = str(target_language or "").strip().casefold()
    return target in {"vietnamese", "vietnamien", "viet", "tiếng việt", "tieng viet"}
def _has_complete_vietnamese_addressing_details(details: str) -> bool:
    self_ref = _vietnamese_addressing_field(details, "self-reference")
    second_p = _vietnamese_addressing_field(details, "second-person pronoun")
    vocative_f = _vietnamese_addressing_field(details, "vocative/address form")
    return bool(self_ref and second_p and vocative_f)
def _vietnamese_addressing_field(details: str, field_name: str) -> str:
    key_patterns = {
        "self-reference": r"(?:self[-_\s]*reference|xưng(?:[-_\s]*hô)?)",
        "second-person pronoun": r"(?:second[-_\s]*person(?:[-_\s]*pronoun)?|gọi)",
        "vocative/address form": r"(?:vocative(?:[/-_]address)?(?:[-_\s]*form)?|danh[-_\s]*xưng|tên[-_\s]*gọi)"
    }
    pattern = key_patterns.get(field_name.casefold(), re.escape(field_name))
    match = re.search(
        rf"(?:\b{pattern}\s*[:=]\s*)(?P<value>[^;|\"\n]+)",
        str(details or ""),
        flags=re.IGNORECASE,
    )
    return _clean_inline_text(match.group("value")).strip(" \"'") if match else ""
def _replace_vietnamese_addressing_field(
    details: str,
    field_name: str,
    value: str,
) -> str:
    key_patterns = {
        "self-reference": r"(?:self[-_\s]*reference|xưng(?:[-_\s]*hô)?)",
        "second-person pronoun": r"(?:second[-_\s]*person(?:[-_\s]*pronoun)?|gọi)",
        "vocative/address form": r"(?:vocative(?:[/-_]address)?(?:[-_\s]*form)?|danh[-_\s]*xưng|tên[-_\s]*gọi)"
    }
    pattern = key_patterns.get(field_name.casefold(), re.escape(field_name))
    return re.sub(
        rf"(?P<prefix>\b{pattern}\s*[:=]\s*)[^;|\"\n]+",
        lambda match: f"{match.group('prefix')}{value}",
        str(details or ""),
        count=1,
        flags=re.IGNORECASE,
    )
def _vietnamese_rule_has_required_cue(
    details: str,
    rule: Dict[str, Any],
) -> bool:
    clean = _clean_inline_text(details).casefold()
    second_person_terms = set(rule.get("second_person") or ())
    return any(cue in clean for cue in rule.get("cues", ())) or (
        ("biological sibling" in clean or "kinship" in clean)
        and bool(second_person_terms & {
            _vietnamese_addressing_field(
                details,
                "second-person pronoun",
            ).casefold(),
            _vietnamese_addressing_field(
                details,
                "vocative/address form",
            ).casefold(),
        })
    )
def _is_vietnamese_attitude_shift(details: str) -> bool:
    clean = _clean_inline_text(details).casefold()
    attitude_cues = (
        "ironic mockery",
        "sarcastic hostility",
        "mỉa mai",
        "châm biếm",
        "thù địch",
        "đổi thái độ",
        "từ mặt",
    )
    return any(cue in clean for cue in attitude_cues)
_VIETNAMESE_POSSESSIVE_TITLE_HEADS = {
    "bác sĩ",
    "cố vấn",
    "giáo sư",
    "giám đốc",
    "hiệu trưởng",
    "huấn luyện viên",
    "luật sư",
    "quản gia",
    "quản lý",
    "thanh tra",
    "trợ lý",
}
_VIETNAMESE_SECOND_PERSON_TERMS = {
    "anh",
    "bà",
    "bác",
    "bác sĩ",
    "bạn",
    "bệ hạ",
    "chàng",
    "cháu",
    "chị",
    "chủ tịch",
    "chủ nhân",
    "chú",
    "cô",
    "cô giáo",
    "cố vấn",
    "công tử",
    "cụ",
    "đại nhân",
    "đội trưởng",
    "đệ tử",
    "em",
    "giáo sư",
    "giám đốc",
    "hậu bối",
    "hiệu trưởng",
    "hoàng thượng",
    "huấn luyện viên",
    "luật sư",
    "mày",
    "mi",
    "ngài",
    "ngươi",
    "nàng",
    "ông",
    "quản gia",
    "quản lý",
    "quý khách",
    "sếp",
    "sư phụ",
    "sư tôn",
    "sư tỷ",
    "sư huynh",
    "thanh tra",
    "thầy",
    "thầy giáo",
    "thiếu gia",
    "tiền bối",
    "tông chủ",
    "trợ lý",
    "trưởng lão",
}
_SOURCE_ADDRESS_TITLE_MAP = {
    "boss": "sếp",
    "captain": "đội trưởng",
    "chief": "sếp",
    "doctor": "bác sĩ",
    "manager": "quản lý",
    "master": "sư phụ",
    "president": "chủ tịch",
    "professor": "giáo sư",
    "senpai": "tiền bối",
    "sensei": "thầy",
    "teacher": "thầy",
    "trainer": "huấn luyện viên",
}
def _split_addressing_variants(term: str) -> List[str]:
    return [part.strip() for part in str(term or "").split("/") if part.strip()]
def _is_pure_vietnamese_second_person_pronoun(term: str) -> bool:
    variants = _split_addressing_variants(term)
    return bool(variants) and all(
        _plain_key(variant) in _VIETNAMESE_SECOND_PERSON_TERMS
        for variant in variants
    )
def _source_address_title_translation(term: str) -> str:
    for variant in _split_addressing_variants(term):
        key = _plain_key(variant)
        for source_title, target_title in _SOURCE_ADDRESS_TITLE_MAP.items():
            if re.search(rf"\b{re.escape(source_title)}\b", key):
                return target_title
    return ""
def _looks_like_source_address_form(term: str, addressee: str = "") -> bool:
    if not term or _is_pure_vietnamese_second_person_pronoun(term):
        return False
    variants = _split_addressing_variants(term)
    addressee_tokens = {
        token
        for token in _plain_key(addressee).split()
        if len(token) > 2
    }
    honorific_pattern = re.compile(
        r"(?i)(?:-(?:san|sama|chan|kun|senpai|sensei|dono|chi)\b)"
    )
    for variant in variants:
        variant_key = _plain_key(variant)
        if not variant_key:
            continue
        if _source_address_title_translation(variant):
            return True
        if honorific_pattern.search(variant):
            return True
        if addressee_tokens and addressee_tokens & set(variant_key.split()):
            return True
        if re.search(r"[A-Za-z]", variant) and not re.search(
            r"[à-ỹÀ-Ỹ]",
            variant,
        ):
            return True
    return False
def _gendered_vietnamese_second_person(
    gender: str,
    *,
    formal_female: str = "chị",
) -> str:
    gender_key = _canonical_gender(gender).casefold()
    if gender_key == "male":
        return "anh"
    if gender_key == "female":
        return formal_female
    return ""
def _first_name_for_vocative(name: str) -> str:
    parts = _canonical_display_name(name).split()
    return parts[0] if parts else _canonical_display_name(name)
def _profile_text(name: str, profiles: Optional[Dict[str, Dict[str, str]]]) -> str:
    profile = (profiles or {}).get(_plain_key(name), {})
    return _clean_inline_text(
        f"{profile.get('name', name)} {profile.get('details', '')}"
    ).casefold()
def _profile_gender(
    name: str,
    profiles: Optional[Dict[str, Dict[str, str]]],
) -> str:
    return str((profiles or {}).get(_plain_key(name), {}).get("gender", ""))
def _profile_has_role(
    name: str,
    profiles: Optional[Dict[str, Dict[str, str]]],
    *roles: str,
) -> bool:
    text = _profile_text(name, profiles)
    return any(role in text for role in roles)
def _relationship_has_any(text: str, cues: Tuple[str, ...]) -> bool:
    clean = _clean_inline_text(text).casefold()
    return any(cue in clean for cue in cues)
def _looks_like_trainer_role(name: str) -> bool:
    clean = _clean_inline_text(name).casefold()
    return any(
        cue in clean
        for cue in (
            "coach",
            "trainer",
            "huấn luyện viên",
        )
    )
def _vietnamese_addressing_from_relationship(
    speaker: str,
    addressee: str,
    relationship_details: str,
    character_profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[Tuple[str, str, str, str]]:
    clean = _clean_inline_text(relationship_details).casefold()
    speaker_is_trainer = _profile_has_role(
        speaker,
        character_profiles,
        "trainer",
        "coach",
        "huấn luyện viên",
    )
    addressee_is_trainer = _profile_has_role(
        addressee,
        character_profiles,
        "trainer",
        "coach",
        "huấn luyện viên",
    )
    speaker_gender = _profile_gender(speaker, character_profiles)
    addressee_gender = _profile_gender(addressee, character_profiles)
    addressee_short = _first_name_for_vocative(addressee)

    if (
        "trainer" in clean
        or "trainee" in clean
        or "coach" in clean
        or speaker_is_trainer
        or addressee_is_trainer
    ):
        if speaker_is_trainer and not addressee_is_trainer:
            self_ref = "anh" if _canonical_gender(speaker_gender).casefold() == "male" else "tôi"
            return (
                self_ref,
                "em",
                addressee_short,
                "trainer-trainee relationship",
            )
        if addressee_is_trainer and not speaker_is_trainer:
            second = _gendered_vietnamese_second_person(addressee_gender)
            if not second:
                second = "huấn luyện viên"
            return (
                "tôi",
                second,
                addressee_short,
                "student-trainer relationship",
            )

    peer_cues = (
        "classmate",
        "close friend",
        "friend",
        "peer",
        "rival",
        "roommate",
        "training partner",
    )
    if _relationship_has_any(clean, peer_cues):
        return ("tớ", "cậu", addressee_short, "peer-level relationship")

    staff_cues = (
        "academy staff",
        "chairwoman",
        "librarian",
        "staff member",
        "student and",
    )
    if _relationship_has_any(clean, staff_cues):
        second = _gendered_vietnamese_second_person(
            addressee_gender,
            formal_female="cô",
        ) or "bạn"
        return ("tôi", second, addressee_short, "professional school setting")

    professional_cues = (
        "professional",
        "competition",
        "competitor",
        "contract",
    )
    if _relationship_has_any(clean, professional_cues):
        second = _gendered_vietnamese_second_person(
            addressee_gender,
            formal_female="cô",
        ) or "bạn"
        return ("tôi", second, addressee_short, "professional relationship")

    return None
def _seed_vietnamese_addressing_from_relationships(
    addressing: str,
    relationships: str,
    alias_map: Dict[str, str],
    character_profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    if addressing.strip() or not relationships.strip():
        return addressing.strip()

    seeded: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for raw_line in relationships.splitlines():
        match = _DYNAMIC_RELATION_PATTERN.match(raw_line)
        if not match:
            continue
        left = _canonical_relationship_party(match.group("left"), alias_map)
        right = _canonical_relationship_party(match.group("right"), alias_map)
        details = _clean_inline_text(match.group("details"))
        if _is_invalid_context_key(left) or _is_invalid_context_key(right):
            continue
        directions = [(left, right)]
        if match.group("arrow") == "↔":
            directions.append((right, left))
        for speaker, addressee in directions:
            key = (_plain_key(speaker), _plain_key(addressee))
            if key in seen or key[0] == key[1]:
                continue
            inferred = _vietnamese_addressing_from_relationship(
                speaker,
                addressee,
                details,
                character_profiles,
            )
            if not inferred:
                continue
            self_ref, second, vocative, reason = inferred
            if not self_ref or not second or not vocative:
                continue
            seen.add(key)
            seeded.append(
                f'- {speaker} → {addressee}: "{vocative}" | '
                f'"self-reference: {self_ref}; second-person pronoun: {second}; '
                f'vocative/address form: {vocative}" | '
                f"seeded from relationship evolution: {reason}."
            )

    return "\n".join(seeded).strip()
def _infer_replacement_pronoun(
    self_ref: str,
    second_person: str,
    vocative: str,
    details: str,
    addressee: str = "",
    character_genders: Optional[Dict[str, str]] = None,
) -> str:
    self_key = _plain_key(self_ref)
    clean = _clean_inline_text(
        f"{second_person} {vocative} {details}"
    ).casefold()
    gender = (character_genders or {}).get(_plain_key(addressee), "")
    title = _source_address_title_translation(second_person) or (
        _source_address_title_translation(vocative)
        if any(cue in clean for cue in ("trainer", "teacher", "boss", "senpai", "sensei"))
        else ""
    )
    peer_cues = (
        "casual",
        "classmate",
        "close friend",
        "competitive rival",
        "friendly",
        "friend",
        "partner",
        "peer",
        "roommate",
        "training partner",
    )
    hierarchy_cues = (
        "age hierarchy",
        "elder",
        "junior",
        "mentor",
        "senior",
        "seniority",
        "student",
        "teacher",
        "trainer",
        "tiền bối",
    )
    formal_cues = (
        "colleague",
        "formal",
        "professional",
        "respectful",
        "strained",
    )

    if self_key in {"anh", "chị"}:
        return "em"
    if self_key in {"em", "con", "cháu", "đệ tử"}:
        return title or _gendered_vietnamese_second_person(gender) or "tiền bối"
    if self_key == "ta":
        return "ngươi"
    if self_key in {"mình", "tớ"}:
        return "cậu"
    if self_key == "tôi":
        if title and any(cue in clean for cue in hierarchy_cues + formal_cues):
            return title
        if any(cue in clean for cue in peer_cues):
            return "cậu"
        if any(cue in clean for cue in hierarchy_cues):
            return title or _gendered_vietnamese_second_person(gender) or "tiền bối"
        if any(cue in clean for cue in formal_cues):
            return (
                title
                or _gendered_vietnamese_second_person(gender, formal_female="cô")
                or "bạn"
            )
        return _gendered_vietnamese_second_person(gender) or "cậu"
    return title or _gendered_vietnamese_second_person(gender) or ""
def _sanitize_vietnamese_second_person_pronoun(second_p: str) -> str:
    if not second_p:
        return second_p
    clean = second_p.strip("`\"' ")
    if " của " in clean.casefold():
        parts = re.split(r"\s+của\s+", clean, flags=re.IGNORECASE)
        head = parts[0].strip()
        if head.casefold() in _VIETNAMESE_POSSESSIVE_TITLE_HEADS:
            return head
    return second_p
def _has_vietnamese_non_peer_formality_cue(details: str) -> bool:
    clean = _clean_inline_text(details).casefold()
    if not clean:
        return False
    non_peer_cues = (
        "academy staff",
        "age hierarchy",
        "boss",
        "coach",
        "colleague",
        "formal",
        "mentor",
        "professional",
        "respectful",
        "senior",
        "seniority",
        "staff",
        "strained",
        "student-trainer",
        "student to mentor",
        "student to senior",
        "teacher",
        "trainer",
        "trainer-trainee",
        "trainee",
        "workplace",
        "hậu bối",
        "huấn luyện viên",
        "tiền bối",
    )
    return any(cue in clean for cue in non_peer_cues)
def _has_vietnamese_peer_context_cue(details: str) -> bool:
    clean = _clean_inline_text(details).casefold()
    return any(
        cue in clean
        for cue in (
            "best friend",
            "classmate",
            "close friend",
            "fellow student",
            "friend",
            "peer",
            "peer-level",
            "rival",
            "roommate",
            "school-year peers",
            "teammate",
            "training partner",
        )
    )
def _has_vietnamese_soft_minh_voice_cue(details: str) -> bool:
    clean = _clean_inline_text(details).casefold()
    return any(
        cue in clean
        for cue in (
            "established character voice",
            "gentle voice",
            "introspective",
            "mình-cậu",
            "self-reflective",
            "soft voice",
            "stored context",
        )
    )
def _has_vietnamese_trainer_to_trainee_cue(
    speaker: str,
    addressee: str,
    details: str,
) -> bool:
    clean = _clean_inline_text(details).casefold()
    if not _looks_like_trainer_role(speaker):
        return False
    if _looks_like_trainer_role(addressee):
        return False
    return any(
        cue in clean
        for cue in (
            "junior",
            "senior-junior",
            "trainer-trainee",
            "trainee",
        )
    )
def _is_vietnamese_junior_to_senior(
    details: str,
    speaker: str = "",
    addressee: str = "",
    second_person: str = "",
    vocative: str = "",
) -> bool:
    """Report whether the addressing engine reads this pair as junior to senior."""
    from src.utils.universal_addressing_engine import (
        _SENIOR_PRONOUN_SETS,
        UniversalAddressingEngine,
    )

    # Calling someone "anh"/"chị"/"thầy" places the speaker below them whatever
    # the social basis says. The basis often names both roles at once
    # ("trainer-trainee"), which cannot tell the two directions apart.
    if _plain_key(second_person) in _SENIOR_PRONOUN_SETS["vi"]:
        return True
    hierarchy = UniversalAddressingEngine(language="vi").resolve_seniority_hierarchy(
        speaker,
        addressee,
        details,
        vocative=vocative,
    )
    return hierarchy == "JUNIOR_TO_SENIOR"
def _repair_vietnamese_addressing_details(
    details: str,
    addressee: str = "",
    speaker: str = "",
    character_genders: Optional[Dict[str, str]] = None,
) -> str:
    """Normalize obvious Vietnamese paired-address fields before merge."""
    if not _has_complete_vietnamese_addressing_details(details):
        return details
    raw_second_p = _vietnamese_addressing_field(details, "second-person pronoun")
    cleaned_second_p = _sanitize_vietnamese_second_person_pronoun(raw_second_p)
    if cleaned_second_p != raw_second_p:
        details = _replace_vietnamese_addressing_field(
            details,
            "second-person pronoun",
            cleaned_second_p,
        )
        raw_second_p = cleaned_second_p
    if _is_vietnamese_attitude_shift(details):
        return details

    self_reference_raw = _vietnamese_addressing_field(details, "self-reference")
    vocative_raw = _vietnamese_addressing_field(details, "vocative/address form")
    if _looks_like_source_address_form(raw_second_p, addressee):
        replacement = _infer_replacement_pronoun(
            self_reference_raw,
            raw_second_p,
            vocative_raw,
            details,
            addressee,
            character_genders,
        )
        if replacement:
            details = _replace_vietnamese_addressing_field(
                details,
                "second-person pronoun",
                replacement,
            )

    self_reference_raw = _vietnamese_addressing_field(details, "self-reference")
    if (
        _plain_key(self_reference_raw) in {"mình", "tớ"}
        and _has_vietnamese_non_peer_formality_cue(details)
        # A junior speaking to a senior says "em", not the distant "tôi", and the
        # addressing engine further down promotes "tớ"/"mình" to exactly that.
        # Rewriting to "tôi" first hid the peer pronoun the promotion looks for,
        # so a student addressing their trainer ended up sounding like a stranger.
        and not _is_vietnamese_junior_to_senior(
            details,
            speaker,
            addressee,
            _vietnamese_addressing_field(details, "second-person pronoun"),
            _vietnamese_addressing_field(details, "vocative/address form"),
        )
    ):
        details = _replace_vietnamese_addressing_field(
            details,
            "self-reference",
            "tôi",
        )

    self_reference_raw = _vietnamese_addressing_field(details, "self-reference")
    raw_second_p = _vietnamese_addressing_field(details, "second-person pronoun")
    if (
        _plain_key(self_reference_raw) == "mình"
        and _plain_key(raw_second_p) in {"cậu", "bạn"}
        and _has_vietnamese_peer_context_cue(details)
        and not _has_vietnamese_soft_minh_voice_cue(details)
    ):
        details = _replace_vietnamese_addressing_field(
            details,
            "self-reference",
            "tớ",
        )

    self_reference_raw = _vietnamese_addressing_field(details, "self-reference")
    raw_second_p = _vietnamese_addressing_field(details, "second-person pronoun")
    if (
        _plain_key(raw_second_p) in {"anh", "chị", "cô", "ông", "bà"}
        and _has_vietnamese_trainer_to_trainee_cue(speaker, addressee, details)
    ):
        if _plain_key(self_reference_raw) == "tôi":
            speaker_gender = ""
            if character_genders:
                speaker_gender = character_genders.get(_plain_key(speaker), "")
            if _canonical_gender(speaker_gender).casefold() == "male":
                details = _replace_vietnamese_addressing_field(
                    details,
                    "self-reference",
                    "anh",
                )
        details = _replace_vietnamese_addressing_field(
            details,
            "second-person pronoun",
            "em",
        )

    self_reference_raw = _vietnamese_addressing_field(details, "self-reference")
    raw_second_p = _vietnamese_addressing_field(details, "second-person pronoun")
    vocative_raw = _vietnamese_addressing_field(details, "vocative/address form")

    # Clean up ambiguous slash pronouns (e.g. "anh/chị")
    if "/" in raw_second_p:
        addressee_g = character_genders.get(_plain_key(addressee), "") if character_genders else ""
        if _canonical_gender(addressee_g).casefold() == "female":
            raw_second_p = "chị"
        elif _canonical_gender(addressee_g).casefold() == "male":
            raw_second_p = "anh"
        else:
            raw_second_p = raw_second_p.split("/")[0].strip()
        details = _replace_vietnamese_addressing_field(details, "second-person pronoun", raw_second_p)

    if "/" in self_reference_raw:
        speaker_g = character_genders.get(_plain_key(speaker), "") if character_genders else ""
        if _canonical_gender(speaker_g).casefold() == "female":
            self_reference_raw = "chị" if "chị" in self_reference_raw else self_reference_raw.split("/")[0].strip()
        elif _canonical_gender(speaker_g).casefold() == "male":
            self_reference_raw = "anh" if "anh" in self_reference_raw else self_reference_raw.split("/")[0].strip()
        else:
            self_reference_raw = self_reference_raw.split("/")[0].strip()
        details = _replace_vietnamese_addressing_field(details, "self-reference", self_reference_raw)

    # Clean up placeholder vocative strings ("none", "N/A", "...", "null")
    if _plain_key(vocative_raw) in {"none", "n/a", "...", "null", "none."}:
        vocative_raw = ""
        details = _replace_vietnamese_addressing_field(details, "vocative/address form", "")

    from src.utils.universal_addressing_engine import UniversalAddressingEngine
    engine = UniversalAddressingEngine(language="vi")

    self_ref = _vietnamese_addressing_field(details, "self-reference")
    second_p = _vietnamese_addressing_field(details, "second-person pronoun")
    vocative = _vietnamese_addressing_field(details, "vocative/address form")

    if _plain_key(self_ref) == "con" and _plain_key(second_p) in {"anh", "chị", "cậu", "bạn"}:
        v_low = _plain_key(vocative)
        c_low = details.casefold()
        if v_low in {"cha", "bố", "ba"} or "father" in c_low or "cha" in c_low or "bố" in c_low:
            rep_p = v_low if v_low in {"cha", "bố", "ba"} else "cha"
            details = _replace_vietnamese_addressing_field(details, "second-person pronoun", rep_p)
            second_p = rep_p
        elif v_low in {"mẹ", "má"} or "mother" in c_low or "mẹ" in c_low or "má" in c_low:
            rep_p = v_low if v_low in {"mẹ", "má"} else "mẹ"
            details = _replace_vietnamese_addressing_field(details, "second-person pronoun", rep_p)
            second_p = rep_p
        elif v_low in {"father", "cha"} or "father" in c_low:
            details = _replace_vietnamese_addressing_field(details, "second-person pronoun", "cha")
            second_p = "cha"
        elif v_low in {"mother", "mẹ"} or "mother" in c_low:
            details = _replace_vietnamese_addressing_field(details, "second-person pronoun", "mẹ")
            second_p = "mẹ"

    if self_ref or second_p:
        rep_self, rep_target, rep_voc = engine.validate_and_repair_pair(
            self_pronoun=self_ref,
            target_pronoun=second_p,
            speaker=speaker,
            addressee=addressee,
            vocative=vocative,
            details_context=details,
            character_genders=character_genders,
        )
        if rep_self != self_ref:
            details = _replace_vietnamese_addressing_field(details, "self-reference", rep_self)
        if rep_target != second_p:
            details = _replace_vietnamese_addressing_field(details, "second-person pronoun", rep_target)
        if rep_voc and rep_voc != vocative:
            details = _replace_vietnamese_addressing_field(details, "vocative/address form", rep_voc)

    self_reference = _vietnamese_addressing_field(details, "self-reference").casefold()
    second_person = _vietnamese_addressing_field(details, "second-person pronoun").casefold()
    vocative_val = _vietnamese_addressing_field(details, "vocative/address form").casefold()

    for rule in _VIETNAMESE_INCOMPATIBLE_ADDRESSING_REPAIRS:
        second_person_terms = set(rule.get("second_person") or ())
        if not (
            self_reference in set(rule.get("self_references") or ())
            and (second_person in second_person_terms or vocative_val in second_person_terms)
            and _vietnamese_rule_has_required_cue(details, rule)
        ):
            continue
        return _replace_vietnamese_addressing_field(
            details,
            "self-reference",
            str(rule["replacement_self_reference"]),
        )

    return details
def _repair_vietnamese_addressing_line(
    line: str,
    alias_map: Dict[str, str],
    character_genders: Optional[Dict[str, str]] = None,
) -> str:
    if "`" in line:
        line = line.replace("`", "")
    parsed = _parse_dynamic_relation(line, alias_map)
    if not parsed:
        return line
    relation_key, rendered, details = parsed
    left_party, arrow, right_party = relation_key

    # Convert invalid bidirectional ↔ to directional → if pronouns are asymmetric
    if arrow == "↔":
        self_ref = _vietnamese_addressing_field(details, "self-reference").casefold()
        second_p = _vietnamese_addressing_field(details, "second-person pronoun").casefold()
        if (self_ref in {"em", "ta", "tôi", "anh", "chị"} and second_p in {"chị", "anh", "em", "thầy", "cô"}) and (self_ref != second_p):
            line = line.replace("↔", "→")
            parsed = _parse_dynamic_relation(line, alias_map)
            if parsed:
                relation_key, rendered, details = parsed
                left_party, arrow, right_party = relation_key

    # Auto-format simple 2-part pairs (e.g. "tớ - cậu", "tôi - cậu (bạn học)") into 3-part canonical format
    clean_details = _clean_inline_text(details).strip('" ')
    if not _vietnamese_addressing_field(details, "self-reference"):
        pair_match = re.search(
            r"^\s*(?P<self>[\w\s]+)\s*[-–—/]\s*(?P<second>[\w\s]+)(?:\s*\((?P<reason>.*?)\))?\s*$",
            clean_details,
        )
        if pair_match:
            self_val = pair_match.group("self").strip()
            second_val = pair_match.group("second").strip()
            reason = (pair_match.group("reason") or "addressing").strip()
            target_display = right_party.title()
            details = f'"{target_display}" | "self-reference: {self_val}; second-person pronoun: {second_val}; vocative/address form: {target_display}" | {reason}'

    repaired_details = _repair_vietnamese_addressing_details(
        details,
        right_party,
        left_party,
        character_genders,
    )
    left_part = rendered[:-len(parsed[2])] if parsed[2] else f"{rendered}: "
    return f"{left_part}{repaired_details}"
def _repair_vietnamese_addressing_block(
    addressing: str,
    alias_map: Dict[str, str],
    character_genders: Optional[Dict[str, str]] = None,
) -> str:
    lines = addressing.splitlines()
    parsed_entries = []
    pair_map = {}

    for raw_line in lines:
        parsed = _parse_dynamic_relation(raw_line, alias_map)
        if parsed:
            relation_key, _, details = parsed
            spk, _, adr = relation_key
            pair_map[(spk.casefold(), adr.casefold())] = (spk, adr, details)
            parsed_entries.append((raw_line, spk, adr, details))
        else:
            parsed_entries.append((raw_line, None, None, None))

    repaired_lines = []
    for raw_line, spk, adr, details in parsed_entries:
        if not spk or not adr:
            repaired_lines.append(raw_line)
            continue

        rev_key = (adr.casefold(), spk.casefold())
        rev_entry = pair_map.get(rev_key)

        enriched_details = details
        if rev_entry:
            _, _, rev_details = rev_entry
            rev_details_lower = rev_details.lower()
            rev_target = _vietnamese_addressing_field(rev_details, "second-person pronoun").casefold()

            if rev_target in {"em", "cháu", "con"} or any(
                k in rev_details_lower
                for k in ("trainer", "teacher", "thầy", "sếp", "giám đốc", "senpai", "sunbae")
            ):
                if not any(
                    k in enriched_details.lower()
                    for k in ("senior", "trainer", "thầy", "sếp", "giám đốc")
                ):
                    if any(k in rev_details_lower for k in ("trainer", "teacher", "thầy", "sếp", "giám đốc")):
                        enriched_details = f"{enriched_details} | junior to senior, trainer/student hierarchy"
                    else:
                        enriched_details = f"{enriched_details} | junior to senior, senior/junior hierarchy"

        line_to_repair = raw_line
        if enriched_details != details and ":" in raw_line:
            parts = raw_line.split(":", 1)
            line_to_repair = f"{parts[0]}: {enriched_details}"

        repaired_lines.append(
            _repair_vietnamese_addressing_line(
                line_to_repair,
                alias_map,
                character_genders,
            )
        )

    return "\n".join(repaired_lines).strip()
def _has_vietnamese_hierarchy_addressing_mismatch(details: str) -> bool:
    clean = _clean_inline_text(details).casefold()
    if not clean or _is_vietnamese_attitude_shift(details):
        return False

    self_reference = _vietnamese_addressing_field(details, "self-reference").casefold()
    second_person = _vietnamese_addressing_field(
        details,
        "second-person pronoun",
    ).casefold()
    vocative = _vietnamese_addressing_field(
        details,
        "vocative/address form",
    ).casefold()

    casual_self = self_reference in {"tớ", "mình"}
    casual_second = second_person in {"cậu", "bạn"}
    hierarchy_cues = (
        "senior",
        "tiền bối",
        "hậu bối",
        "năm trên",
        "năm dưới",
        "năm nhất",
        "upper-year",
        "underclass",
        "junior to senior",
        "student to senior",
        "student to mentor",
        "younger to older",
        "older student",
        "older sibling",
        "older sister",
        "older brother",
        "lớn tuổi",
        "hơn tuổi",
        "sư huynh",
        "sư tỷ",
        "huynh trưởng",
    )
    has_hierarchy_cue = any(cue in clean for cue in hierarchy_cues)
    vocative_marks_seniority = any(
        cue in vocative
        for cue in ("senior", "tiền bối", "anh ", "chị ", "thầy", "cô", "sư huynh", "sư tỷ")
    )
    return (has_hierarchy_cue or vocative_marks_seniority) and (
        casual_second or (casual_self and not vocative_marks_seniority)
    )
def _has_vietnamese_register_addressing_mismatch(details: str) -> bool:
    clean = _clean_inline_text(details).casefold()
    if not clean or _is_vietnamese_attitude_shift(details):
        return False

    self_reference = _vietnamese_addressing_field(details, "self-reference").casefold()
    second_person = _vietnamese_addressing_field(
        details,
        "second-person pronoun",
    ).casefold()
    if second_person != "ngươi":
        return False

    modern_self_references = {
        "anh",
        "chị",
        "cháu",
        "con",
        "cô",
        "em",
        "mình",
        "tôi",
        "tớ",
    }
    return self_reference in modern_self_references
def _has_vietnamese_addressing_mismatch(details: str) -> bool:
    second_person = _vietnamese_addressing_field(
        details,
        "second-person pronoun",
    )
    if (
        second_person
        and not _is_pure_vietnamese_second_person_pronoun(second_person)
        and _looks_like_source_address_form(second_person)
    ):
        return True
    return (
        _has_vietnamese_hierarchy_addressing_mismatch(details)
        or _has_vietnamese_register_addressing_mismatch(details)
    )
_VIETNAMESE_GENERIC_NEUTRAL_SELF_REFERENCES = {"tôi"}
_VIETNAMESE_GENERIC_NEUTRAL_SECOND_PERSON = {
    "anh",
    "bà",
    "bạn",
    "chị",
    "cậu",
    "cô",
    "ông",
}
_VIETNAMESE_ANCHORED_KINSHIP_SELF_REFERENCES = {
    "anh",
    "bà",
    "bác",
    "cháu",
    "chị",
    "con",
    "cô",
    "em",
    "ông",
    "thầy",
}
_VIETNAMESE_ANCHORED_KINSHIP_SECOND_PERSON = {
    "anh",
    "bà",
    "bác",
    "cháu",
    "chị",
    "chú",
    "cô",
    "dì",
    "dượng",
    "em",
    "ông",
    "thầy",
    "thím",
}
_VIETNAMESE_KINSHIP_ADDRESSING_CUES = (
    "anh em",
    "biological sibling",
    "brother",
    "chị em",
    "em ruột",
    "family",
    "older brother",
    "older sister",
    "sibling",
    "sister",
    "younger brother",
    "younger sister",
    "sư huynh",
    "sư tỷ",
    "sư đệ",
    "sư muội",
    "sư phụ",
    "sư tôn",
    "sư phó",
    "huynh trưởng",
    "tỷ tỷ",
    "ca ca",
    "muội muội",
    "đại ca",
    "nhị ca",
    "tam ca",
    "tiểu muội",
    "bác",
    "chú",
    "dì",
    "thím",
    "dượng",
    "cụ",
    "cụ ông",
    "cụ bà",
    "phụ thân",
    "mẫu thân",
    "nương",
    "huynh đệ",
)
_VIETNAMESE_STATUS_ADDRESSING_CUES = (
    "bệ hạ",
    "công chúa",
    "deference",
    "điện hạ",
    "emperor",
    "empress",
    "her majesty",
    "highness",
    "his majesty",
    "hoàng đế",
    "hoàng hậu",
    "majesty",
    "ngài",
    "nữ hoàng",
    "queen",
    "royal",
    "title",
    "your majesty",
    "tông chủ",
    "trưởng lão",
    "thiếu gia",
    "tiểu thư",
    "quận chúa",
    "công tử",
    "thế tử",
    "chủ nhân",
    "thiên tuế",
    "chủ tịch",
    "giám đốc",
    "quản gia",
    "bác sĩ",
    "luật sư",
    "thầy giáo",
    "cô giáo",
)
_VIETNAMESE_INCOMPATIBLE_ADDRESSING_REPAIRS = (
    {
        "self_references": {"mình", "tôi", "tớ"},
        "second_person": {"chị", "sư tỷ", "tỷ tỷ"},
        "replacement_self_reference": "em",
        "cues": (
            "big sister",
            "chị em",
            "chị ruột",
            "elder sister",
            "older sister",
            "sư tỷ",
            "tỷ tỷ",
            "chị gái",
            "senior",
            "senpai",
            "tiền bối",
            "senior/junior",
            "junior to senior",
            "student to senior",
        ),
    },
    {
        "self_references": {"mình", "tôi", "tớ"},
        "second_person": {"anh", "sư huynh", "ca ca"},
        "replacement_self_reference": "em",
        "cues": (
            "anh em",
            "anh ruột",
            "big brother",
            "elder brother",
            "older brother",
            "sư huynh",
            "ca ca",
            "huynh trưởng",
            "anh trai",
            "senior",
            "senpai",
            "tiền bối",
            "senior/junior",
            "junior to senior",
            "student to senior",
        ),
    },
    {
        "self_references": {"mình", "tôi", "tớ"},
        "second_person": {"bác", "chú", "dì", "thím", "dượng"},
        "replacement_self_reference": "cháu",
        "cues": (
            "bác",
            "chú",
            "dì",
            "thím",
            "dượng",
            "nephew",
            "niece",
            "uncle",
            "aunt",
        ),
    },
)
def _is_anchored_vietnamese_kinship_pair(details: str) -> bool:
    self_reference = _vietnamese_addressing_field(
        details,
        "self-reference",
    ).casefold()
    second_person = _vietnamese_addressing_field(
        details,
        "second-person pronoun",
    ).casefold()
    return (
        self_reference in _VIETNAMESE_ANCHORED_KINSHIP_SELF_REFERENCES
        and second_person in _VIETNAMESE_ANCHORED_KINSHIP_SECOND_PERSON
    )
def _vietnamese_addressing_anchor_score(details: str) -> int:
    clean = _clean_inline_text(details).casefold()
    if not clean:
        return 0

    score = 0
    if any(cue in clean for cue in _VIETNAMESE_KINSHIP_ADDRESSING_CUES):
        score += 3
    if any(cue in clean for cue in _VIETNAMESE_STATUS_ADDRESSING_CUES):
        score += 3
    if _is_anchored_vietnamese_kinship_pair(details):
        score += 3
    hierarchy_cues = (
        "age",
        "hierarchy",
        "older",
        "rank",
        "senior",
        "seniority",
        "status",
        "tiền bối",
        "trên tuổi",
        "sư huynh",
        "sư tỷ",
    )
    if any(cue in clean for cue in hierarchy_cues):
        score += 2
    return score
def _is_generic_vietnamese_neutral_pair(details: str) -> bool:
    self_reference = _vietnamese_addressing_field(
        details,
        "self-reference",
    ).casefold()
    second_person = _vietnamese_addressing_field(
        details,
        "second-person pronoun",
    ).casefold()
    vocative = _vietnamese_addressing_field(
        details,
        "vocative/address form",
    ).casefold()

    # Professional and formal titles are exempt from generic neutral downgrade status
    formal_titles = (
        "chủ tịch",
        "giám đốc",
        "quản gia",
        "bác sĩ",
        "luật sư",
        "thanh tra",
        "giáo sư",
        "quý khách",
        "ngài",
        "sư phụ",
        "trưởng lão",
    )
    if any(title in second_person or title in vocative for title in formal_titles):
        return False

    return (
        self_reference in _VIETNAMESE_GENERIC_NEUTRAL_SELF_REFERENCES
        and second_person in _VIETNAMESE_GENERIC_NEUTRAL_SECOND_PERSON
        and (
            not vocative
            or vocative == second_person
            or vocative in _VIETNAMESE_GENERIC_NEUTRAL_SECOND_PERSON
        )
    )
def _is_vietnamese_neutralization_of_kinship_pair(
    current_details: str,
    proposed_details: str,
) -> bool:
    if _is_vietnamese_attitude_shift(proposed_details):
        return False
    if not _is_anchored_vietnamese_kinship_pair(current_details):
        return False
    proposed_self_reference = _vietnamese_addressing_field(
        proposed_details,
        "self-reference",
    ).casefold()
    proposed_second_person = _vietnamese_addressing_field(
        proposed_details,
        "second-person pronoun",
    ).casefold()
    return (
        proposed_self_reference in _VIETNAMESE_GENERIC_NEUTRAL_SELF_REFERENCES
        and proposed_second_person in _VIETNAMESE_GENERIC_NEUTRAL_SECOND_PERSON
    )
_SITUATIONAL_CONTEXT_KEYWORDS = (
    "roleplay", "café", "cafe", "maid", "butler", "disguise", "cosplay",
    "acting", "stage play", "theatrical", "mock", "undercover", "pretending",
    "masquerade", "fake identity", "temporary", "situational", "scenario-bound",
    "transient", "sarcastic", "one-off", "performance", "costume", "game role"
)
def _is_situational_context_details(details: str) -> bool:
    text = str(details or "").casefold()
    return any(kw in text for kw in _SITUATIONAL_CONTEXT_KEYWORDS)
def _is_vietnamese_social_downgrade(
    current_details: str,
    proposed_details: str,
) -> bool:
    if not (
        _has_complete_vietnamese_addressing_details(current_details)
        and _has_complete_vietnamese_addressing_details(proposed_details)
    ):
        return False
    if _is_situational_context_details(proposed_details) and not _is_situational_context_details(current_details):
        return True
    current_anchor_score = _vietnamese_addressing_anchor_score(current_details)
    if current_anchor_score <= 0:
        return False
    proposed_anchor_score = _vietnamese_addressing_anchor_score(proposed_details)
    if _is_vietnamese_neutralization_of_kinship_pair(
        current_details,
        proposed_details,
    ):
        return True
    return (
        _is_generic_vietnamese_neutral_pair(proposed_details)
        and proposed_anchor_score < current_anchor_score
    )
def _filter_vietnamese_addressing_delta(
    proposed_addressing: str,
    alias_map: Dict[str, str],
    current_addressing: str = "",
    character_genders: Optional[Dict[str, str]] = None,
) -> str:
    """Drop Vietnamese addressing rows with incomplete or self-conflicting data."""
    current_details_by_key: Dict[Tuple[str, str, str], str] = {}
    for raw_line in current_addressing.splitlines():
        parsed = _parse_dynamic_relation(raw_line, alias_map)
        if parsed:
            relation_key, _, details = parsed
            current_details_by_key[relation_key] = details

    output: List[str] = []
    for raw_line in proposed_addressing.splitlines():
        raw_line = _repair_vietnamese_addressing_line(
            raw_line,
            alias_map,
            character_genders,
        )
        parsed = _parse_dynamic_relation(raw_line, alias_map)
        if not parsed:
            output.append(raw_line)
            continue
        relation_key, _, details = parsed
        is_delete = details.strip().rstrip(" .;:").casefold() in (
            _DYNAMIC_DELETE_VALUES
        )
        if is_delete:
            output.append(raw_line)
            continue
        current_details = current_details_by_key.get(relation_key, "")
        if current_details and _is_vietnamese_social_downgrade(
            current_details,
            details,
        ):
            continue
        if (
            _has_complete_vietnamese_addressing_details(details)
            and not _has_vietnamese_addressing_mismatch(details)
        ):
            output.append(raw_line)
    return "\n".join(output).strip()
def _remove_vietnamese_addressing_mismatches(
    addressing: str,
    alias_map: Dict[str, str],
) -> str:
    """Drop stored Vietnamese rows when their own paired-address data conflicts."""
    output: List[str] = []
    for raw_line in addressing.splitlines():
        parsed = _parse_dynamic_relation(raw_line, alias_map)
        if not parsed:
            output.append(raw_line)
            continue
        relation_key, _, details = parsed
        if not _has_vietnamese_addressing_mismatch(details):
            output.append(raw_line)
    return "\n".join(output).strip()
