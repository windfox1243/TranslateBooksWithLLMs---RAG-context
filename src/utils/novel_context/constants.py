"""Shared constants, vocabularies and regexes for novel context handling."""
from __future__ import annotations

import re
import logging

logger = logging.getLogger("novel_context")
WINDOWS_RESERVED_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
SAFE_FILENAME_PUNCTUATION = {"_", "-", "."}
DYNAMIC_STATE_START = "---DYNAMIC_STATE_START---"
DYNAMIC_STATE_END = "---DYNAMIC_STATE_END---"
CHARACTERS_SECTION = "## CHARACTERS & GENDERS"
ALIASES_SECTION = "## CHARACTER ALIASES"
NAME_MAP_SECTION = "## NAME TRANSLATION MAP"
GLOSSARY_SECTION = "## GLOSSARY & TERMINOLOGY"
ADDRESSING_SECTION = "## CURRENT ADDRESSING FORMS"
RELATIONSHIP_SECTION = "## RELATIONSHIP EVOLUTION"
_INVALID_CONTEXT_KEYS = {
    "",
    "-",
    "delete",
    "n/a",
    "na",
    "name",
    "none",
    "null",
    "unknown",
    "character",
    "character a",
    "character b",
    "canonical name",
    "recommended target term",
    "source term",
    "target term",
    "correction",
    "corrections",
    "identity link",
    "identity links",
    "new characters",
    "new glossary",
    "dynamic state",
    "characters & genders",
}
_BARE_NARRATIVE_ROLE_NAMES = {
    "hero",
    "main character",
    "main protagonist",
    "player character",
    "protagonist",
    "the hero",
    "the main character",
    "the main protagonist",
    "the player character",
    "the protagonist",
}
_TRANSFERABLE_ROLE_ONLY_NAMES = {
    "administrator",
    "game administrator",
    "gacha manager",
    "gacha room manager",
    "gacha room npc",
    "main player",
    "non player character",
    "non-player character",
    "npc",
    "player",
    "summon",
    "summoned character",
    "summoner",
    "summoner nim",
    "the summoner",
    "user",
}
_ADDRESS_TERM_SUFFIXES = {
    "nim",
    "sama",
    "san",
    "kun",
    "chan",
    "ssi",
    "sensei",
    "senpai",
    "sunbae",
}
_GENDER_LABELS = {
    "male",
    "female",
    "non-binary",
    "nonbinary",
    "unknown",
    "unspecified",
}
_SPECIFIC_GENDER_LABELS = {
    "male",
    "female",
    "non-binary",
    "nonbinary",
}
_NAME_TITLES = {
    "captain",
    "commander",
    "count",
    "countess",
    "doctor",
    "dr",
    "duchess",
    "duke",
    "emperor",
    "empress",
    "general",
    "king",
    "lady",
    "lieutenant",
    "lord",
    "major",
    "marshal",
    "prince",
    "princess",
    "professor",
    "queen",
    "sergeant",
}
_MULTIWORD_NAME_TITLES = (
    ("lieutenant", "colonel"),
    ("lieutenant", "commander"),
    ("major", "general"),
)
_ROLE_ONLY_TITLES = _NAME_TITLES | {
    "lieutenant colonel",
    "lieutenant commander",
    "major general",
}
_ROLE_TITLE_KEYS = tuple(
    sorted(_ROLE_ONLY_TITLES, key=lambda item: (-len(item.split()), item))
)
_UNIQUE_ROLE_TITLES = {"emperor", "empress", "king", "queen"}
_RELATIVE_AGE_WORDS = {
    "elder",
    "eldest",
    "older",
    "oldest",
    "younger",
    "youngest",
}
_GENERIC_ROLE_WORDS = {
    "attendant",
    "civilian",
    "commander",
    "corporal",
    "doctor",
    "guard",
    "knight",
    "manager",
    "medic",
    "officer",
    "private",
    "referee",
    "sergeant",
    "soldier",
    "soldiers",
    "victim",
    "student",
    "students",
    "classmate",
    "classmates",
    "teacher",
    "teachers",
    "bystander",
    "bystanders",
    "passerby",
    "passersby",
    "pedestrian",
    "pedestrians",
    "crowd",
    "people",
}
_CJK_GENERIC_ROLE_NAMES = {
    "同学",
    "学生",
    "男同学",
    "女同学",
    "男学生",
    "女学生",
    "男生",
    "女生",
    "老师",
    "教师",
    "路人",
    "行人",
}
_ENGLISH_GENERIC_ROLE_NAMES = {
    "student",
    "students",
    "classmate",
    "classmates",
    "teacher",
    "teachers",
    "bystander",
    "bystanders",
    "passerby",
    "passersby",
    "pedestrian",
    "pedestrians",
    "crowd",
    "people",
}
_CJK_NON_NAME_ADDRESS_LABELS = {
    "会长",
    "前辈",
    "后辈",
    "美少女",
    "学生会长",
}
_UNSET_NAME_TRANSLATION = "(not set)"
_AMBIGUOUS_SHORT_NAME_KEYS = {
    "baek",
    "choi",
    "dokgo",
    "han",
    "jang",
    "jeong",
    "jung",
    "kang",
    "kim",
    "kurosaki",
    "lee",
    "lim",
    "namgoong",
    "park",
    "seo",
    "shin",
    "yoon",
}
_SHORT_NAME_EVIDENCE_STOPWORDS = {
    "about",
    "against",
    "along",
    "another",
    "around",
    "character",
    "current",
    "currently",
    "figure",
    "former",
    "hostile",
    "other",
    "person",
    "protagonist",
    "student",
    "teacher",
    "toward",
    "towards",
    "whose",
    "with",
}
_INCIDENTAL_CHARACTER_MARKERS = {
    "abdominal wound",
    "advertisement",
    "attending physician",
    "author of the advertisement",
    "background",
    "body collection",
    "deceased",
    "doctor treating",
    "dying",
    "fallen",
    "generic",
    "incidental",
    "killed",
    "medical professional",
    "missing leg",
    "new recruit",
    "npc entity",
    "observing",
    "oversees",
    "one scene",
    "one-scene",
    "overseeing",
    "physician",
    "screaming in pain",
    "searching for",
    "severed arm",
    "soldier",
    "unnamed",
    "wounded",
    "programmed machine",
    "automated enforcement",
    "background npc",
    "minor npc",
    "unnamed npc",
    "throwaway",
    "one-off",
    "incidental npc",
    "minor role",
}
_EXPLICIT_NPC_MARKERS = {
    "programmed machine",
    "automated enforcement",
    "npc entity",
    "background npc",
    "minor npc",
    "unnamed npc",
    "throwaway",
    "one-off",
    "incidental npc",
}
_PHYSICAL_DESCRIPTOR_ANCHORS = {
    "armor",
    "armour",
    "badge",
    "black coat",
    "cloak",
    "coat",
    "dress",
    "eyepatch",
    "glasses",
    "hat",
    "hood",
    "jacket",
    "mask",
    "robe",
    "scar",
    "sword",
    "uniform",
    "weapon",
}
_PHYSICAL_DESCRIPTOR_RELATIONS = (
    "carrying",
    "holding",
    "in",
    "wearing",
    "with",
)
_GROUP_ENTITY_WORDS = {
    "academy",
    "agency",
    "army",
    "battalion",
    "clan",
    "company",
    "corporation",
    "country",
    "dynasty",
    "empire",
    "faction",
    "family",
    "force",
    "government",
    "guild",
    "house",
    "kingdom",
    "lineage",
    "military",
    "nation",
    "organization",
    "party",
    "school",
    "squad",
    "temple",
    "unit",
}
_RECURRING_CHARACTER_MARKERS = {
    "appears repeatedly",
    "canonical",
    "callsign",
    "code name",
    "codename",
    "important",
    "major character",
    "mentor",
    "named",
    "recurring",
    "repeatedly",
    "returns later",
    "source-named",
}
_NON_CHARACTER_METADATA_NAMES = {
    "author",
    "fan art",
    "hiatus",
    "notice",
    "serialization",
    "serialization time",
}
_NON_CHARACTER_METADATA_DETAIL_PATTERNS = (
    r"\b(?:author|writer|translator|illustrator)\s+of\s+(?:the\s+)?"
    r"(?:current\s+)?(?:work|novel|story|book|series)\b",
    r"\b(?:author|writer)\s*,\s*(?:writer\s+)?of\s+(?:the\s+)?"
    r"(?:current\s+)?(?:work|novel|story|book|series)\b",
    r"\b(?:posted|uploaded|published|serialized)\s+(?:chapter|episode|"
    r"notice|fan\s+art)\b",
)
_NON_CHARACTER_ITEM_DETAIL_PATTERNS = (
    r"\b(?:active|awakening|combat|passive|status|system)\s+skill\b",
    r"\bskill\s+that\b",
    r"\b(?:ability|buff|debuff|quest|stat|title)\s+that\b",
    r"\b(?:named|legendary|holy|magic|cursed|demonic|divine)\s+"
    r"(?:sword|weapon|artifact|artefact|relic|item|equipment)\b",
    r"\b(?:sword|weapon|artifact|artefact|relic|item|equipment|spell|"
    r"skill|ability|technique)\s+(?:used|wielded|owned|summoned|created|"
    r"forged|equipped|activated)\b",
)
_ROMANTIC_RELATION_LABELS = {
    "beloved",
    "boyfriend",
    "ex boyfriend",
    "ex girlfriend",
    "ex lover",
    "ex partner",
    "fiance",
    "fiancee",
    "fiancé",
    "fiancée",
    "former boyfriend",
    "former girlfriend",
    "former lover",
    "former partner",
    "girlfriend",
    "husband",
    "lover",
    "partner",
    "romantic partner",
    "significant other",
    "spouse",
    "wife",
}
_GENDERED_ROMANTIC_RELATION_LABELS = {
    "boyfriend": "Male",
    "ex boyfriend": "Male",
    "fiance": "Male",
    "fiancé": "Male",
    "former boyfriend": "Male",
    "girlfriend": "Female",
    "ex girlfriend": "Female",
    "fiancee": "Female",
    "fiancée": "Female",
    "former girlfriend": "Female",
    "wife": "Female",
    "husband": "Male",
}
_KINSHIP_GENDERS = {
    "father": "Male",
    "mother": "Female",
    "brother": "Male",
    "sister": "Female",
    "son": "Male",
    "daughter": "Female",
    "husband": "Male",
    "wife": "Female",
    "uncle": "Male",
    "aunt": "Female",
    "grandfather": "Male",
    "grandmother": "Female",
    "nephew": "Male",
    "niece": "Female",
}
_KINSHIP_WORDS = set(_KINSHIP_GENDERS.keys())
_DIRECT_GENDER_WORDS = {
    "male": "Male",
    "female": "Female",
    "boy": "Male",
    "girl": "Female",
    "man": "Male",
    "woman": "Female",
    "gentleman": "Male",
    "lady": "Female",
    "guy": "Male",
}
_ROMANTIC_RELATION_PATTERN = (
    r"(?:(?:ex|former)[-\s]+)?(?:girlfriend|boyfriend|lover|partner|"
    r"spouse|wife|husband|romantic\s+partner|significant\s+other|"
    r"beloved|fianc[eé]e?)"
)
_RELATIONSHIP_OBJECT_PRONOUN_VERBS = (
    r"cheated\s+on|abandoned|left|betrayed|dumped|broke\s+up\s+with"
)
_WORK_ENTITY_WORDS = {
    "advertisement",
    "anime",
    "app",
    "book",
    "film",
    "game",
    "manga",
    "movie",
    "novel",
    "series",
    "story",
    "website",
    "webtoon",
}
_WORK_ENTITY_NON_PERSON_ROLES = {
    "administrator",
    "avatar",
    "character",
    "entity",
    "manager",
    "operator",
    "player",
    "protagonist",
    "user",
}
_NARRATIVE_ROLE_NAME_PATTERN = re.compile(
    r"^(?:the\s+)?(?:(?:main\s+)?protagonist|main\s+character|"
    r"hero|player\s+character|fictional\s+character)\s+"
    r"(?:of|in|from)\s+.+$",
    flags=re.IGNORECASE,
)
_NARRATIVE_WORK_PATTERN = re.compile(
    r"\b(?:(?:main\s+)?protagonist|main\s+character|hero|player\s+character)"
    r"\s+(?:of|in|from)\s+(?:the\s+)?(?:(?:game|novel|story|series)\s+)?"
    r"[\"'“”‘’]?(?P<work>[^;,.\"'“”‘’]+)",
    flags=re.IGNORECASE,
)
