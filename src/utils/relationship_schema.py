"""Typed contracts for relationship graph extraction, merge, and projection."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


RELATIONSHIP_CONTRACT_VERSION = "1.0"

SYMMETRIC_RELATIONSHIP_TYPES = frozenset({
    "ally",
    "associated",
    "colleague",
    "enemy",
    "family",
    "friend",
    "peer",
    "rival",
    "romantic_partner",
    "sibling",
    "spouse",
})

ASYMMETRIC_RELATIONSHIP_INVERSES = {
    "child": "parent",
    "master": "servant",
    "mentor": "student",
    "parent": "child",
    "servant": "master",
    "student": "mentor",
    "subordinate": "superior",
    "superior": "subordinate",
}

SOURCE_SENIOR_TYPES = frozenset({"master", "mentor", "parent", "superior"})
SOURCE_JUNIOR_TYPES = frozenset({"child", "servant", "student", "subordinate"})
NON_CHARACTER_ENTITY_TYPES = frozenset({
    "ability",
    "artifact",
    "item",
    "object",
    "skill",
    "spell",
    "system",
    "ui_term",
    "weapon",
})

_ENTITY_CUES = {
    "weapon": ("weapon", "sword", "blade", "spear", "bow", "gun", "dagger", "axe"),
    "artifact": ("artifact", "relic", "treasure"),
    "spell": ("spell", "magic incantation"),
    "skill": ("skill", "technique", "combat move"),
    "ability": ("ability", "power"),
    "ui_term": ("ui term", "interface label", "menu label"),
    "system": ("system term", "system message", "game system"),
    "item": ("item", "equipment", "armor", "potion", "tool"),
}

_RELATIONSHIP_CUES = (
    ("parent", ("parent of", "father of", "mother of")),
    ("child", ("child of", "son of", "daughter of")),
    ("mentor", ("mentor of", "teacher of", "trainer of", "coach of")),
    ("student", ("student of", "trainee of", "disciple of")),
    ("superior", ("superior of", "commander of", "boss of")),
    ("subordinate", ("subordinate of", "employee of", "reports to")),
    ("master", ("master of",)),
    ("servant", ("servant of",)),
    ("sibling", ("sibling", "brother", "sister")),
    ("spouse", ("spouse", "husband", "wife", "married")),
    ("romantic_partner", ("lover", "romantic partner", "girlfriend", "boyfriend")),
    ("rival", ("rival",)),
    ("enemy", ("enemy", "hostile")),
    ("ally", ("ally", "allied")),
    ("friend", ("friend",)),
    ("colleague", ("colleague", "coworker", "co-worker")),
    ("peer", ("peer", "classmate")),
    ("family", ("family", "relative", "kin")),
)


def normalize_relationship_name(value: Any) -> str:
    """Return a Unicode-stable key for exact relationship identity matching."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.casefold().strip().split())


def clean_relationship_text(value: Any, max_length: int = 500) -> str:
    """Collapse whitespace and bound text stored in graph diagnostics."""

    clean = " ".join(str(value or "").strip().split())
    return clean[:max_length]


def classify_entity_type(value: Any, details: Any = "") -> str:
    """Classify canonical context metadata without language-specific fallbacks."""

    explicit = normalize_relationship_name(value)
    if explicit in {"character", "person"}:
        return "character"
    if explicit in NON_CHARACTER_ENTITY_TYPES:
        return explicit
    haystack = normalize_relationship_name(f"{value} {details}")
    for entity_type, cues in _ENTITY_CUES.items():
        if any(re.search(rf"\b{re.escape(cue)}\b", haystack) for cue in cues):
            return entity_type
    return "character"


def normalize_relationship_type(value: Any, details: Any = "") -> str:
    """Normalize a model or markdown relationship label to a stable type."""

    key = normalize_relationship_name(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "allies": "ally",
        "coworker": "colleague",
        "coworkers": "colleague",
        "enemies": "enemy",
        "friends": "friend",
        "husband": "spouse",
        "lover": "romantic_partner",
        "partners": "romantic_partner",
        "rivals": "rival",
        "siblings": "sibling",
        "wife": "spouse",
    }
    key = aliases.get(key, key)
    known = (
        SYMMETRIC_RELATIONSHIP_TYPES
        | frozenset(ASYMMETRIC_RELATIONSHIP_INVERSES)
        | {"addressing"}
    )
    if key in known:
        return key

    haystack = normalize_relationship_name(f"{value} {details}")
    for relationship_type, cues in _RELATIONSHIP_CUES:
        if any(cue in haystack for cue in cues):
            return relationship_type
    return "associated"


def default_relationship_direction(relationship_type: str) -> str:
    return "symmetric" if relationship_type in SYMMETRIC_RELATIONSHIP_TYPES else "directed"


def default_relationship_hierarchy(relationship_type: str) -> str:
    if relationship_type in SOURCE_SENIOR_TYPES:
        return "source_senior"
    if relationship_type in SOURCE_JUNIOR_TYPES:
        return "source_junior"
    if relationship_type == "peer":
        return "peer"
    return "unknown"


@dataclass
class RelationshipCandidate:
    """A structured relationship fact proposed by markdown, an LLM, or a user."""

    source: str
    target: str
    relationship_type: str = "associated"
    direction: str = "symmetric"
    scope: str = "durable"
    hierarchy: str = "unknown"
    relative_age: str = "unknown"
    rank_relation: str = "unknown"
    intimacy: str = "unknown"
    register: str = "neutral"
    evidence_quote: str = ""
    evidence_spans: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.5
    provenance: str = "llm_context"
    source_entity_type: str = "character"
    target_entity_type: str = "character"
    file_id: str = ""
    dialogue_turn_id: str = ""
    details: str = ""
    parser_status: str = "json"
    action: str = "upsert"
    judge_decision: str = ""
    judge_reason: str = ""

    def __post_init__(self) -> None:
        self.source = clean_relationship_text(self.source, 160)
        self.target = clean_relationship_text(self.target, 160)
        self.details = clean_relationship_text(self.details)
        self.relationship_type = normalize_relationship_type(
            self.relationship_type,
            self.details,
        )
        direction = normalize_relationship_name(self.direction)
        if direction not in {"directed", "symmetric"}:
            direction = default_relationship_direction(self.relationship_type)
        if self.relationship_type in SYMMETRIC_RELATIONSHIP_TYPES:
            direction = "symmetric"
        self.direction = direction
        scope = normalize_relationship_name(self.scope).replace("-", "_")
        if scope in {"temporary", "scene", "roleplay", "disguise", "dream", "quoted_history"}:
            scope = "situational"
        if scope not in {"durable", "situational"}:
            scope = "durable"
        self.scope = scope
        hierarchy = normalize_relationship_name(self.hierarchy).replace("-", "_")
        if hierarchy not in {"source_senior", "source_junior", "peer", "unknown"}:
            hierarchy = default_relationship_hierarchy(self.relationship_type)
        if hierarchy == "unknown":
            hierarchy = default_relationship_hierarchy(self.relationship_type)
        self.hierarchy = hierarchy
        relative_age = normalize_relationship_name(self.relative_age).replace("-", "_")
        if relative_age not in {
            "source_older", "source_younger", "same_age", "unknown",
        }:
            relative_age = "unknown"
        if relative_age == "unknown" and self.relationship_type == "sibling":
            relative_age = {
                "source_senior": "source_older",
                "source_junior": "source_younger",
                "peer": "same_age",
            }.get(self.hierarchy, "unknown")
        self.relative_age = relative_age
        rank_relation = normalize_relationship_name(self.rank_relation).replace("-", "_")
        if rank_relation not in {
            "source_higher", "source_lower", "equal", "unknown",
        }:
            rank_relation = "unknown"
        self.rank_relation = rank_relation
        self.intimacy = clean_relationship_text(self.intimacy, 80) or "unknown"
        self.register = clean_relationship_text(self.register, 80) or "neutral"
        self.evidence_quote = clean_relationship_text(self.evidence_quote)
        cleaned_spans = []
        for item in self.evidence_spans or []:
            if isinstance(item, str):
                item = {"quote": item}
            if not isinstance(item, dict):
                continue
            quote = clean_relationship_text(
                item.get("quote") or item.get("evidence_quote")
            )
            if not quote or any(span["quote"] == quote for span in cleaned_spans):
                continue
            cleaned_spans.append({
                "quote": quote,
                "role": clean_relationship_text(item.get("role"), 80),
                "dialogue_turn_id": clean_relationship_text(
                    item.get("dialogue_turn_id"), 120
                ),
            })
        if not cleaned_spans and self.evidence_quote:
            cleaned_spans.append({
                "quote": self.evidence_quote,
                "role": "",
                "dialogue_turn_id": self.dialogue_turn_id,
            })
        self.evidence_spans = cleaned_spans
        if not self.evidence_quote and cleaned_spans:
            self.evidence_quote = cleaned_spans[0]["quote"]
        self.provenance = clean_relationship_text(self.provenance, 80) or "llm_context"
        self.source_entity_type = classify_entity_type(
            self.source_entity_type,
            self.details,
        )
        self.target_entity_type = classify_entity_type(
            self.target_entity_type,
            self.details,
        )
        self.file_id = clean_relationship_text(self.file_id, 200)
        self.dialogue_turn_id = clean_relationship_text(self.dialogue_turn_id, 120)
        self.parser_status = clean_relationship_text(self.parser_status, 40) or "unknown"
        self.action = normalize_relationship_name(self.action) or "upsert"
        if self.action not in {"upsert", "delete", "quarantine"}:
            self.action = "upsert"
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.5

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        default_provenance: str = "llm_context",
        parser_status: str = "json",
    ) -> Optional["RelationshipCandidate"]:
        if not isinstance(data, dict):
            return None
        source = data.get("source") or data.get("character_a") or data.get("speaker")
        target = data.get("target") or data.get("character_b") or data.get("addressee")
        if not str(source or "").strip() or not str(target or "").strip():
            return None
        details = data.get("details") or data.get("description") or data.get("relationship") or ""
        relationship_type = (
            data.get("relationship_type")
            or data.get("type")
            or data.get("relationship")
            or "associated"
        )
        return cls(
            source=str(source),
            target=str(target),
            relationship_type=str(relationship_type),
            direction=str(data.get("direction") or ""),
            scope=str(data.get("scope") or "durable"),
            hierarchy=str(data.get("hierarchy") or "unknown"),
            relative_age=str(data.get("relative_age") or "unknown"),
            rank_relation=str(data.get("rank_relation") or "unknown"),
            intimacy=str(data.get("intimacy") or "unknown"),
            register=str(data.get("register") or "neutral"),
            evidence_quote=str(data.get("evidence_quote") or data.get("evidence") or ""),
            evidence_spans=list(data.get("evidence_spans") or []),
            confidence=data.get("confidence", 0.5),
            provenance=str(data.get("provenance") or default_provenance),
            source_entity_type=str(data.get("source_entity_type") or "character"),
            target_entity_type=str(data.get("target_entity_type") or "character"),
            file_id=str(data.get("file_id") or ""),
            dialogue_turn_id=str(data.get("dialogue_turn_id") or ""),
            details=str(details),
            parser_status=str(data.get("parser_status") or parser_status),
            action=str(data.get("action") or ("delete" if data.get("delete") else "upsert")),
            judge_decision=str(data.get("judge_decision") or ""),
            judge_reason=str(data.get("judge_reason") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipMergeDecision:
    status: str
    reason: str
    validator: str
    edge_id: Optional[int] = None
    audit_id: Optional[int] = None


@dataclass
class RelationshipProjection:
    prompt_text: str = ""
    selection_reasons: List[Dict[str, str]] = field(default_factory=list)
    fallback_reasons: List[str] = field(default_factory=list)
    edge_count: int = 0
    addressing_count: int = 0
    conflict_count: int = 0


@dataclass
class RelationshipJudgeResult:
    decision: str
    confidence: float
    reason: str = ""
    parse_status: str = "json"


def _strip_json_fence(raw: str) -> str:
    text = str(raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def parse_relationship_candidate_block(
    raw: Any,
    *,
    default_provenance: str = "llm_context",
) -> Tuple[List[RelationshipCandidate], str]:
    """Parse the typed relationship-candidate JSON block used by context LLMs."""

    payload: Any = raw
    if isinstance(raw, str):
        text = _strip_json_fence(raw)
        if not text:
            return [], "absent"
        candidates = [text]
        first_brace = text.find("{")
        first_bracket = text.find("[")
        starts = [index for index in (first_brace, first_bracket) if index >= 0]
        if starts:
            candidates.append(text[min(starts):])
        payload = None
        for candidate_text in candidates:
            try:
                payload = json.loads(candidate_text)
                break
            except (TypeError, ValueError, json.JSONDecodeError):
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate_text)
                try:
                    payload = json.loads(repaired)
                    break
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        if payload is None:
            return [], "invalid_json"

    if isinstance(payload, dict):
        items = (
            payload.get("relationships")
            or payload.get("relationship_candidates")
            or payload.get("candidates")
            or []
        )
    elif isinstance(payload, list):
        items = payload
    else:
        return [], "invalid_contract"
    if not isinstance(items, list):
        return [], "invalid_contract"

    parsed = []
    for item in items:
        candidate = RelationshipCandidate.from_dict(
            item,
            default_provenance=default_provenance,
            parser_status="json",
        )
        if candidate:
            parsed.append(candidate)
    return parsed, "json"
