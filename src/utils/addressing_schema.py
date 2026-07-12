"""Structured directed-addressing contract used by context contract v2."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ADDRESSING_CONTRACT_VERSION = 3

_REGISTER_VALUES = {
    "neutral",
    "formal",
    "polite",
    "casual",
    "intimate",
    "hostile",
    "vulgar",
    "archaic",
    "familial",
}
_SCOPE_VALUES = {"durable", "situational"}
_USAGE_VALUES = {
    "direct_address",
    "second_person",
    "self_reference",
    "indirect_reference",
    "unknown",
}


def _clean(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _bounded_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class AddressingSourceForm:
    """One exact source-language form supporting a directed address rule."""

    text: str
    usage: str = "direct_address"
    evidence_quote: str = ""
    source_language: str = ""
    scope: str = "durable"
    confidence: float = 0.5

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        default_evidence: str = "",
        default_language: str = "",
        default_scope: str = "durable",
        default_confidence: float = 0.5,
    ) -> Optional["AddressingSourceForm"]:
        if isinstance(value, str):
            text = _clean(value, 160)
            if not text:
                return None
            return cls(
                text=text,
                evidence_quote=_clean(default_evidence),
                source_language=_clean(default_language, 80),
                scope=default_scope,
                confidence=default_confidence,
            )
        if not isinstance(value, dict):
            return None
        text = _clean(
            value.get("text")
            or value.get("source_form")
            or value.get("form"),
            160,
        )
        if not text:
            return None
        usage = _clean(value.get("usage") or "direct_address", 40).casefold()
        if usage not in _USAGE_VALUES:
            usage = "unknown"
        scope = _clean(value.get("scope") or default_scope, 40).casefold()
        if scope not in _SCOPE_VALUES:
            scope = default_scope
        return cls(
            text=text,
            usage=usage,
            evidence_quote=_clean(
                value.get("evidence_quote") or default_evidence
            ),
            source_language=_clean(
                value.get("source_language") or default_language,
                80,
            ),
            scope=scope,
            confidence=_bounded_confidence(
                value.get("confidence"),
                default_confidence,
            ),
        )


@dataclass
class AddressingCandidateV2:
    """Evidence-bearing speaker-to-addressee addressing update."""

    speaker: str
    addressee: str
    self_reference: str = ""
    second_person: str = ""
    vocative: str = ""
    source_forms: List[AddressingSourceForm] = field(default_factory=list)
    register: str = "neutral"
    social_basis: List[str] = field(default_factory=list)
    notes: str = ""
    scope: str = "durable"
    evidence_quote: str = ""
    dialogue_turn_id: str = ""
    confidence: float = 0.5
    action: str = "upsert"
    provenance: str = "llm_context"

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        source_language: str = "",
        provenance: str = "llm_context",
    ) -> Optional["AddressingCandidateV2"]:
        if not isinstance(data, dict):
            return None
        speaker = _clean(data.get("speaker") or data.get("source"), 160)
        addressee = _clean(data.get("addressee") or data.get("target"), 160)
        action = _clean(data.get("action") or "upsert", 20).casefold()
        if action not in {"upsert", "delete"}:
            action = "upsert"
        if not speaker or not addressee:
            return None

        target = data.get("target_form") or data.get("target") or {}
        if not isinstance(target, dict):
            target = {}
        self_reference = _clean(
            target.get("self_reference")
            or data.get("self_reference")
            or data.get("self_pronoun"),
            100,
        )
        second_person = _clean(
            target.get("second_person")
            or target.get("addressee_reference")
            or target.get("second_person_pronoun")
            or data.get("addressee_reference")
            or data.get("second_person")
            or data.get("second_pronoun"),
            100,
        )
        vocative = _clean(
            target.get("vocative")
            or target.get("address_form")
            or data.get("vocative"),
            160,
        )
        if action == "upsert" and (not self_reference or not second_person):
            return None

        register = _clean(data.get("register") or "neutral", 40).casefold()
        if register not in _REGISTER_VALUES:
            register = "neutral"
        scope = _clean(data.get("scope") or "durable", 40).casefold()
        if scope not in _SCOPE_VALUES:
            scope = "durable"
        evidence_quote = _clean(data.get("evidence_quote"))
        confidence = _bounded_confidence(data.get("confidence"), 0.5)

        raw_basis = data.get("social_basis") or data.get("basis") or []
        if isinstance(raw_basis, str):
            raw_basis = re.split(r"[,;|]", raw_basis)
        social_basis = []
        for item in raw_basis if isinstance(raw_basis, list) else []:
            clean = _clean(item, 100)
            if clean and clean.casefold() not in {
                value.casefold() for value in social_basis
            }:
                social_basis.append(clean)

        raw_forms = data.get("source_forms")
        if raw_forms is None:
            raw_form = data.get("source_form")
            raw_forms = [raw_form] if raw_form else []
        if isinstance(raw_forms, (str, dict)):
            raw_forms = [raw_forms]
        source_forms = []
        for item in raw_forms if isinstance(raw_forms, list) else []:
            parsed = AddressingSourceForm.from_value(
                item,
                default_evidence=evidence_quote,
                default_language=source_language,
                default_scope=scope,
                default_confidence=confidence,
            )
            if parsed and parsed.text.casefold() not in {
                value.text.casefold() for value in source_forms
            }:
                source_forms.append(parsed)

        return cls(
            speaker=speaker,
            addressee=addressee,
            self_reference=self_reference,
            second_person=second_person,
            vocative=vocative,
            source_forms=source_forms,
            register=register,
            social_basis=social_basis,
            notes=_clean(data.get("notes"), 500),
            scope=scope,
            evidence_quote=evidence_quote,
            dialogue_turn_id=_clean(data.get("dialogue_turn_id"), 120),
            confidence=confidence,
            action=action,
            provenance=_clean(data.get("provenance") or provenance, 80),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["addressee_reference"] = self.second_person
        return result

    def to_delta(self):
        from src.utils.context_schema import AddressingUpdateDelta

        return AddressingUpdateDelta(
            speaker=self.speaker,
            addressee=self.addressee,
            self_pronoun=self.self_reference,
            second_pronoun=self.second_person,
            vocative=self.vocative,
            register=self.register,
            confidence=self.confidence,
            evidence_quote=self.evidence_quote,
            is_situational=self.scope == "situational",
            source_forms=[asdict(item) for item in self.source_forms],
            social_basis=list(self.social_basis),
            scope=self.scope,
            contract_version=ADDRESSING_CONTRACT_VERSION,
            dialogue_turn_id=self.dialogue_turn_id,
            action=self.action,
            provenance=self.provenance,
        )


def parse_addressing_candidate_block(
    text: str,
    *,
    source_language: str = "",
) -> Tuple[List[AddressingCandidateV2], str]:
    """Parse the v2 addressing JSON block without accepting prose as state."""

    raw = str(text or "").strip()
    if not raw:
        return [], "absent"
    fence = re.search(
        r"```(?:json)?\s*(.*?)```",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        raw = fence.group(1).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], "invalid_json"
    if not isinstance(payload, dict):
        return [], "invalid_contract"
    updates = payload.get("updates")
    if not isinstance(updates, list):
        return [], "invalid_contract"
    candidates = []
    for item in updates:
        candidate = AddressingCandidateV2.from_dict(
            item,
            source_language=source_language,
        )
        if candidate:
            candidates.append(candidate)
    return candidates, "json"


def context_contract_version(prompt_options: Optional[Dict[str, Any]]) -> int:
    """Resolve the persisted context contract; absent means legacy v1."""

    try:
        return int((prompt_options or {}).get("context_contract_version", 1))
    except (TypeError, ValueError):
        return 1
