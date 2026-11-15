"""
Structured data schema definitions for directed addressing context.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AddressingUpdateDelta:
    """
    Structured delta extracted from LLM output representing an addressing update.
    """
    speaker: str
    addressee: str
    self_pronoun: str
    second_pronoun: str
    vocative: str = ""
    register: str = "polite"
    confidence: float = 1.0
    evidence_quote: str = ""
    is_situational: bool = False
    source_forms: List[Dict[str, Any]] = field(default_factory=list)
    social_basis: List[str] = field(default_factory=list)
    scope: str = "durable"
    contract_version: int = 1
    dialogue_turn_id: str = ""
    action: str = "upsert"
    provenance: str = "unknown"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["AddressingUpdateDelta"]:
        """Parse dictionary safely into AddressingUpdateDelta instance."""
        if not isinstance(data, dict):
            return None
        speaker = str(data.get("speaker") or "").strip()
        addressee = str(data.get("addressee") or "").strip()
        self_pronoun = str(data.get("self_pronoun") or data.get("self") or "").strip()
        second_pronoun = str(data.get("second_pronoun") or data.get("second") or "").strip()

        if not speaker or not addressee or not self_pronoun or not second_pronoun:
            return None

        # Ignore group dialogue or unassigned addressee
        lowered_addressee = addressee.lower()
        if lowered_addressee in ("everyone", "group", "crowd", "all", "mọi người", "cả nhóm", "không rõ"):
            return None

        try:
            confidence = float(data.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 1.0

        raw_scope = str(data.get("scope") or "").strip().lower()
        is_sit = bool(data.get("is_situational") or raw_scope == "situational")
        try:
            contract_version = int(data.get("contract_version") or 1)
        except (TypeError, ValueError):
            contract_version = 1

        return cls(
            speaker=speaker,
            addressee=addressee,
            self_pronoun=self_pronoun,
            second_pronoun=second_pronoun,
            vocative=str(data.get("vocative") or "").strip(),
            register=str(data.get("register") or "polite").strip().lower(),
            confidence=confidence,
            evidence_quote=str(data.get("evidence_quote") or data.get("evidence") or "").strip(),
            is_situational=is_sit,
            source_forms=list(data.get("source_forms") or []),
            social_basis=list(data.get("social_basis") or []),
            scope=raw_scope if raw_scope in {"durable", "situational"} else "durable",
            contract_version=contract_version,
            dialogue_turn_id=str(data.get("dialogue_turn_id") or "").strip(),
            action=str(data.get("action") or "upsert").strip().lower(),
            provenance=str(data.get("provenance") or "unknown").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SITUATIONAL_KEYWORDS = (
    "roleplay", "café", "cafe", "maid", "butler", "disguise", "cosplay",
    "acting", "stage play", "theatrical", "mock", "undercover", "pretending",
    "masquerade", "fake identity", "temporary", "situational", "scenario-bound",
    "transient", "sarcastic", "one-off", "performance", "costume", "game role"
)


def is_situational_context(item: Any) -> bool:
    """Check whether an addressing delta, dict, or text description represents a temporary situational context."""
    if not item:
        return False
    if isinstance(item, AddressingUpdateDelta):
        if item.is_situational:
            return True
        text = f"{item.vocative} {item.evidence_quote} {item.register}".lower()
    elif isinstance(item, dict):
        if item.get("is_situational") or str(item.get("scope", "")).lower() == "situational":
            return True
        text = f"{item.get('vocative', '')} {item.get('evidence_quote', '')} {item.get('register', '')}".lower()
    else:
        text = str(item).lower()
    return any(kw in text for kw in SITUATIONAL_KEYWORDS)
