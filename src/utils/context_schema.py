"""
Structured data schema definitions for directed addressing context.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


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

        return cls(
            speaker=speaker,
            addressee=addressee,
            self_pronoun=self_pronoun,
            second_pronoun=second_pronoun,
            vocative=str(data.get("vocative") or "").strip(),
            register=str(data.get("register") or "polite").strip().lower(),
            confidence=confidence,
            evidence_quote=str(data.get("evidence_quote") or data.get("evidence") or "").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_addressing_deltas_from_text(raw_text: str) -> List[AddressingUpdateDelta]:
    """
    Safely extract AddressingUpdateDelta objects from raw LLM text output (JSON blocks or embedded JSON objects).
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    deltas: List[AddressingUpdateDelta] = []
    json_candidates = []

    # 1. Search for markdown code block ```json ... ```
    codeblock_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)
    for block in codeblock_matches:
        json_candidates.append(block.strip())

    # 2. Search for raw JSON objects {...}
    json_object_matches = re.findall(r"(\{[\s\S]*?\})", raw_text)
    for obj_str in json_object_matches:
        json_candidates.append(obj_str.strip())

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            items = []
            if isinstance(parsed, dict):
                if "addressing_updates" in parsed and isinstance(parsed["addressing_updates"], list):
                    items = parsed["addressing_updates"]
                elif "updates" in parsed and isinstance(parsed["updates"], list):
                    items = parsed["updates"]
                else:
                    items = [parsed]
            elif isinstance(parsed, list):
                items = parsed

            for item in items:
                delta = AddressingUpdateDelta.from_dict(item)
                if delta:
                    deltas.append(delta)
        except Exception:
            continue

    # Deduplicate deltas by (speaker, addressee)
    seen = set()
    unique_deltas = []
    for d in deltas:
        key = (d.speaker.lower(), d.addressee.lower())
        if key not in seen:
            seen.add(key)
            unique_deltas.append(d)

    return unique_deltas

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

        return cls(
            speaker=speaker,
            addressee=addressee,
            self_pronoun=self_pronoun,
            second_pronoun=second_pronoun,
            vocative=str(data.get("vocative") or "").strip(),
            register=str(data.get("register") or "polite").strip().lower(),
            confidence=confidence,
            evidence_quote=str(data.get("evidence_quote") or data.get("evidence") or "").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AddressingRuleState:
    """
    Persistent state representation of a directed addressing rule (speaker -> addressee).
    """
    speaker_name: str
    addressee_name: str
    self_pronoun: str
    target_pronoun: str
    vocative: str = ""
    register: str = "polite"
    confidence: float = 1.0
    is_locked: bool = False
    last_chunk_index: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AddressingRuleState":
        return cls(
            speaker_name=str(data.get("speaker_name") or data.get("speaker") or ""),
            addressee_name=str(data.get("addressee_name") or data.get("addressee") or ""),
            self_pronoun=str(data.get("self_pronoun") or ""),
            target_pronoun=str(data.get("target_pronoun") or data.get("second_pronoun") or ""),
            vocative=str(data.get("vocative") or ""),
            register=str(data.get("register") or "polite"),
            confidence=float(data.get("confidence", 1.0)),
            is_locked=bool(data.get("is_locked", False)),
            last_chunk_index=int(data.get("last_chunk_index", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
