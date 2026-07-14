"""Deterministic, character-agnostic social hierarchy extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from src.core.context.contracts import SocialHierarchyEvidence
from src.persistence.database import Database
from src.utils.addressing_schema import AddressingCandidateV2
from src.utils.context_merge_engine import ContextMergeEngine
from src.utils.relationship_reasoning_engine import RelationshipReasoningEngine
from src.utils.relationship_schema import RelationshipCandidate
from src.utils.text_matching import reference_mentions_label


_SENIOR_FORMS = {
    "senpai", "sempai", "sensei", "sunbae", "sunbae-nim",
    "先輩", "前辈", "前輩", "学长", "學長", "学姐", "學姐",
    "师兄", "師兄", "师姐", "師姐", "선배", "선배님",
    "senior", "mentor", "teacher", "superior", "upperclassman",
    "upperclasswoman", "older brother", "older sister",
}
_JUNIOR_FORMS = {
    "kouhai", "kohai", "hoobae", "后辈", "後輩", "学弟", "學弟",
    "学妹", "學妹", "师弟", "師弟", "师妹", "師妹", "후배",
    "junior", "student", "subordinate",
    "younger brother", "younger sister",
}
_NEUTRAL_SUFFIXES = {"san", "chan", "kun", "sama", "ssi", "nim"}
_OLDER_KIN_FORMS = {"older brother", "older sister"}
_YOUNGER_KIN_FORMS = {"younger brother", "younger sister"}


def _turns(dialogue_attribution: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        item for item in (dialogue_attribution or {}).get("turns") or []
        if isinstance(item, dict)
    ]


def _source_form(cue: str, forms: Iterable[str]) -> str:
    for form in sorted(forms, key=len, reverse=True):
        match = re.search(
            rf"(?<!\w)[\w'’.-]*{re.escape(form)}(?!\w)",
            cue,
            re.IGNORECASE,
        )
        if match:
            return match.group(0).strip(" .,!?\"'“”‘’")
    return ""


def _bounded_scene(source_text: str, cue: str, radius: int = 650) -> str:
    start = source_text.find(cue)
    if start < 0:
        start = source_text.casefold().find(cue.casefold())
    if start < 0:
        return ""
    return source_text[max(0, start - radius):min(len(source_text), start + len(cue) + radius)]


def _possessive_senior_scene(
    scene: str,
    speaker: str,
    addressee: str,
    source_language: str,
) -> bool:
    """Accept only a bounded two-participant scene with an explicit senior cue."""

    if not scene or not (
        reference_mentions_label(speaker, scene, source_language)
        and reference_mentions_label(addressee, scene, source_language)
    ):
        return False
    folded = scene.casefold()
    possessive = re.search(
        r"\b(?:her|his|their)\s+(?:admired\s+)?(?:senpai|senior|mentor|teacher)\b",
        folded,
    )
    return bool(possessive)


def extract_social_hierarchy_evidence(
    source_text: str,
    dialogue_attribution: Optional[Dict[str, Any]],
    *,
    source_language: str = "",
) -> List[SocialHierarchyEvidence]:
    """Extract directed hierarchy without inferring chronological age."""

    results: List[SocialHierarchyEvidence] = []
    seen = set()
    for turn in _turns(dialogue_attribution):
        speaker = str(turn.get("speaker") or "").strip()
        addressee = str(turn.get("addressee") or "").strip()
        cue = str(turn.get("cue") or "").strip()
        try:
            confidence = float(turn.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if (
            confidence < 0.80
            or not speaker
            or not addressee
            or "unknown" in {speaker.casefold(), addressee.casefold()}
            or not cue
        ):
            continue

        senior_form = _source_form(cue, _SENIOR_FORMS)
        junior_form = _source_form(cue, _JUNIOR_FORMS)
        scene = _bounded_scene(source_text, cue)
        if not senior_form and _possessive_senior_scene(
            scene, speaker, addressee, source_language,
        ):
            # The direct address proves the pair; adjacent narration proves
            # which member is the senior. Neutral honorifics remain neutral by
            # themselves and are accepted only with this independent cue.
            neutral = _source_form(cue, _NEUTRAL_SUFFIXES)
            if neutral:
                senior_form = neutral

        if senior_form:
            explicit_age = any(
                form in senior_form.casefold() for form in _OLDER_KIN_FORMS
            )
            evidence = SocialHierarchyEvidence(
                source=speaker,
                target=addressee,
                hierarchy="source_junior",
                rank_relation="source_lower",
                relative_age="source_younger" if explicit_age else "unknown",
                evidence_quote=cue,
                source_form=senior_form,
                dialogue_turn_id=str(turn.get("id") or ""),
                confidence=min(1.0, confidence),
                basis="explicit senior honorific or bounded senior narration",
            )
        elif junior_form:
            explicit_age = any(
                form in junior_form.casefold() for form in _YOUNGER_KIN_FORMS
            )
            evidence = SocialHierarchyEvidence(
                source=speaker,
                target=addressee,
                hierarchy="source_senior",
                rank_relation="source_higher",
                relative_age="source_older" if explicit_age else "unknown",
                evidence_quote=cue,
                source_form=junior_form,
                dialogue_turn_id=str(turn.get("id") or ""),
                confidence=min(1.0, confidence),
                basis="explicit junior honorific",
            )
        else:
            continue
        key = (evidence.source.casefold(), evidence.target.casefold(), evidence.hierarchy)
        if key not in seen:
            seen.add(key)
            results.append(evidence)
    return results


def _vietnamese_pair(
    hierarchy: str,
    source_gender: str,
    target_gender: str,
) -> Optional[tuple[str, str]]:
    if hierarchy == "source_junior":
        if target_gender == "female":
            return "em", "chị"
        if target_gender == "male":
            return "em", "anh"
    if hierarchy == "source_senior":
        if source_gender == "female":
            return "chị", "em"
        if source_gender == "male":
            return "anh", "em"
    return None


def apply_social_hierarchy_evidence(
    *,
    translation_id: str,
    db: Database,
    source_text: str,
    dialogue_attribution: Optional[Dict[str, Any]],
    source_language: str,
    target_language: str,
    chunk_index: int,
    known_character_names: Optional[Iterable[str]] = None,
    log_callback=None,
) -> List[SocialHierarchyEvidence]:
    """Persist accepted hierarchy and derive safe target addressing immediately."""

    evidence_items = extract_social_hierarchy_evidence(
        source_text,
        dialogue_attribution,
        source_language=source_language,
    )
    if not evidence_items:
        return []
    relationship_engine = RelationshipReasoningEngine(db=db)
    addressing_engine = ContextMergeEngine(db=db)
    known = list(known_character_names or [])
    active = [
        str(value)
        for turn in _turns(dialogue_attribution)
        for value in (turn.get("speaker"), turn.get("addressee"))
        if value and str(value).casefold() != "unknown"
    ]
    for item in evidence_items:
        candidate = RelationshipCandidate(
            source=item.source,
            target=item.target,
            relationship_type=item.relationship_type,
            direction="directed",
            scope=item.scope,
            hierarchy=item.hierarchy,
            relative_age=item.relative_age,
            rank_relation=item.rank_relation,
            intimacy="",
            register="polite",
            evidence_quote=item.evidence_quote,
            evidence_spans=[{
                "quote": item.evidence_quote,
                "role": "relationship",
                "dialogue_turn_id": item.dialogue_turn_id,
            }],
            confidence=item.confidence,
            provenance="deterministic_social_evidence",
            source_entity_type="character",
            target_entity_type="character",
            dialogue_turn_id=item.dialogue_turn_id,
            details=item.basis,
            parser_status="deterministic",
        )
        decision = relationship_engine.merge_candidate(
            translation_id,
            chunk_index,
            candidate,
            source_text=source_text,
            known_character_names=known,
            active_character_names=active,
            language=source_language,
            log_callback=log_callback,
        )
        if decision.status not in {"accepted", "unchanged"}:
            continue
        if decision.edge_id:
            db.resolve_relationship_evidence(
                translation_id, decision.edge_id, "promoted",
            )
        for conflict in db.get_relationship_conflicts(
            translation_id, status="open", limit=500,
        ):
            if {
                str(conflict.get("source_name") or "").casefold(),
                str(conflict.get("target_name") or "").casefold(),
            } == {item.source.casefold(), item.target.casefold()}:
                db.resolve_relationship_conflict(
                    translation_id,
                    int(conflict["id"]),
                )
        if "viet" not in target_language.casefold() and target_language.casefold() != "vi":
            continue
        source_node = db.get_relationship_node_by_name(translation_id, item.source) or {}
        target_node = db.get_relationship_node_by_name(translation_id, item.target) or {}
        pair = _vietnamese_pair(
            item.hierarchy,
            str(source_node.get("gender") or "unknown").casefold(),
            str(target_node.get("gender") or "unknown").casefold(),
        )
        if not pair:
            continue
        self_pronoun, second_person = pair
        raw_candidate = {
            "speaker": item.source,
            "addressee": item.target,
            "source_forms": [{
                "text": item.source_form,
                "usage": "direct_address",
                "evidence_quote": item.evidence_quote,
                "source_language": source_language,
                "scope": item.scope,
                "confidence": item.confidence,
            }],
            "target_form": {
                "self_reference": self_pronoun,
                "second_person": second_person,
                "vocative": item.source_form or "none",
            },
            "register": "polite",
            "social_basis": [item.basis],
            "scope": item.scope,
            "evidence_quote": item.evidence_quote,
            "dialogue_turn_id": item.dialogue_turn_id,
            "confidence": item.confidence,
            "action": "upsert",
            "contract_version": 5,
            "provenance": "deterministic_social_evidence",
        }
        addressing = AddressingCandidateV2.from_dict(
            raw_candidate,
            source_language=source_language,
        )
        if addressing:
            addressing_applied = addressing_engine.apply_delta(
                translation_id,
                chunk_index,
                addressing.to_delta(),
                trigger_source="deterministic_social_evidence",
                log_callback=log_callback,
                target_language=target_language,
                known_character_names=known,
                active_character_names=active,
                dialogue_attribution=dialogue_attribution,
                source_text=source_text,
                source_language=source_language,
            )
            if addressing_applied:
                db.resolve_addressing_evidence(
                    translation_id,
                    item.source,
                    item.target,
                    "promoted",
                )
    return evidence_items
