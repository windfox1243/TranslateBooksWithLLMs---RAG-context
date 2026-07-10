"""
Deterministic merge policy engine for directed addressing updates.
"""

import logging
from typing import Optional, Dict, Any, List, Callable, Iterable
from src.utils.context_schema import AddressingUpdateDelta, AddressingRuleState
from src.persistence.database import Database
import src.config as config

logger = logging.getLogger("context_merge_engine")

# Hierarchy of registers for stability comparison
REGISTER_HIERARCHY = {
    "formal": 4,
    "polite": 3,
    "intimate": 2,
    "casual": 2,
    "vulgar": 1,
    "hostile": 1,
}

TRUSTED_PAIR_SOURCES = {
    "job_context_load",
    "manual",
    "manual_context",
    "markdown_context",
    "rest_api",
    "user_manual",
}

VI_SENIOR_FORMS = {"anh", "chị", "thầy", "cô", "sếp", "bác", "chú", "ông", "bà", "ngài"}
VI_JUNIOR_FORMS = {"em", "cháu", "con", "hậu bối"}
VI_PEER_FORMS = {"cậu", "bạn", "tớ", "mình", "tao", "mày"}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def _is_unspecified(value: Any) -> bool:
    return _norm(value) in {"", "unspecified", "unknown", "none", "null", "-"}


def _matches_label(candidate: str, label: str, language: str = "") -> bool:
    if not candidate or not label:
        return False
    if _norm(candidate) == _norm(label):
        return True
    try:
        from src.utils.text_matching import active_label_matches_name

        return (
            active_label_matches_name(candidate, label, language)
            or active_label_matches_name(label, candidate, language)
        )
    except Exception:
        return False


def _matches_any_label(value: str, labels: Iterable[str], language: str = "") -> bool:
    return any(_matches_label(value, label, language) for label in labels if label)


def _dialogue_state_pair(dialogue_attribution: Optional[Dict[str, Any]]) -> tuple[str, str]:
    if not isinstance(dialogue_attribution, dict):
        return "", ""
    state_after = dialogue_attribution.get("state_after") or {}
    if not isinstance(state_after, dict):
        return "", ""
    speaker = str(state_after.get("speaker") or "").strip()
    addressee = str(state_after.get("addressee") or "").strip()
    if _norm(speaker) in {"unknown", "none", "null"}:
        speaker = ""
    if _norm(addressee) in {"unknown", "none", "null"}:
        addressee = ""
    return speaker, addressee


def _vi_pair_direction(self_pronoun: str, target_pronoun: str) -> str:
    self_key = _norm(self_pronoun)
    target_key = _norm(target_pronoun)
    if self_key in VI_JUNIOR_FORMS and target_key in VI_SENIOR_FORMS:
        return "junior_to_senior"
    if self_key in VI_SENIOR_FORMS and target_key in VI_JUNIOR_FORMS:
        return "senior_to_junior"
    if self_key in VI_PEER_FORMS and target_key in VI_PEER_FORMS:
        return "peer"
    if self_key and self_key == target_key:
        return "same"
    return ""


class ContextMergeEngine:
    """
    Applies deterministic business rules to merge LLM addressing deltas into DB state.
    """

    def __init__(self, db: Optional[Database] = None, confidence_threshold: Optional[float] = None):
        self.db = db or Database()
        if confidence_threshold is None:
            conf_val = getattr(config, "ADDRESSING_MERGE_CONFIDENCE_THRESHOLD", 0.80)
            try:
                self.confidence_threshold = float(conf_val)
            except (ValueError, TypeError):
                self.confidence_threshold = 0.80
        else:
            self.confidence_threshold = float(confidence_threshold)

    def apply_delta(
        self,
        translation_id: str,
        chunk_index: int,
        delta: AddressingUpdateDelta,
        trigger_source: str = "llm_delta",
        log_callback: Optional[Callable[[str, str], None]] = None,
        target_language: str = "",
        known_character_names: Optional[Iterable[str]] = None,
        active_character_names: Optional[Iterable[str]] = None,
        dialogue_attribution: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate and merge a single AddressingUpdateDelta into persistent database state.

        Returns True if state was updated, False if rejected by policy rules.
        """
        if not delta:
            return False

        existing_rules = self.db.get_addressing_rules(translation_id)
        existing_rule_map = {
            (r["speaker_name"].lower(), r["addressee_name"].lower()): r
            for r in existing_rules
        }

        pair_key = (delta.speaker.lower(), delta.addressee.lower())
        existing = existing_rule_map.get(pair_key)
        reverse = existing_rule_map.get((delta.addressee.lower(), delta.speaker.lower()))

        def reject(reason: str) -> bool:
            msg = (
                f"Rejected addressing update for {delta.speaker} -> {delta.addressee} "
                f"[chunk {chunk_index + 1}]: {reason}"
            )
            logger.info(f"[Merge Engine] {msg}")
            self.db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=chunk_index,
                speaker_name=delta.speaker,
                addressee_name=delta.addressee,
                old_state=existing,
                new_state={
                    "status": "rejected",
                    "reason": reason,
                    "speaker_name": delta.speaker,
                    "addressee_name": delta.addressee,
                    "self_pronoun": delta.self_pronoun,
                    "target_pronoun": delta.second_pronoun,
                    "vocative": delta.vocative,
                    "register": delta.register,
                    "confidence": delta.confidence,
                },
                trigger_source=f"{trigger_source}:rejected",
                evidence_quote=delta.evidence_quote,
                confidence=delta.confidence,
            )
            if log_callback:
                log_callback("addressing_rejected", msg)
            return False

        try:
            from src.utils.language_profiles import get_language_profile

            profile = get_language_profile(target_language)
        except Exception:
            profile = None
        requires_paired_forms = bool(
            profile and profile.addressing_family == "vietnamese"
        )
        trusted_pair_source = trigger_source in TRUSTED_PAIR_SOURCES
        strong_evidence = trusted_pair_source or (
            trigger_source == "llm_delta" and bool(str(delta.evidence_quote or "").strip())
        )
        try:
            from src.utils.relationship_reasoning_engine import (
                relationship_support_for_addressing,
            )

            graph_support = relationship_support_for_addressing(
                self.db,
                translation_id,
                delta.speaker,
                delta.addressee,
            )
        except Exception:
            graph_support = {
                "supported": False,
                "hierarchy": "unknown",
                "edges": [],
            }

        if requires_paired_forms and _is_unspecified(delta.self_pronoun):
            return reject(
                "target language requires a complete paired address form; "
                "self-reference is unspecified."
            )

        known_labels = list(known_character_names or [])
        active_labels = list(active_character_names or [])
        dialogue_speaker, dialogue_addressee = _dialogue_state_pair(dialogue_attribution)
        if dialogue_speaker:
            active_labels.append(dialogue_speaker)
        if dialogue_addressee:
            active_labels.append(dialogue_addressee)

        active_pair_matches = (
            dialogue_speaker
            and dialogue_addressee
            and _matches_label(delta.speaker, dialogue_speaker, target_language)
            and _matches_label(delta.addressee, dialogue_addressee, target_language)
        )
        dialogue_pair_reversed = (
            dialogue_speaker
            and dialogue_addressee
            and _matches_label(delta.speaker, dialogue_addressee, target_language)
            and _matches_label(delta.addressee, dialogue_speaker, target_language)
        )
        if dialogue_pair_reversed and not strong_evidence:
            return reject(
                "local dialogue attribution indicates the opposite speaker/addressee direction."
            )

        if known_labels:
            speaker_known = _matches_any_label(delta.speaker, known_labels, target_language)
            addressee_known = _matches_any_label(delta.addressee, known_labels, target_language)
            speaker_active = _matches_any_label(delta.speaker, active_labels, target_language)
            addressee_active = _matches_any_label(delta.addressee, active_labels, target_language)
            if not ((speaker_known and addressee_known) or (speaker_active and addressee_active)):
                return reject(
                    "speaker/addressee pair is not supported by known character context "
                    "or trusted active dialogue participants."
                )

        if not existing and not (
            trusted_pair_source
            or active_pair_matches
            or strong_evidence
            or graph_support.get("supported")
        ):
            return reject(
                "new addressing pair lacks trusted source, source evidence, or active dialogue support."
            )

        if requires_paired_forms and trigger_source not in {
            "manual",
            "rest_api",
            "user_manual",
        }:
            graph_hierarchy = graph_support.get("hierarchy", "unknown")
            incoming_direction = _vi_pair_direction(
                delta.self_pronoun,
                delta.second_pronoun,
            )
            expected_direction = {
                "source_senior": "senior_to_junior",
                "source_junior": "junior_to_senior",
                "peer": "peer",
            }.get(graph_hierarchy, "")
            if (
                incoming_direction
                and expected_direction
                and incoming_direction != expected_direction
            ):
                return reject(
                    "paired address hierarchy contradicts the accepted relationship graph "
                    f"({graph_hierarchy})."
                )

        # Rule 1: User Lock Override
        if existing and existing.get("is_locked"):
            return reject("locked by user.")

        # Rule 2: Confidence Threshold Check
        if delta.confidence < self.confidence_threshold:
            return reject(
                f"confidence {delta.confidence:.2f} < threshold {self.confidence_threshold:.2f}."
            )

        # Rule 3: Register Stability Policy
        if existing:
            old_self = existing.get("self_pronoun")
            old_second = existing.get("target_pronoun")
            old_register = existing.get("register", "polite").lower()
            new_register = delta.register.lower()

            if (
                old_self == delta.self_pronoun
                and old_second == delta.second_pronoun
                and old_register == new_register
            ):
                return False

            old_rank = REGISTER_HIERARCHY.get(old_register, 3)
            new_rank = REGISTER_HIERARCHY.get(new_register, 3)
            if abs(old_rank - new_rank) > 1 and not strong_evidence:
                return reject(
                    f"register shift {old_register} -> {new_register} lacks trusted evidence."
                )

        if (
            reverse
            and profile
            and profile.addressing_family == "vietnamese"
            and not strong_evidence
        ):
            incoming_direction = _vi_pair_direction(
                delta.self_pronoun,
                delta.second_pronoun,
            )
            reverse_direction = _vi_pair_direction(
                reverse.get("self_pronoun", ""),
                reverse.get("target_pronoun", ""),
            )
            if (
                incoming_direction
                and reverse_direction
                and incoming_direction == reverse_direction
                and incoming_direction != "peer"
            ):
                return reject(
                    "reverse pair already has the same hierarchy direction; "
                    "incoming pair looks like a speaker/addressee swap."
                )

        # Rule 4: Temporary & Situational Context Protection Policy
        from src.utils.context_schema import is_situational_context
        is_incoming_situational = is_situational_context(delta)

        if existing:
            existing_is_situational = is_situational_context(existing)

            if is_incoming_situational and not existing_is_situational:
                return reject(
                    f"identified as temporary/situational context "
                    f"('{delta.vocative}' / '{delta.register}'); preserving durable baseline rule."
                )

        old_state_dict = existing if existing else None
        new_state_dict = {
            "speaker_name": delta.speaker,
            "addressee_name": delta.addressee,
            "self_pronoun": delta.self_pronoun,
            "target_pronoun": delta.second_pronoun,
            "vocative": delta.vocative,
            "register": delta.register,
            "confidence": delta.confidence,
        }

        success = self.db.upsert_addressing_rule(
            translation_id=translation_id,
            speaker_name=delta.speaker,
            addressee_name=delta.addressee,
            self_pronoun=delta.self_pronoun,
            target_pronoun=delta.second_pronoun,
            vocative=delta.vocative,
            register=delta.register,
            confidence=delta.confidence,
            is_locked=0 if not existing else existing.get("is_locked", 0),
            chunk_index=chunk_index,
        )

        if success:
            self.db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=chunk_index,
                speaker_name=delta.speaker,
                addressee_name=delta.addressee,
                old_state=old_state_dict,
                new_state=new_state_dict,
                trigger_source=trigger_source,
                evidence_quote=delta.evidence_quote,
                confidence=delta.confidence,
            )
            merged_msg = (
                f"Updated addressing rule: {delta.speaker} -> {delta.addressee} "
                f"('{delta.self_pronoun}' / '{delta.second_pronoun}') [chunk {chunk_index + 1}]"
            )
            logger.info(f"[Merge Engine] {merged_msg}")
            if log_callback:
                log_callback("addressing_merged", merged_msg)
            return True

        return False

    def apply_batch_deltas(
        self,
        translation_id: str,
        chunk_index: int,
        deltas: List[AddressingUpdateDelta],
        trigger_source: str = "llm_delta",
        log_callback: Optional[Callable[[str, str], None]] = None,
        target_language: str = "",
        known_character_names: Optional[Iterable[str]] = None,
        active_character_names: Optional[Iterable[str]] = None,
        dialogue_attribution: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Apply a batch of addressing update deltas. Returns number of applied updates."""
        applied_count = 0
        for delta in deltas:
            if self.apply_delta(
                translation_id,
                chunk_index,
                delta,
                trigger_source=trigger_source,
                log_callback=log_callback,
                target_language=target_language,
                known_character_names=known_character_names,
                active_character_names=active_character_names,
                dialogue_attribution=dialogue_attribution,
            ):
                applied_count += 1
        return applied_count
