"""
Deterministic merge policy engine for directed addressing updates.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
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
        log_callback: Optional[Callable[[str, str], None]] = None
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

        # Rule 1: User Lock Override
        if existing and existing.get("is_locked"):
            msg = f"Rejected addressing update for {delta.speaker} -> {delta.addressee}: Locked by user."
            logger.info(f"[Merge Engine] {msg}")
            if log_callback:
                log_callback("addressing_rejected", msg)
            return False

        # Rule 2: Confidence Threshold Check
        if delta.confidence < self.confidence_threshold:
            msg = (
                f"Rejected addressing update for {delta.speaker} -> {delta.addressee}: "
                f"Confidence {delta.confidence:.2f} < threshold {self.confidence_threshold:.2f}."
            )
            logger.info(f"[Merge Engine] {msg}")
            if log_callback:
                log_callback("addressing_rejected", msg)
            return False

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
            if abs(old_rank - new_rank) > 1 and delta.confidence < 0.85 and not delta.evidence_quote:
                msg = (
                    f"Rejected register shift for {delta.speaker} -> {delta.addressee} "
                    f"({old_register} -> {new_register}) without quote evidence."
                )
                logger.info(f"[Merge Engine] {msg}")
                if log_callback:
                    log_callback("addressing_rejected", msg)
                return False

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
        log_callback: Optional[Callable[[str, str], None]] = None
    ) -> int:
        """Apply a batch of addressing update deltas. Returns number of applied updates."""
        applied_count = 0
        for delta in deltas:
            if self.apply_delta(
                translation_id,
                chunk_index,
                delta,
                trigger_source=trigger_source,
                log_callback=log_callback
            ):
                applied_count += 1
        return applied_count

