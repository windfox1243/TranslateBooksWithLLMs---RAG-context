"""Pair-level reconciliation for retained relationship/addressing evidence."""

from __future__ import annotations

from typing import Optional

from src.persistence.database import Database
from src.utils.relationship_reasoning_engine import relationship_support_for_addressing


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


class ContextReconciler:
    """Promote safe materialized rules while retaining their observations."""

    def __init__(self, db: Database):
        self.db = db

    def reconcile_pair(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
        *,
        target_language: str = "",
        chunk_index: int = 0,
        trigger_source: str = "context_reconciler",
    ) -> bool:
        if "viet" not in target_language.casefold() and target_language.casefold() != "vi":
            return False
        provisional = next((
            item for item in self.db.get_addressing_rules(
                translation_id, validation_status="provisional",
            )
            if str(item.get("speaker_name") or "").casefold() == speaker_name.casefold()
            and str(item.get("addressee_name") or "").casefold() == addressee_name.casefold()
        ), None)
        if not provisional or provisional.get("is_locked"):
            return False
        support = relationship_support_for_addressing(
            self.db, translation_id, speaker_name, addressee_name,
        )
        if support.get("conflict"):
            return False
        source_node = self.db.get_relationship_node_by_name(
            translation_id, speaker_name,
        ) or {}
        target_node = self.db.get_relationship_node_by_name(
            translation_id, addressee_name,
        ) or {}
        pair = _vietnamese_pair(
            str(support.get("hierarchy") or "unknown"),
            str(source_node.get("gender") or "unknown").casefold(),
            str(target_node.get("gender") or "unknown").casefold(),
        )
        if not pair:
            return False
        evidence = self.db.get_addressing_evidence(
            translation_id, speaker_name, addressee_name,
        )
        direct = next((
            item for item in evidence
            if str(item.get("usage") or "").casefold()
            in {"direct_address", "second_person"}
            and str(item.get("source_form") or "").strip()
        ), None)
        if not direct:
            return False
        old_state = dict(provisional)
        self_pronoun, target_pronoun = pair
        promoted = self.db.upsert_addressing_rule(
            translation_id=translation_id,
            speaker_name=speaker_name,
            addressee_name=addressee_name,
            self_pronoun=self_pronoun,
            target_pronoun=target_pronoun,
            vocative=str(provisional.get("vocative") or direct.get("source_form") or ""),
            register=str(provisional.get("register") or "polite"),
            social_basis=[
                "addressee is senior to speaker"
                if support.get("hierarchy") == "source_junior"
                else "speaker is senior to addressee"
            ],
            scope=str(provisional.get("scope") or "durable"),
            contract_version=max(5, int(provisional.get("contract_version") or 1)),
            confidence=max(0.80, float(support.get("confidence") or 0.0)),
            is_locked=0,
            chunk_index=chunk_index,
            notes=str(provisional.get("notes") or ""),
            validation_status="active",
            validation_reason="reconciled_from_accepted_relationship",
            provenance=trigger_source,
        )
        if not promoted:
            return False
        new_state = next((
            item for item in self.db.get_addressing_rules(translation_id)
            if str(item.get("speaker_name") or "").casefold() == speaker_name.casefold()
            and str(item.get("addressee_name") or "").casefold() == addressee_name.casefold()
        ), {})
        self.db.resolve_addressing_evidence(
            translation_id, speaker_name, addressee_name, "promoted",
        )
        self.db.add_context_audit_log(
            translation_id=translation_id,
            chunk_index=chunk_index,
            speaker_name=speaker_name,
            addressee_name=addressee_name,
            old_state=old_state,
            new_state=new_state,
            trigger_source=trigger_source,
            evidence_quote=str(direct.get("evidence_quote") or ""),
            confidence=float(new_state.get("confidence") or 0.8),
        )
        return True

    def reconcile_translation(
        self,
        translation_id: str,
        *,
        target_language: str = "",
        chunk_index: int = 0,
    ) -> int:
        promoted = 0
        for rule in list(self.db.get_addressing_rules(
            translation_id, validation_status="provisional",
        )):
            promoted += int(self.reconcile_pair(
                translation_id,
                str(rule.get("speaker_name") or ""),
                str(rule.get("addressee_name") or ""),
                target_language=target_language,
                chunk_index=chunk_index,
            ))
        return promoted
