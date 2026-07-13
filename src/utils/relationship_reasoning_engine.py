"""Deterministic relationship graph validation and merge engine."""

from __future__ import annotations

import inspect
import json
import re
import unicodedata
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.persistence.database import Database
from src.utils.progress_logging import emit_progress_log
from src.utils.relationship_schema import (
    ASYMMETRIC_RELATIONSHIP_INVERSES,
    NON_CHARACTER_ENTITY_TYPES,
    SYMMETRIC_RELATIONSHIP_TYPES,
    RelationshipCandidate,
    RelationshipJudgeResult,
    RelationshipMergeDecision,
    clean_relationship_text,
    default_relationship_hierarchy,
    normalize_relationship_name,
)
from src.utils.text_matching import active_label_matches_name, reference_mentions_label


TRUSTED_RELATIONSHIP_PROVENANCE = frozenset({
    "db_addressing",
    "job_context_load",
    "manual",
    "manual_context",
    "markdown_context",
    "relationship_markdown",
    "rest_api",
    "user_manual",
})

_HARD_CONFLICTS = {
    "parent": {"child", "romantic_partner", "sibling", "spouse"},
    "child": {"parent", "romantic_partner", "sibling", "spouse"},
    "sibling": {"child", "parent", "romantic_partner", "spouse"},
    "spouse": {"child", "parent", "sibling"},
    "romantic_partner": {"child", "parent", "sibling"},
    "mentor": {"student"},
    "student": {"mentor"},
    "superior": {"subordinate"},
    "subordinate": {"superior"},
    "master": {"servant"},
    "servant": {"master"},
}

_INVERSE_RELATIONSHIPS = {
    "parent": "child",
    "child": "parent",
    "mentor": "student",
    "student": "mentor",
    "master": "servant",
    "servant": "master",
    "superior": "subordinate",
    "subordinate": "superior",
}


def _candidate_evidence_quotes(candidate: RelationshipCandidate) -> list[str]:
    quotes = [
        str(item.get("quote") or "").strip()
        for item in candidate.evidence_spans or []
        if isinstance(item, dict)
    ]
    if candidate.evidence_quote and candidate.evidence_quote not in quotes:
        quotes.insert(0, candidate.evidence_quote)
    return [quote for quote in quotes if quote]


def _explicit_inverse_role(candidate: RelationshipCandidate) -> bool:
    details = normalize_relationship_name(candidate.details)
    cues = {
        "parent": {" child ", " son ", " daughter "},
        "child": {" parent ", " father ", " mother "},
        "mentor": {" student ", " pupil "},
        "student": {" mentor ", " teacher "},
        "master": {" servant "},
        "servant": {" master "},
        "superior": {" subordinate "},
        "subordinate": {" superior "},
    }.get(candidate.relationship_type, set())
    padded = f" {details} "
    return any(cue in padded for cue in cues)


def _names_match(candidate: str, known: str, language: str = "") -> bool:
    if normalize_relationship_name(candidate) == normalize_relationship_name(known):
        return True
    return (
        active_label_matches_name(candidate, known, language)
        or active_label_matches_name(known, candidate, language)
    )


def _matches_any_name(value: str, names: Iterable[str], language: str = "") -> bool:
    return any(_names_match(value, name, language) for name in names if name)


_EVIDENCE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "…": "...",
})


def _canonical_evidence(value: str) -> tuple[str, list[int]]:
    """Normalize evidence while retaining a map to original offsets."""

    output: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for original_index, original in enumerate(str(value or "")):
        expanded = unicodedata.normalize("NFKC", original).translate(
            _EVIDENCE_TRANSLATION
        ).casefold()
        for char in expanded:
            if char.isspace():
                pending_space = bool(output)
                continue
            if pending_space:
                output.append(" ")
                offsets.append(original_index)
                pending_space = False
            output.append(char)
            offsets.append(original_index)
    return "".join(output).strip(), offsets


def _match_source_quote(quote: str, source_text: str) -> tuple[str, Optional[int], Optional[int]]:
    value = str(quote or "").strip()
    source = str(source_text or "")
    if not value or not source:
        return "missing", None, None
    exact = source.find(value)
    if exact >= 0:
        return "exact", exact, exact + len(value)
    quote_key, _ = _canonical_evidence(value)
    source_key, source_offsets = _canonical_evidence(source)
    start = source_key.find(quote_key) if quote_key else -1
    if start < 0 or not source_offsets:
        return "unsupported", None, None
    end_key = start + len(quote_key) - 1
    if start >= len(source_offsets) or end_key >= len(source_offsets):
        return "unsupported", None, None
    return "normalized", source_offsets[start], source_offsets[end_key] + 1


def _quote_is_source_supported(quote: str, source_text: str) -> bool:
    return _match_source_quote(quote, source_text)[0] in {"exact", "normalized"}


def _edge_changed(existing: Dict[str, Any], candidate: RelationshipCandidate) -> bool:
    return any((
        normalize_relationship_name(existing.get("direction")) != candidate.direction,
        normalize_relationship_name(existing.get("hierarchy")) != candidate.hierarchy,
        normalize_relationship_name(existing.get("relative_age")) != candidate.relative_age,
        normalize_relationship_name(existing.get("rank_relation")) != candidate.rank_relation,
        normalize_relationship_name(existing.get("intimacy")) != normalize_relationship_name(candidate.intimacy),
        normalize_relationship_name(existing.get("register")) != normalize_relationship_name(candidate.register),
        normalize_relationship_name(existing.get("details")) != normalize_relationship_name(candidate.details),
    ))


class RelationshipReasoningEngine:
    """Validate, merge, quarantine, and audit structured relationship facts."""

    def __init__(self, db: Optional[Database] = None, confidence_threshold: float = 0.72):
        self.db = db or Database()
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    def _log(
        self,
        log_callback: Optional[Callable],
        event: str,
        message: str,
        *,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        emit_progress_log(
            log_callback,
            event,
            message,
            level=level,
            layer="relationship_reasoning",
            data=data or {},
        )

    def _record_conflict(
        self,
        translation_id: str,
        chunk_index: int,
        candidate: RelationshipCandidate,
        *,
        status: str,
        validator: str,
        reason: str,
        severity: str = "warning",
        edge_id: Optional[int] = None,
        log_callback: Optional[Callable] = None,
    ) -> RelationshipMergeDecision:
        conflict_id = self.db.add_relationship_conflict(
            translation_id=translation_id,
            source_name=candidate.source,
            target_name=candidate.target,
            severity=severity,
            validator=validator,
            reason=reason,
            remediation_hint=(
                "Provide an exact source quote or correct the relationship manually; "
                "locked user facts take precedence."
            ),
            candidate=candidate.to_dict(),
            chunk_index=chunk_index,
            edge_id=edge_id,
        )
        event = (
            "relationship_provisional" if status == "provisional"
            else "relationship_quarantined" if status == "quarantined"
            else "relationship_rejected"
        )
        self._log(
            log_callback,
            event,
            f"{status.title()} relationship {candidate.source} -> {candidate.target} "
            f"[{candidate.relationship_type}, chunk {chunk_index + 1}]: {reason}",
            level="warning",
            data={
                "chunk_index": chunk_index,
                "source": candidate.source,
                "target": candidate.target,
                "relationship_type": candidate.relationship_type,
                "validator": validator,
                "reason": reason,
                "confidence": candidate.confidence,
                "provenance": candidate.provenance,
            },
        )
        return RelationshipMergeDecision(
            status=status,
            reason=reason,
            validator=validator,
            edge_id=edge_id,
            audit_id=conflict_id,
        )

    def record_parse_failure(
        self,
        translation_id: str,
        chunk_index: int,
        parser_status: str,
        *,
        log_callback: Optional[Callable] = None,
    ) -> RelationshipMergeDecision:
        candidate = RelationshipCandidate(
            source="<context-update>",
            target="<relationship-contract>",
            relationship_type="associated",
            confidence=0.0,
            provenance="llm_context",
            parser_status=parser_status,
        )
        return self._record_conflict(
            translation_id,
            chunk_index,
            candidate,
            status="quarantined",
            validator="relationship_contract",
            reason=f"Relationship candidate contract could not be parsed ({parser_status}).",
            severity="error",
            log_callback=log_callback,
        )

    def register_node(
        self,
        translation_id: str,
        canonical_name: str,
        *,
        aliases: Optional[List[str]] = None,
        entity_type: str = "character",
        gender: str = "unknown",
        is_locked: bool = False,
    ) -> Optional[int]:
        normalized = normalize_relationship_name(canonical_name)
        if not normalized:
            return None
        return self.db.upsert_relationship_node(
            translation_id=translation_id,
            canonical_name=clean_relationship_text(canonical_name, 160),
            normalized_name=normalized,
            aliases=[clean_relationship_text(alias, 160) for alias in aliases or [] if alias],
            entity_type=entity_type,
            gender=gender,
            is_locked=int(bool(is_locked)),
        )

    def register_alias(
        self,
        translation_id: str,
        canonical_name: str,
        alias: str,
        *,
        alias_entity_type: str = "character",
        chunk_index: int = 0,
        log_callback: Optional[Callable] = None,
    ) -> bool:
        node = self.db.get_relationship_node_by_name(
            translation_id,
            normalize_relationship_name(canonical_name),
        )
        if not node:
            return False
        if node.get("entity_type") != "character" or alias_entity_type in NON_CHARACTER_ENTITY_TYPES:
            candidate = RelationshipCandidate(
                source=alias,
                target=canonical_name,
                relationship_type="associated",
                source_entity_type=alias_entity_type,
                provenance="markdown_context",
                confidence=1.0,
            )
            self._record_conflict(
                translation_id,
                chunk_index,
                candidate,
                status="rejected",
                validator="entity_type",
                reason="Objects, weapons, items, skills, and system terms cannot be character identity aliases.",
                severity="error",
                log_callback=log_callback,
            )
            return False
        return self.db.add_relationship_node_alias(
            translation_id,
            node["normalized_name"],
            clean_relationship_text(alias, 160),
        )

    def _resolve_node(
        self,
        translation_id: str,
        name: str,
        entity_type: str,
        known_character_names: Iterable[str],
        language: str,
        trusted: bool,
    ) -> Optional[Dict[str, Any]]:
        node = self.db.get_relationship_node_by_name(
            translation_id,
            normalize_relationship_name(name),
        )
        if node:
            return node
        if entity_type != "character":
            return None
        if not trusted and not _matches_any_name(name, known_character_names, language):
            return None
        node_id = self.register_node(
            translation_id,
            name,
            entity_type=entity_type,
        )
        if node_id is None:
            return None
        return self.db.get_relationship_node_by_name(
            translation_id,
            normalize_relationship_name(name),
        )

    def _quarantine_edge(
        self,
        translation_id: str,
        chunk_index: int,
        candidate: RelationshipCandidate,
        source_node: Optional[Dict[str, Any]],
        target_node: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        if not source_node or not target_node:
            return None
        existing = self.db.get_relationship_edges_for_pair(
            translation_id,
            source_node["normalized_name"],
            target_node["normalized_name"],
            statuses=["accepted"],
        )
        if any(edge.get("relationship_type") == candidate.relationship_type for edge in existing):
            return None
        return self.db.upsert_relationship_edge(
            translation_id=translation_id,
            source_node_id=source_node["id"],
            target_node_id=target_node["id"],
            relationship_type=candidate.relationship_type,
            direction=candidate.direction,
            scope=candidate.scope,
            hierarchy=candidate.hierarchy,
            relative_age=candidate.relative_age,
            rank_relation=candidate.rank_relation,
            intimacy=candidate.intimacy,
            register=candidate.register,
            confidence=candidate.confidence,
            status="quarantined",
            is_locked=0,
            chunk_index=chunk_index,
            provenance=candidate.provenance,
            details=candidate.details,
            evidence_tier="contradictory",
            reason_code="deterministic_conflict",
            validator_version=2,
        )

    def _provisional_edge(
        self, translation_id: str, chunk_index: int,
        candidate: RelationshipCandidate,
        source_node: Optional[Dict[str, Any]],
        target_node: Optional[Dict[str, Any]],
        *, reason_code: str, source_text: str = "",
    ) -> Optional[int]:
        if not source_node or not target_node:
            return None
        existing = self.db.get_relationship_edges_for_pair(
            translation_id, source_node["normalized_name"],
            target_node["normalized_name"], include_reverse=False,
        )
        same = next((
            edge for edge in existing
            if edge.get("relationship_type") == candidate.relationship_type
            and edge.get("scope") == candidate.scope
        ), None)
        supporting = {
            int(item.get("chunk_index", 0))
            for item in self.db.get_relationship_evidence(
                translation_id, same.get("id") if same else None,
            )
        } if same else set()
        supporting.add(int(chunk_index))
        edge_id = self.db.upsert_relationship_edge(
            translation_id=translation_id,
            source_node_id=source_node["id"], target_node_id=target_node["id"],
            relationship_type=candidate.relationship_type,
            direction=candidate.direction, scope=candidate.scope,
            hierarchy=candidate.hierarchy, relative_age=candidate.relative_age,
            rank_relation=candidate.rank_relation, intimacy=candidate.intimacy,
            register=candidate.register, confidence=candidate.confidence,
            status="provisional", is_locked=0, chunk_index=chunk_index,
            provenance=candidate.provenance, details=candidate.details,
            evidence_tier="indirect", reason_code=reason_code,
            supporting_units=len(supporting), validator_version=2,
        )
        for span in candidate.evidence_spans or [{"quote": candidate.evidence_quote}]:
            quote = str(span.get("quote") or "").strip()
            if not quote:
                continue
            match_kind, source_start, source_end = _match_source_quote(
                quote, source_text
            )
            self.db.add_relationship_evidence(
                translation_id=translation_id, edge_id=edge_id,
                chunk_index=chunk_index, evidence_quote=quote,
                provenance=candidate.provenance,
                parser_status=candidate.parser_status,
                confidence=candidate.confidence, file_id=candidate.file_id,
                dialogue_turn_id=str(
                    span.get("dialogue_turn_id") or candidate.dialogue_turn_id
                ),
                match_kind=match_kind, source_start=source_start,
                source_end=source_end,
            )
        return edge_id

    def merge_candidate(
        self,
        translation_id: str,
        chunk_index: int,
        candidate: RelationshipCandidate,
        *,
        source_text: str = "",
        known_character_names: Optional[Iterable[str]] = None,
        active_character_names: Optional[Iterable[str]] = None,
        language: str = "",
        log_callback: Optional[Callable] = None,
    ) -> RelationshipMergeDecision:
        """Validate and merge one relationship candidate into persistent state."""

        known_names = list(known_character_names or [])
        active_names = list(active_character_names or [])
        trusted = candidate.provenance in TRUSTED_RELATIONSHIP_PROVENANCE

        if not candidate.source or not candidate.target:
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="identity",
                reason="Relationship endpoints must both be named.",
                log_callback=log_callback,
            )
        if normalize_relationship_name(candidate.source) == normalize_relationship_name(candidate.target):
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="identity",
                reason="A relationship edge cannot connect a character to itself.",
                log_callback=log_callback,
            )
        if (
            candidate.source_entity_type in NON_CHARACTER_ENTITY_TYPES
            or candidate.target_entity_type in NON_CHARACTER_ENTITY_TYPES
        ):
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="entity_type",
                reason="Only character nodes can own relationship edges.",
                severity="error", log_callback=log_callback,
            )

        source_node = self._resolve_node(
            translation_id, candidate.source, candidate.source_entity_type,
            known_names + active_names, language, trusted,
        )
        target_node = self._resolve_node(
            translation_id, candidate.target, candidate.target_entity_type,
            known_names + active_names, language, trusted,
        )
        if not source_node or not target_node:
            edge_id = self._quarantine_edge(
                translation_id, chunk_index, candidate, source_node, target_node,
            )
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="provisional", validator="known_character",
                reason="Relationship endpoints await unambiguous canonical character evidence.",
                edge_id=edge_id, log_callback=log_callback,
            )
        if (
            source_node.get("entity_type") != "character"
            or target_node.get("entity_type") != "character"
        ):
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="entity_type",
                reason="A stored non-character entity cannot participate in a character relationship.",
                severity="error", log_callback=log_callback,
            )

        candidate = replace(
            candidate,
            source=source_node["canonical_name"],
            target=target_node["canonical_name"],
        )
        if not trusted and candidate.relationship_type in ASYMMETRIC_RELATIONSHIP_INVERSES:
            inverse_type = _INVERSE_RELATIONSHIPS.get(candidate.relationship_type)
            inverse_hierarchy = {
                "source_senior": "source_junior",
                "source_junior": "source_senior",
            }.get(default_relationship_hierarchy(candidate.relationship_type))
            supplied_signals = [
                value for value in (
                    candidate.hierarchy if candidate.hierarchy != "unknown" else None,
                    {
                        "source_older": "source_senior",
                        "source_younger": "source_junior",
                        "same_age": "peer",
                    }.get(candidate.relative_age),
                    {
                        "source_higher": "source_senior",
                        "source_lower": "source_junior",
                        "equal": "peer",
                    }.get(candidate.rank_relation),
                ) if value
            ]
            if inverse_type and supplied_signals and set(supplied_signals) == {inverse_hierarchy}:
                candidate = replace(
                    candidate,
                    relationship_type=inverse_type,
                    direction="directed",
                    judge_reason=(
                        f"Canonicalized inverse role from {_INVERSE_RELATIONSHIPS[inverse_type]} "
                        f"to {inverse_type}; evidence and hierarchy signals agree."
                    ),
                )
        expected_hierarchy = {
            "parent": "source_senior",
            "mentor": "source_senior",
            "master": "source_senior",
            "superior": "source_senior",
            "child": "source_junior",
            "student": "source_junior",
            "servant": "source_junior",
            "subordinate": "source_junior",
        }.get(candidate.relationship_type)
        age_hierarchy = {
            "source_older": "source_senior",
            "source_younger": "source_junior",
            "same_age": "peer",
        }.get(candidate.relative_age)
        rank_hierarchy = {
            "source_higher": "source_senior",
            "source_lower": "source_junior",
            "equal": "peer",
        }.get(candidate.rank_relation)
        # Age, institutional rank, social relationship, and conversational
        # hierarchy are independent dimensions. Only a role-implied hierarchy
        # constrains the relationship type itself.
        if (
            expected_hierarchy
            and candidate.hierarchy not in {"unknown", expected_hierarchy}
        ):
            edge_id = self._quarantine_edge(
                translation_id, chunk_index, candidate, source_node, target_node,
            )
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="quarantined", validator="semantic_hierarchy",
                reason=(
                    f"Relationship '{candidate.relationship_type}' requires "
                    f"hierarchy '{expected_hierarchy}', not "
                    f"'{candidate.hierarchy}'."
                ),
                severity="error", edge_id=edge_id,
                log_callback=log_callback,
            )
        if expected_hierarchy and candidate.hierarchy == "unknown":
            candidate = replace(candidate, hierarchy=expected_hierarchy)
        elif candidate.hierarchy == "unknown" and (age_hierarchy or rank_hierarchy):
            candidate = replace(candidate, hierarchy=age_hierarchy or rank_hierarchy)
        if candidate.direction == "symmetric" and source_node["id"] > target_node["id"]:
            source_node, target_node = target_node, source_node
            reversed_hierarchy = {
                "source_senior": "source_junior",
                "source_junior": "source_senior",
            }.get(candidate.hierarchy, candidate.hierarchy)
            reversed_age = {
                "source_older": "source_younger",
                "source_younger": "source_older",
            }.get(candidate.relative_age, candidate.relative_age)
            reversed_rank = {
                "source_higher": "source_lower",
                "source_lower": "source_higher",
            }.get(candidate.rank_relation, candidate.rank_relation)
            candidate = replace(
                candidate,
                source=source_node["canonical_name"],
                target=target_node["canonical_name"],
                hierarchy=reversed_hierarchy,
                relative_age=reversed_age,
                rank_relation=reversed_rank,
            )

        pair_edges = self.db.get_relationship_edges_for_pair(
            translation_id,
            source_node["normalized_name"],
            target_node["normalized_name"],
            include_reverse=True,
        )
        pair_edges = [
            edge for edge in pair_edges
            if edge.get("relationship_type") != "addressing"
        ]
        same_edge = next((
            edge for edge in pair_edges
            if edge["source_node_id"] == source_node["id"]
            and edge["target_node_id"] == target_node["id"]
            and edge.get("relationship_type") == candidate.relationship_type
            and edge.get("scope") == candidate.scope
        ), None)

        if candidate.action == "delete":
            deletable_edges = [
                edge for edge in pair_edges
                if edge.get("relationship_type") != "addressing"
            ]
            if not deletable_edges:
                return RelationshipMergeDecision(
                    status="unchanged", reason="Relationship edge does not exist.",
                    validator="delete", edge_id=None,
                )
            locked_edge = next(
                (edge for edge in deletable_edges if edge.get("is_locked")),
                None,
            )
            if locked_edge:
                return self._record_conflict(
                    translation_id, chunk_index, candidate,
                    status="rejected", validator="lock",
                    reason="Relationship edge is locked by the user.",
                    severity="error", edge_id=locked_edge["id"],
                    log_callback=log_callback,
                )
            deleted_ids = [
                edge["id"] for edge in deletable_edges
                if self.db.delete_relationship_edge(translation_id, edge["id"])
            ]
            return RelationshipMergeDecision(
                status="accepted" if deleted_ids else "rejected",
                reason=(
                    f"Deleted {len(deleted_ids)} relationship edge(s)."
                    if deleted_ids
                    else "Relationship edge could not be deleted."
                ),
                validator="delete",
                edge_id=deleted_ids[0] if deleted_ids else None,
            )

        if same_edge and same_edge.get("is_locked"):
            if not _edge_changed(same_edge, candidate):
                return RelationshipMergeDecision(
                    status="unchanged", reason="Locked relationship already matches.",
                    validator="lock", edge_id=same_edge["id"],
                )
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="lock",
                reason="Relationship edge is locked by the user.",
                severity="error", edge_id=same_edge["id"],
                log_callback=log_callback,
            )

        locked_pair = next((edge for edge in pair_edges if edge.get("is_locked")), None)
        if locked_pair and locked_pair.get("relationship_type") != candidate.relationship_type:
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="lock",
                reason="A locked relationship fact already defines this character pair.",
                severity="error", edge_id=locked_pair["id"],
                log_callback=log_callback,
            )

        if candidate.scope == "durable" and not trusted:
            evidence_quotes = _candidate_evidence_quotes(candidate)
            if not evidence_quotes:
                edge_id = self._provisional_edge(
                    translation_id, chunk_index, candidate, source_node, target_node,
                    reason_code="source_evidence_missing", source_text=source_text,
                )
                return self._record_conflict(
                    translation_id, chunk_index, candidate,
                    status="provisional", validator="source_evidence",
                    reason="Awaiting a source evidence span or independent corroboration.",
                    edge_id=edge_id, log_callback=log_callback,
                )
            unsupported_quotes = [
                quote for quote in evidence_quotes
                if not _quote_is_source_supported(quote, source_text)
            ]
            if unsupported_quotes:
                edge_id = self._provisional_edge(
                    translation_id, chunk_index, candidate, source_node, target_node,
                    reason_code="source_evidence_unmatched", source_text=source_text,
                )
                return self._record_conflict(
                    translation_id, chunk_index, candidate,
                    status="provisional", validator="source_evidence",
                    reason="Evidence could not yet be matched to the source unit.",
                    edge_id=edge_id, log_callback=log_callback,
                )
            participant_text = "\n".join([*evidence_quotes, source_text])
            source_supported = any(
                reference_mentions_label(label, participant_text, language)
                for label in [candidate.source, *(source_node.get("aliases") or [])]
                if label
            )
            target_supported = any(
                reference_mentions_label(label, participant_text, language)
                for label in [candidate.target, *(target_node.get("aliases") or [])]
                if label
            )
            evidence_roles = {
                str(item.get("role") or "").casefold()
                for item in candidate.evidence_spans or []
                if isinstance(item, dict)
            }
            known_keys = {
                normalize_relationship_name(name)
                for name in known_names
            }
            if (
                "source" in evidence_roles
                and normalize_relationship_name(candidate.source) in known_keys
            ):
                source_supported = True
            if (
                "target" in evidence_roles
                and normalize_relationship_name(candidate.target) in known_keys
            ):
                target_supported = True
            if "source" in evidence_roles and any(
                normalize_relationship_name(name)
                == normalize_relationship_name(candidate.source)
                for name in active_names
            ):
                source_supported = True
            if "target" in evidence_roles and any(
                normalize_relationship_name(name)
                == normalize_relationship_name(candidate.target)
                for name in active_names
            ):
                target_supported = True
            if not (source_supported and target_supported):
                edge_id = self._provisional_edge(
                    translation_id, chunk_index, candidate, source_node, target_node,
                    reason_code="participants_ambiguous", source_text=source_text,
                )
                return self._record_conflict(
                    translation_id, chunk_index, candidate,
                    status="provisional", validator="source_participants",
                    reason="Awaiting unambiguous evidence for both relationship participants.",
                    edge_id=edge_id, log_callback=log_callback,
                )

        if not trusted and candidate.judge_decision == "reject":
            edge_id = self._provisional_edge(
                translation_id, chunk_index, candidate, source_node, target_node,
                reason_code="judge_disagreement", source_text=source_text,
            )
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="provisional", validator="llm_judge",
                reason=(
                    "Optional LLM evidence judge rejected the ambiguous candidate; "
                    "additional evidence or manual review is required."
                ),
                edge_id=edge_id, log_callback=log_callback,
            )

        if (
            not trusted
            and candidate.confidence < self.confidence_threshold
            and candidate.scope != "durable"
        ):
            edge_id = self._provisional_edge(
                translation_id, chunk_index, candidate, source_node, target_node,
                reason_code="confidence_unconfirmed", source_text=source_text,
            )
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="provisional", validator="confidence",
                reason=(
                    f"Confidence {candidate.confidence:.2f} is below the "
                    f"{self.confidence_threshold:.2f} acceptance threshold."
                ),
                edge_id=edge_id, log_callback=log_callback,
            )

        if candidate.relationship_type in ASYMMETRIC_RELATIONSHIP_INVERSES:
            expected_reverse = ASYMMETRIC_RELATIONSHIP_INVERSES[candidate.relationship_type]
            inverse_family = {candidate.relationship_type, expected_reverse}
            for edge in pair_edges:
                is_reverse = (
                    edge["source_node_id"] == target_node["id"]
                    and edge["target_node_id"] == source_node["id"]
                    and edge.get("status") == "accepted"
                )
                existing_reverse_type = edge.get("relationship_type")
                if (
                    is_reverse
                    and existing_reverse_type in inverse_family
                    and existing_reverse_type != expected_reverse
                ):
                    return self._record_conflict(
                        translation_id, chunk_index, candidate,
                        status="rejected", validator="reverse_semantics",
                        reason=(
                            "Reverse relationship is incompatible: expected "
                            f"'{expected_reverse}', found '{edge.get('relationship_type')}'."
                        ),
                        severity="error", edge_id=edge["id"],
                        log_callback=log_callback,
                    )

        for edge in pair_edges:
            if edge.get("status") != "accepted" or edge.get("scope") != "durable":
                continue
            existing_type = str(edge.get("relationship_type") or "")
            same_direction = (
                edge["source_node_id"] == source_node["id"]
                and edge["target_node_id"] == target_node["id"]
            )
            if (
                same_direction
                and existing_type in _HARD_CONFLICTS.get(
                    candidate.relationship_type,
                    set(),
                )
            ):
                return self._record_conflict(
                    translation_id, chunk_index, candidate,
                    status="rejected", validator="relationship_compatibility",
                    reason=(
                        f"Durable relationship '{candidate.relationship_type}' conflicts "
                        f"with stored '{existing_type}'."
                    ),
                    severity="error", edge_id=edge["id"],
                    log_callback=log_callback,
                )
            if (
                edge["source_node_id"] == source_node["id"]
                and edge["target_node_id"] == target_node["id"]
                and edge.get("hierarchy") in {"source_senior", "source_junior"}
                and candidate.hierarchy in {"source_senior", "source_junior"}
                and edge.get("hierarchy") != candidate.hierarchy
                and not trusted
            ):
                return self._record_conflict(
                    translation_id, chunk_index, candidate,
                    status="quarantined", validator="hierarchy_flip",
                    reason="A durable seniority direction cannot flip without a trusted manual correction.",
                    edge_id=edge["id"], log_callback=log_callback,
                )

        if same_edge and not _edge_changed(same_edge, candidate) and same_edge.get("status") == "accepted":
            evidence_id = None
            for span in candidate.evidence_spans or [{"quote": candidate.evidence_quote}]:
                quote = str(span.get("quote") or "").strip()
                if not quote:
                    continue
                evidence_id = self.db.add_relationship_evidence(
                    translation_id=translation_id,
                    edge_id=same_edge["id"],
                    chunk_index=chunk_index,
                    evidence_quote=quote,
                    provenance=candidate.provenance,
                    parser_status=candidate.parser_status,
                    confidence=candidate.confidence,
                    file_id=candidate.file_id,
                    dialogue_turn_id=str(span.get("dialogue_turn_id") or candidate.dialogue_turn_id),
                    match_kind=_match_source_quote(quote, source_text)[0],
                    source_start=_match_source_quote(quote, source_text)[1],
                    source_end=_match_source_quote(quote, source_text)[2],
                )
            return RelationshipMergeDecision(
                status="unchanged", reason="Relationship already matches accepted graph state.",
                validator="deduplication", edge_id=same_edge["id"], audit_id=evidence_id,
            )

        edge_id = self.db.upsert_relationship_edge(
            translation_id=translation_id,
            source_node_id=source_node["id"],
            target_node_id=target_node["id"],
            relationship_type=candidate.relationship_type,
            direction=candidate.direction,
            scope=candidate.scope,
            hierarchy=candidate.hierarchy,
            relative_age=candidate.relative_age,
            rank_relation=candidate.rank_relation,
            intimacy=candidate.intimacy,
            register=candidate.register,
            confidence=candidate.confidence,
            status="accepted",
            is_locked=0 if not same_edge else same_edge.get("is_locked", 0),
            chunk_index=chunk_index,
            provenance=candidate.provenance,
            details=candidate.details,
            evidence_tier=("direct" if _candidate_evidence_quotes(candidate) else "trusted"),
            reason_code="validated",
            supporting_units=1,
            match_kind=(
                _match_source_quote(_candidate_evidence_quotes(candidate)[0], source_text)[0]
                if _candidate_evidence_quotes(candidate) else "trusted"
            ),
            validator_version=2,
        )
        if edge_id is None:
            return self._record_conflict(
                translation_id, chunk_index, candidate,
                status="rejected", validator="persistence",
                reason="Relationship graph persistence failed.",
                severity="error", log_callback=log_callback,
            )
        evidence_id = None
        for span in candidate.evidence_spans or [{"quote": candidate.evidence_quote}]:
            quote = str(span.get("quote") or "").strip()
            if not quote:
                continue
            evidence_id = self.db.add_relationship_evidence(
                translation_id=translation_id,
                edge_id=edge_id,
                chunk_index=chunk_index,
                evidence_quote=quote,
                provenance=candidate.provenance,
                parser_status=candidate.parser_status,
                confidence=candidate.confidence,
                file_id=candidate.file_id,
                dialogue_turn_id=str(span.get("dialogue_turn_id") or candidate.dialogue_turn_id),
                match_kind=_match_source_quote(quote, source_text)[0],
                source_start=_match_source_quote(quote, source_text)[1],
                source_end=_match_source_quote(quote, source_text)[2],
            )
        self._log(
            log_callback,
            "relationship_accepted",
            f"Accepted relationship {candidate.source} -> {candidate.target} "
            f"[{candidate.relationship_type}, chunk {chunk_index + 1}].",
            data={
                "chunk_index": chunk_index,
                "source": candidate.source,
                "target": candidate.target,
                "relationship_type": candidate.relationship_type,
                "scope": candidate.scope,
                "hierarchy": candidate.hierarchy,
                "confidence": candidate.confidence,
                "provenance": candidate.provenance,
            },
        )
        return RelationshipMergeDecision(
            status="accepted", reason="Relationship passed deterministic validation.",
            validator="accepted", edge_id=edge_id, audit_id=evidence_id,
        )

    def merge_candidates(
        self,
        translation_id: str,
        chunk_index: int,
        candidates: Iterable[RelationshipCandidate],
        **kwargs: Any,
    ) -> List[RelationshipMergeDecision]:
        return [
            self.merge_candidate(
                translation_id,
                chunk_index,
                candidate,
                **kwargs,
            )
            for candidate in candidates
            if candidate
        ]


def migrate_relationship_reasoning_v2(
    db: Optional[Database], translation_id: str,
) -> Dict[str, Any]:
    """Re-evaluate safe legacy quarantines once while preserving audit history."""

    migration_key = "relationship_validator_v2"
    if (
        db is None or not translation_id
        or not hasattr(db, "claim_reasoning_migration")
        or not db.claim_reasoning_migration(translation_id, migration_key)
    ):
        return {"status": "unchanged", "accepted": 0}
    accepted = 0
    reviewed = 0
    try:
        conflicts = db.get_relationship_conflicts(
            translation_id, status="open", limit=1000,
        )
        eligible_validators = {
            "semantic_hierarchy", "reverse_semantics", "source_evidence",
            "source_participants", "llm_judge", "confidence",
        }
        eligible = [
            item for item in conflicts
            if item.get("validator") in eligible_validators
            and isinstance(item.get("candidate"), dict)
        ]
        db_path = str(getattr(db, "db_path", "") or "")
        if eligible and db_path and db_path != ":memory:":
            backup_path = db_path + ".pre-v1.15.0-beta.43.bak"
            if not __import__("os").path.exists(backup_path):
                db.backup_to(backup_path)
        chunks = {
            int(item.get("chunk_index", 0)): item
            for item in db.get_chunks(translation_id)
        }
        known_names = [
            str(item.get("canonical_name") or "")
            for item in db.get_relationship_nodes(translation_id)
        ]
        engine = RelationshipReasoningEngine(db=db)
        for conflict in eligible:
            edge_id = conflict.get("edge_id")
            edge = next((
                item for item in db.get_relationship_edges(translation_id)
                if item.get("id") == edge_id
            ), None)
            if edge and edge.get("is_locked"):
                continue
            candidate = RelationshipCandidate.from_dict(
                conflict.get("candidate") or {},
                default_provenance="llm_context",
            )
            if candidate is None:
                continue
            chunk_index = int(conflict.get("chunk_index", 0) or 0)
            source_text = str(
                (chunks.get(chunk_index) or {}).get("original_text") or ""
            )
            decision = engine.merge_candidate(
                translation_id, chunk_index, candidate,
                source_text=source_text,
                known_character_names=known_names,
            )
            reviewed += 1
            if decision.status in {"accepted", "unchanged"}:
                db.resolve_relationship_conflict(
                    translation_id, int(conflict["id"])
                )
                accepted += 1
            elif (
                decision.status == "provisional"
                and decision.audit_id
                and int(decision.audit_id) != int(conflict["id"])
            ):
                db.resolve_relationship_conflict(
                    translation_id, int(conflict["id"])
                )
        db.finish_reasoning_migration(
            translation_id, migration_key, "completed",
            {"reviewed": reviewed, "accepted": accepted},
        )
        return {"status": "completed", "reviewed": reviewed, "accepted": accepted}
    except Exception as exc:
        db.finish_reasoning_migration(
            translation_id, migration_key, "failed",
            {"error": type(exc).__name__},
        )
        return {"status": "failed", "error": type(exc).__name__}


def relationship_support_for_addressing(
    db: Optional[Database],
    translation_id: str,
    speaker_name: str,
    addressee_name: str,
) -> Dict[str, Any]:
    """Resolve direct and explainable multi-hop seniority for one pair."""

    if db is None or not translation_id:
        return {
            "supported": False,
            "hierarchy": "unknown",
            "edges": [],
            "confidence": 0.0,
            "path": [],
            "conflict": False,
        }
    source_key = normalize_relationship_name(speaker_name)
    target_key = normalize_relationship_name(addressee_name)
    source_node = db.get_relationship_node_by_name(translation_id, source_key)
    target_node = db.get_relationship_node_by_name(translation_id, target_key)
    if not source_node or not target_node:
        return {
            "supported": False,
            "hierarchy": "unknown",
            "edges": [],
            "confidence": 0.0,
            "path": [],
            "conflict": False,
        }

    all_edges = [
        edge for edge in db.get_relationship_edges(
            translation_id,
            statuses=["accepted"],
        )
        if edge.get("relationship_type") != "addressing"
    ]
    pair_edges = [
        edge for edge in all_edges
        if {edge.get("source_node_id"), edge.get("target_node_id")}
        == {source_node["id"], target_node["id"]}
    ]

    def inverse(value: str) -> str:
        return {
            "source_senior": "source_junior",
            "source_junior": "source_senior",
        }.get(value, value)

    def edge_hierarchy(edge: Dict[str, Any]) -> str:
        hierarchy = str(edge.get("hierarchy") or "unknown")
        if hierarchy != "unknown":
            return hierarchy
        relative_age = str(edge.get("relative_age") or "unknown")
        if relative_age == "source_older":
            return "source_senior"
        if relative_age == "source_younger":
            return "source_junior"
        if relative_age == "same_age":
            return "peer"
        rank_relation = str(edge.get("rank_relation") or "unknown")
        if rank_relation == "source_higher":
            return "source_senior"
        if rank_relation == "source_lower":
            return "source_junior"
        if rank_relation == "equal":
            return "peer"
        relationship_type = str(edge.get("relationship_type") or "")
        if relationship_type in {"parent", "mentor", "master", "superior"}:
            return "source_senior"
        if relationship_type in {"child", "student", "servant", "subordinate"}:
            return "source_junior"
        if relationship_type in {"friend", "peer"}:
            return "peer"
        return "unknown"

    def edge_priority(edge: Dict[str, Any]) -> int:
        if edge.get("is_locked"):
            return 4
        if str(edge.get("provenance") or "") in TRUSTED_RELATIONSHIP_PROVENANCE:
            return 3
        return 2

    adjacency: Dict[int, List[Dict[str, Any]]] = {}
    for edge in all_edges:
        hierarchy = edge_hierarchy(edge)
        if hierarchy == "unknown":
            continue
        confidence = float(edge.get("confidence") or 0.0)
        if edge.get("relationship_type") in {"friend", "peer"}:
            confidence = min(confidence, 0.80)
        base = {
            "edge_id": edge.get("id"),
            "relationship_type": edge.get("relationship_type"),
            "confidence": confidence,
            "priority": edge_priority(edge),
            "source_name": edge.get("source_name"),
            "target_name": edge.get("target_name"),
        }
        adjacency.setdefault(edge["source_node_id"], []).append({
            **base,
            "to": edge["target_node_id"],
            "hierarchy": hierarchy,
        })
        adjacency.setdefault(edge["target_node_id"], []).append({
            **base,
            "to": edge["source_node_id"],
            "hierarchy": inverse(hierarchy),
        })

    def combine(left: str, right: str) -> str:
        if left == "peer":
            return right
        if right == "peer":
            return left
        if left == right and left in {"source_senior", "source_junior"}:
            return left
        return "unknown"

    paths: List[Dict[str, Any]] = []

    def walk(node_id: int, relation: str, arcs: List[Dict[str, Any]], seen: set) -> None:
        if len(arcs) >= 3:
            return
        for arc in adjacency.get(node_id, []):
            next_id = int(arc["to"])
            if next_id in seen:
                continue
            combined = arc["hierarchy"] if not arcs else combine(
                relation,
                arc["hierarchy"],
            )
            if combined == "unknown":
                continue
            next_arcs = [*arcs, arc]
            if next_id == target_node["id"]:
                confidence = min(
                    float(item["confidence"]) for item in next_arcs
                ) * (0.9 ** max(0, len(next_arcs) - 1))
                paths.append({
                    "hierarchy": combined,
                    "confidence": round(confidence, 4),
                    "priority": (
                        min(int(item["priority"]) for item in next_arcs)
                        if len(next_arcs) == 1
                        else 1
                    ),
                    "derived": len(next_arcs) > 1,
                    "path": [
                        {
                            key: item.get(key)
                            for key in (
                                "edge_id", "source_name", "target_name",
                                "relationship_type", "hierarchy", "confidence",
                            )
                        }
                        for item in next_arcs
                    ],
                })
            else:
                walk(next_id, combined, next_arcs, {*seen, next_id})

    walk(source_node["id"], "unknown", [], {source_node["id"]})
    if not paths:
        return {
            "supported": bool(pair_edges),
            "hierarchy": "unknown",
            "edges": pair_edges,
            "confidence": 0.0,
            "path": [],
            "conflict": False,
        }
    paths.sort(
        key=lambda item: (item["priority"], item["confidence"]),
        reverse=True,
    )
    best = paths[0]
    competing = next((
        item for item in paths[1:]
        if item["priority"] == best["priority"]
        and item["hierarchy"] != best["hierarchy"]
        and abs(item["confidence"] - best["confidence"]) <= 0.10
    ), None)
    if competing:
        if best["derived"]:
            db.upsert_relationship_derivation(
                translation_id,
                source_node["canonical_name"],
                target_node["canonical_name"],
                "unknown",
                best["confidence"],
                best["path"],
                "conflicting_seniority_paths",
                status="quarantined",
                chunk_index=max(
                    [int(edge.get("last_chunk_index") or 0) for edge in all_edges]
                    or [0]
                ),
            )
        existing_conflict = next((
            item for item in db.get_relationship_conflicts(
                translation_id,
                status="open",
                limit=500,
            )
            if item.get("validator") == "derived_seniority"
            and {
                normalize_relationship_name(item.get("source_name")),
                normalize_relationship_name(item.get("target_name")),
            } == {source_key, target_key}
        ), None)
        if not existing_conflict:
            db.add_relationship_conflict(
                translation_id=translation_id,
                source_name=source_node["canonical_name"],
                target_name=target_node["canonical_name"],
                severity="warning",
                validator="derived_seniority",
                reason=(
                    "Competing seniority paths differ in direction within the "
                    "0.10 confidence quarantine margin."
                ),
                remediation_hint=(
                    "Add exact source evidence or lock a manual relationship fact."
                ),
                candidate={
                    "best_path": best,
                    "competing_path": competing,
                },
                chunk_index=max(
                    [int(edge.get("last_chunk_index") or 0) for edge in all_edges]
                    or [0]
                ),
            )
        return {
            "supported": True,
            "hierarchy": "unknown",
            "edges": pair_edges,
            "confidence": best["confidence"],
            "path": best["path"],
            "conflict": True,
            "competing_path": competing["path"],
            "basis": "conflicting_seniority_paths",
            "source_gender": source_node.get("gender", "unknown"),
            "target_gender": target_node.get("gender", "unknown"),
        }
    basis = "direct_relationship" if not best["derived"] else "derived_relationship_chain"
    if best["derived"] and best["confidence"] >= 0.72:
        db.upsert_relationship_derivation(
            translation_id,
            source_node["canonical_name"],
            target_node["canonical_name"],
            best["hierarchy"],
            best["confidence"],
            best["path"],
            basis,
            chunk_index=max(
                [int(edge.get("last_chunk_index") or 0) for edge in all_edges]
                or [0]
            ),
        )
    return {
        "supported": True,
        "hierarchy": best["hierarchy"],
        "edges": pair_edges,
        "confidence": best["confidence"],
        "path": best["path"],
        "conflict": False,
        "derived": best["derived"],
        "basis": basis,
        "source_gender": source_node.get("gender", "unknown"),
        "target_gender": target_node.get("gender", "unknown"),
    }


def relationship_candidate_needs_llm_judge(candidate: RelationshipCandidate) -> bool:
    """Return whether optional semantic evidence classification may add value."""

    return bool(
        candidate.relationship_type == "associated"
        or candidate.confidence < 0.85
        or not candidate.evidence_quote
        or candidate.hierarchy == "unknown"
    )


async def judge_relationship_candidate(
    llm_client: Any,
    candidate: RelationshipCandidate,
    source_text: str,
    *,
    model_name: str = "",
    locked_facts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[RelationshipCandidate, RelationshipJudgeResult]:
    """Ask an optional LLM judge to classify ambiguous evidence.

    The result annotates confidence only. It never changes endpoints, creates
    evidence, or bypasses deterministic validation and lock precedence.
    """

    contract = {
        "decision": "support | reject | uncertain",
        "confidence": 0.0,
        "reason": "brief evidence classification",
    }
    prompt = (
        "Classify whether the source excerpt explicitly supports the proposed "
        "relationship. Do not infer from genre stereotypes or names. Return only "
        "one JSON object matching this contract:\n"
        f"{json.dumps(contract)}\n\n"
        f"Candidate: {json.dumps(candidate.to_dict(), ensure_ascii=False)}\n"
        f"Locked facts: {json.dumps(locked_facts or [], ensure_ascii=False)}\n"
        f"Source excerpt: {str(source_text or '')[:4000]}"
    )
    system_prompt = (
        "You are a conservative relationship evidence classifier. Locked facts "
        "are immutable. Use 'uncertain' whenever both named participants and the "
        "relationship direction are not explicit in the source."
    )
    if hasattr(llm_client, "generate_async"):
        response = llm_client.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=0.0,
        )
    else:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=0.0,
        )
    if inspect.isawaitable(response):
        response = await response
    raw = str(getattr(response, "content", "") or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        result = RelationshipJudgeResult("uncertain", 0.0, "Judge returned invalid JSON.", "invalid_json")
        return candidate, result
    decision = normalize_relationship_name(payload.get("decision"))
    if decision not in {"support", "reject", "uncertain"}:
        decision = "uncertain"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = clean_relationship_text(payload.get("reason"), 300)
    result = RelationshipJudgeResult(decision, confidence, reason, "json")
    judged = replace(
        candidate,
        confidence=max(candidate.confidence, confidence) if decision == "support" else candidate.confidence,
        judge_decision=decision,
        judge_reason=reason,
    )
    return judged, result
