"""
Deterministic merge policy engine for directed addressing updates.
"""

from dataclasses import replace
from typing import Optional, Dict, Any, List, Callable, Iterable
import re
import unicodedata
from src.utils.context_schema import AddressingUpdateDelta
from src.persistence.database import Database
from src.utils.progress_logging import emit_progress_log
import src.config as config

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
VI_NEUTRAL_SELF_FORMS = {"tôi"}
_SOURCE_KINSHIP_TERMS = {
    "brother", "sister", "older brother", "older sister", "younger brother",
    "younger sister", "teacher", "senior", "junior", "master", "father",
    "mother", "mom", "dad",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def _evidence_key(value: Any) -> str:
    """Normalize typographic variants without erasing dialogue boundaries."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "„": '"',
                "‟": '"',
                "‘": "'",
                "’": "'",
                "‚": "'",
                "‛": "'",
            }
        )
    ).replace("…", "...")
    return re.sub(r"\s+", " ", text.casefold()).strip()


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


def _source_form_support_reason(
    source_form: str,
    source_text: str,
    evidence: str,
    source_language: str = "",
) -> str:
    """Return an empty string when evidence is supported, otherwise a reason code."""

    from src.utils.text_matching import reference_mentions_label

    if not reference_mentions_label(source_form, source_text, source_language):
        return "source_form_missing"
    if evidence and _evidence_key(evidence) not in _evidence_key(source_text):
        return "evidence_quote_mismatch"
    return ""


def _source_form_supported(
    source_form: str,
    source_text: str,
    evidence: str,
    source_language: str = "",
) -> bool:
    return not _source_form_support_reason(
        source_form,
        source_text,
        evidence,
        source_language,
    )


def _untranslated_generic_vocative(delta: AddressingUpdateDelta, target_language: str) -> bool:
    try:
        from src.utils.language_profiles import get_language_profile

        profile = get_language_profile(target_language)
    except Exception:
        profile = None
    if not profile or profile.addressing_family != "vietnamese":
        return False
    vocative = _norm(delta.vocative)
    source_forms = {
        _norm(item.get("text"))
        for item in delta.source_forms
        if isinstance(item, dict)
    }
    return bool(
        vocative
        and vocative in source_forms
        and vocative in _SOURCE_KINSHIP_TERMS
    )


def _vi_pair_direction(self_pronoun: str, target_pronoun: str) -> str:
    self_key = _norm(self_pronoun)
    target_key = _norm(target_pronoun)
    if self_key in VI_JUNIOR_FORMS and target_key in VI_SENIOR_FORMS:
        return "junior_to_senior"
    if self_key in VI_SENIOR_FORMS and target_key in VI_JUNIOR_FORMS:
        return "senior_to_junior"
    if self_key in VI_PEER_FORMS and target_key in VI_PEER_FORMS:
        return "peer"
    if self_key in VI_NEUTRAL_SELF_FORMS and target_key in VI_PEER_FORMS:
        return "peer"
    if self_key and self_key == target_key:
        return "same"
    return ""


def _ground_social_basis(
    social_basis: Iterable[str],
    source_text: str,
    graph_hierarchy: str,
) -> List[str]:
    """Keep social claims independent and backed by their own evidence."""

    canonical_graph_basis = {
        "source_senior": "speaker is senior to addressee",
        "source_junior": "addressee is senior to speaker",
        "peer": "peer",
    }.get(str(graph_hierarchy or "unknown"), "")
    if canonical_graph_basis:
        return [canonical_graph_basis]

    source_key = _evidence_key(source_text)
    grounded = []
    for item in social_basis or []:
        value = str(item or "").strip()
        if value and _evidence_key(value) in source_key:
            grounded.append(value)
    return list(dict.fromkeys(grounded))


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
        source_text: str = "",
        source_language: str = "",
    ) -> bool:
        """
        Evaluate and merge a single AddressingUpdateDelta into persistent database state.

        Returns True if state was updated, False if rejected by policy rules.
        """
        if not delta:
            return False

        all_existing_rules = self.db.get_addressing_rules(
            translation_id, validation_status=None,
        )
        existing_rules = [
            rule for rule in all_existing_rules
            if rule.get("validation_status") == "active"
        ]
        existing_rule_map = {
            (r["speaker_name"].lower(), r["addressee_name"].lower()): r
            for r in existing_rules
        }

        pair_key = (delta.speaker.lower(), delta.addressee.lower())
        existing = existing_rule_map.get(pair_key)
        reverse = existing_rule_map.get((delta.addressee.lower(), delta.speaker.lower()))
        existing_any = next((
            rule for rule in all_existing_rules
            if (
                str(rule.get("speaker_name") or "").lower(),
                str(rule.get("addressee_name") or "").lower(),
            ) == pair_key
        ), None)

        def persist_evidence() -> None:
            for source_form in delta.source_forms:
                if not isinstance(source_form, dict):
                    continue
                db_form = str(source_form.get("text") or "").strip()
                if not db_form:
                    continue
                self.db.add_addressing_evidence(
                    translation_id,
                    delta.speaker,
                    delta.addressee,
                    db_form,
                    usage=str(source_form.get("usage") or "direct_address"),
                    source_language=str(
                        source_form.get("source_language") or source_language or ""
                    ),
                    evidence_quote=str(
                        source_form.get("evidence_quote")
                        or delta.evidence_quote
                        or ""
                    ),
                    scope=str(source_form.get("scope") or delta.scope),
                    confidence=float(
                        source_form.get("confidence") or delta.confidence
                    ),
                    provenance=delta.provenance or trigger_source,
                    dialogue_turn_id=delta.dialogue_turn_id,
                    chunk_index=chunk_index,
                )

        def reject(reason: str) -> bool:
            msg = (
                f"Rejected addressing update for {delta.speaker} -> {delta.addressee} "
                f"[chunk {chunk_index + 1}]: {reason}"
            )
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
            emit_progress_log(
                log_callback,
                "addressing_rejected",
                msg,
                level="warning",
                layer="db_addressing",
                chunk_index=chunk_index,
                data={
                    "speaker": delta.speaker,
                    "addressee": delta.addressee,
                    "reason": reason,
                    "confidence": delta.confidence,
                    "trigger_source": trigger_source,
                },
            )
            return False

        def provisional(reason: str) -> bool:
            """Persist evidence without projecting an unsupported rule."""

            if existing and existing.get("validation_status") == "active":
                return reject(
                    f"provisional observation did not replace the active rule: {reason}"
                )
            if existing_any and existing_any.get("is_locked"):
                return reject("locked by user.")
            if existing_any and existing_any.get("validation_status") == "quarantined":
                return reject(
                    f"provisional observation did not replace a quarantined rule: {reason}"
                )
            success = self.db.upsert_addressing_rule(
                translation_id=translation_id,
                speaker_name=delta.speaker,
                addressee_name=delta.addressee,
                self_pronoun=delta.self_pronoun,
                target_pronoun=delta.second_pronoun,
                vocative=delta.vocative,
                register=delta.register,
                social_basis=list(delta.social_basis or []),
                scope=delta.scope,
                contract_version=delta.contract_version,
                confidence=min(float(delta.confidence), 0.79),
                is_locked=0,
                chunk_index=chunk_index,
                validation_status="provisional",
                validation_reason=reason,
                provenance=delta.provenance or trigger_source,
            )
            if not success:
                return False
            persist_evidence()
            self.db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=chunk_index,
                speaker_name=delta.speaker,
                addressee_name=delta.addressee,
                old_state=existing_any,
                new_state={
                    "status": "provisional",
                    "reason": reason,
                    "speaker_name": delta.speaker,
                    "addressee_name": delta.addressee,
                    "self_pronoun": delta.self_pronoun,
                    "target_pronoun": delta.second_pronoun,
                    "vocative": delta.vocative,
                    "register": delta.register,
                    "confidence": min(float(delta.confidence), 0.79),
                },
                trigger_source=f"{trigger_source}:provisional",
                evidence_quote=delta.evidence_quote,
                confidence=min(float(delta.confidence), 0.79),
            )
            emit_progress_log(
                log_callback,
                "addressing_provisional",
                (
                    f"Addressing update awaiting corroboration for "
                    f"{delta.speaker} -> {delta.addressee}: {reason}"
                ),
                level="warning",
                layer="db_addressing",
                chunk_index=chunk_index,
                data={
                    "speaker": delta.speaker,
                    "addressee": delta.addressee,
                    "reason": reason,
                    "trigger_source": trigger_source,
                },
            )
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

        if delta.contract_version >= 4 and not trusted_pair_source:
            delta = replace(
                delta,
                social_basis=_ground_social_basis(
                    delta.social_basis,
                    source_text,
                    str(graph_support.get("hierarchy") or "unknown"),
                ),
            )

        v2_source_supported = False
        if delta.contract_version >= 2 and not trusted_pair_source:
            source_forms = [
                item for item in delta.source_forms
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            if not source_forms:
                return reject(
                    "context contract v2 requires at least one exact source form."
                )
            support_failures = [
                (
                    str(item.get("text") or "").strip(),
                    _source_form_support_reason(
                        str(item.get("text") or ""),
                        source_text,
                        str(item.get("evidence_quote") or delta.evidence_quote or ""),
                        source_language,
                    ),
                )
                for item in source_forms
            ]
            missing_forms = [text for text, reason in support_failures if reason == "source_form_missing"]
            mismatched_quotes = [text for text, reason in support_failures if reason == "evidence_quote_mismatch"]
            if missing_forms:
                return reject(
                    "source form is not present in the source chunk "
                    "[source_form_missing]: " + ", ".join(missing_forms[:3])
                )
            if mismatched_quotes:
                return reject(
                    "source evidence quote is not present in the source chunk "
                    "[evidence_quote_mismatch]: " + ", ".join(mismatched_quotes[:3])
                )
            target_evidence = [
                item for item in source_forms
                if _norm(item.get("usage")) in {"direct_address", "second_person"}
                and _norm(item.get("text"))
                and _source_form_supported(
                    str(item.get("text") or ""),
                    str(item.get("evidence_quote") or ""),
                    "",
                    source_language,
                )
            ]
            if not target_evidence:
                return reject(
                    "directed addressing requires an exact spoken direct-address "
                    "or second-person source form in its evidence quote."
                )
            v2_source_supported = True
            strong_evidence = True

        if (
            delta.contract_version >= 2
            and _untranslated_generic_vocative(delta, target_language)
        ):
            return reject(
                "target vocative preserves an untranslated generic source-language "
                "kinship/title term."
            )

        if (
            requires_paired_forms
            and delta.contract_version >= 4
            and not trusted_pair_source
        ):
            incoming_direction = _vi_pair_direction(
                delta.self_pronoun,
                delta.second_pronoun,
            )
            if not incoming_direction:
                return provisional(
                    "unsupported_vietnamese_target_pair"
                )
            if (
                incoming_direction == "peer"
                and delta.scope == "durable"
                and graph_support.get("hierarchy", "unknown") == "unknown"
            ):
                supporting_units = {
                    int(item.get("chunk_index", -1))
                    for item in self.db.get_addressing_evidence(
                        translation_id,
                        delta.speaker,
                        delta.addressee,
                    )
                    if int(item.get("chunk_index", -1)) >= 0
                }
                supporting_units.add(int(chunk_index))
                if len(supporting_units) < 2:
                    return provisional(
                        "peer_pair_awaiting_corroboration"
                    )

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
                explicit_scene_override = bool(
                    graph_support.get("derived")
                    and v2_source_supported
                    and delta.scope == "situational"
                )
                if not explicit_scene_override:
                    return reject(
                        "paired address hierarchy contradicts the accepted relationship graph "
                        f"({graph_hierarchy})."
                    )
                delta = replace(
                    delta,
                    social_basis=list(dict.fromkeys([
                        *delta.social_basis,
                        "explicit scene-level source address overrides derived default",
                    ])),
                )

        # Rule 1: User Lock Override
        if (
            existing
            and existing.get("is_locked")
            and trigger_source not in {"manual", "rest_api", "user_manual"}
        ):
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
                and _norm(existing.get("vocative")) == _norm(delta.vocative)
                and old_register == new_register
                and list(existing.get("social_basis") or [])
                == list(delta.social_basis or [])
                and _norm(existing.get("scope")) == _norm(delta.scope)
            ):
                for source_form in delta.source_forms:
                    if isinstance(source_form, dict):
                        db_form = str(source_form.get("text") or "").strip()
                        if db_form:
                            self.db.add_addressing_evidence(
                                translation_id,
                                delta.speaker,
                                delta.addressee,
                                db_form,
                                usage=str(source_form.get("usage") or "direct_address"),
                                source_language=str(
                                    source_form.get("source_language")
                                    or source_language
                                    or ""
                                ),
                                evidence_quote=str(
                                    source_form.get("evidence_quote")
                                    or delta.evidence_quote
                                    or ""
                                ),
                                scope=str(source_form.get("scope") or delta.scope),
                                confidence=float(
                                    source_form.get("confidence")
                                    or delta.confidence
                                ),
                                provenance=delta.provenance or trigger_source,
                                dialogue_turn_id=delta.dialogue_turn_id,
                                chunk_index=chunk_index,
                            )
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
            "source_forms": list(delta.source_forms or []),
            "social_basis": list(delta.social_basis or []),
            "scope": delta.scope,
            "contract_version": delta.contract_version,
        }

        success = self.db.upsert_addressing_rule(
            translation_id=translation_id,
            speaker_name=delta.speaker,
            addressee_name=delta.addressee,
            self_pronoun=delta.self_pronoun,
            target_pronoun=delta.second_pronoun,
            vocative=delta.vocative,
            register=delta.register,
            social_basis=list(delta.social_basis or []),
            scope=delta.scope,
            contract_version=delta.contract_version,
            confidence=delta.confidence,
            is_locked=0 if not existing else existing.get("is_locked", 0),
            chunk_index=chunk_index,
            validation_status="active",
            validation_reason="",
            provenance=delta.provenance or trigger_source,
        )

        if success:
            persist_evidence()
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
            emit_progress_log(
                log_callback,
                "addressing_merged",
                merged_msg,
                layer="db_addressing",
                chunk_index=chunk_index,
                data={
                    "speaker": delta.speaker,
                    "addressee": delta.addressee,
                    "confidence": delta.confidence,
                    "trigger_source": trigger_source,
                    "source_form_count": len(delta.source_forms),
                },
            )
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
