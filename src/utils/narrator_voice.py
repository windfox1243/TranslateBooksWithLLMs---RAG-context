"""Evidence-backed cross-language narrator voice timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.utils.language_profiles import get_language_profile


VOICE_CONTRACT_VERSION = 2
ACTIVATION_CONFIDENCE = 0.90
MIN_SUPPORTING_SEGMENTS = 2
DISCOURSE_MODES = frozenset({
    "narration", "dialogue", "thought", "letter", "embedded_story",
})
TRANSITION_TYPES = frozenset({"none", "chapter", "scene", "explicit"})
BOOTSTRAP_INITIAL_BOUNDARIES = (2, 4, 8, 12)
BOOTSTRAP_INTERVAL = 4


@dataclass
class NarratorVoiceProfile:
    narrator_key: str = "default"
    narrator_identity: str = "unknown"
    point_of_view: str = "unknown"
    self_reference: str = ""
    formality: str = "neutral"
    speech_level: str = ""
    gender: str = "unknown"
    number: str = "singular"
    dialect: str = ""
    tense: str = ""
    stylistic_markers: List[str] = field(default_factory=list)
    dimensions: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: str = "unknown"
    scope: str = "durable"
    start_chunk_index: int = 0
    end_chunk_index: Optional[int] = None
    is_locked: bool = False
    status: str = "provisional"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceObservation:
    segment_id: str
    discourse_mode: str
    narrator_key: str = "default"
    narrator_identity: str = "unknown"
    point_of_view: str = "unknown"
    source_quote: str = ""
    target_quote: str = ""
    dimensions: Dict[str, str] = field(default_factory=dict)
    transition_type: str = "none"
    transition_evidence: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _exact_span(haystack: str, needle: str) -> bool:
    value = str(needle or "").strip()
    return bool(value and str(haystack or "").count(value) == 1)


def eligible_bootstrap_boundary(completed_count: int) -> Optional[int]:
    """Return the latest retry boundary reached by a completed-unit count."""

    count = max(0, int(completed_count or 0))
    if count < BOOTSTRAP_INITIAL_BOUNDARIES[0]:
        return None
    if count <= BOOTSTRAP_INITIAL_BOUNDARIES[-1]:
        return max(value for value in BOOTSTRAP_INITIAL_BOUNDARIES if value <= count)
    return BOOTSTRAP_INITIAL_BOUNDARIES[-1] + (
        (count - BOOTSTRAP_INITIAL_BOUNDARIES[-1]) // BOOTSTRAP_INTERVAL
    ) * BOOTSTRAP_INTERVAL


def narrator_policy_payload(target_language: str) -> Dict[str, Any]:
    """Return the public, serializable provisional policy for a language."""

    policy = get_language_profile(target_language).narrator_default_policy
    return asdict(policy)


def _provisional_profile(
    target_language: str, *, narrator_key: str = "default",
    narrator_identity: str = "unknown", confidence: float = 0.75,
) -> NarratorVoiceProfile:
    policy = narrator_policy_payload(target_language)
    dimensions = {
        "point_of_view": "first",
        "self_reference_strategy": str(policy.get("strategy") or "evidence_only"),
    }
    self_reference = str(policy.get("self_reference") or "")
    if self_reference:
        dimensions["self_reference"] = self_reference
    fallback = str(policy.get("fallback_self_reference") or "")
    if fallback:
        dimensions["fallback_self_reference"] = fallback
    return NarratorVoiceProfile(
        narrator_key=narrator_key or "default",
        narrator_identity=narrator_identity or "unknown",
        point_of_view="first",
        self_reference=self_reference,
        dimensions=dimensions,
        confidence=max(0.0, min(0.89, float(confidence or 0.0))),
        provenance="language_default",
        status="provisional",
        start_chunk_index=0,
    )


def normalize_voice_observation(raw: Any) -> Optional[VoiceObservation]:
    """Normalize one untrusted model observation without inferring missing facts."""

    if not isinstance(raw, dict):
        return None
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    discourse_mode = str(raw.get("discourse_mode") or "").strip().casefold()
    if discourse_mode not in DISCOURSE_MODES:
        return None
    transition = str(raw.get("transition_type") or "none").strip().casefold()
    if transition not in TRANSITION_TYPES:
        transition = "none"
    dimensions = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    return VoiceObservation(
        segment_id=str(raw.get("segment_id") or "").strip().upper(),
        discourse_mode=discourse_mode,
        narrator_key=str(raw.get("narrator_key") or "default").strip() or "default",
        narrator_identity=str(raw.get("narrator_identity") or "unknown").strip() or "unknown",
        point_of_view=str(raw.get("point_of_view") or "unknown").strip() or "unknown",
        source_quote=str(raw.get("source_quote") or "").strip(),
        target_quote=str(raw.get("target_quote") or "").strip(),
        dimensions={str(k): str(v) for k, v in dimensions.items() if str(v).strip()},
        transition_type=transition,
        transition_evidence=str(raw.get("transition_evidence") or "").strip(),
        confidence=confidence,
    )


def validate_voice_observations(
    raw_observations: Iterable[Any], *, source_text: str, target_text: str,
    target_language: str,
) -> tuple[List[VoiceObservation], List[Dict[str, Any]]]:
    """Keep only observations grounded in exact source and target spans."""

    allowed = set(get_language_profile(target_language).narrator_voice_dimensions)
    accepted: List[VoiceObservation] = []
    rejected: List[Dict[str, Any]] = []
    for raw in raw_observations or []:
        item = normalize_voice_observation(raw)
        reason = ""
        if item is None:
            reason = "invalid_contract"
        elif not item.segment_id:
            reason = "missing_segment_id"
        elif not _exact_span(source_text, item.source_quote):
            reason = "source_evidence_not_unique"
        elif not _exact_span(target_text, item.target_quote):
            reason = "target_evidence_not_unique"
        elif item.discourse_mode in {"dialogue", "letter"} and item.transition_type == "none":
            # Dialogue and letters are retained as evidence but never establish
            # the surrounding narrator by themselves.
            pass
        if item is not None:
            item.dimensions = {
                key: value for key, value in item.dimensions.items()
                if key in allowed
            }
            if item.transition_type == "explicit" and not item.transition_evidence:
                reason = "missing_transition_evidence"
        if reason:
            rejected.append({"observation": raw, "reason": reason})
        elif item is not None:
            accepted.append(item)
    return accepted, rejected


def profile_from_observations(
    observations: Sequence[VoiceObservation], *, chunk_index: int,
    provenance: str = "senior_editor",
) -> Optional[NarratorVoiceProfile]:
    """Build a durable profile only from two agreeing narrative observations."""

    narrative = [
        item for item in observations
        if item.discourse_mode in {"narration", "thought", "embedded_story"}
        and item.confidence >= ACTIVATION_CONFIDENCE
    ]
    if len({item.segment_id for item in narrative}) < MIN_SUPPORTING_SEGMENTS:
        return None
    first = narrative[0]
    agreeing = [
        item for item in narrative
        if item.narrator_key == first.narrator_key
        and item.point_of_view == first.point_of_view
        and item.dimensions == first.dimensions
    ]
    if len({item.segment_id for item in agreeing}) < MIN_SUPPORTING_SEGMENTS:
        return None
    dimensions = dict(first.dimensions)
    return NarratorVoiceProfile(
        narrator_key=first.narrator_key,
        narrator_identity=first.narrator_identity,
        point_of_view=first.point_of_view,
        self_reference=dimensions.get("self_reference", ""),
        formality=dimensions.get("formality", dimensions.get("politeness", "neutral")),
        speech_level=dimensions.get("speech_level", ""),
        gender=dimensions.get("gender", "unknown"),
        number=dimensions.get("number", "singular"),
        dialect=dimensions.get("regional_register", dimensions.get("dialect", "")),
        tense=dimensions.get("tense", ""),
        stylistic_markers=[dimensions["style"]] if dimensions.get("style") else [],
        dimensions=dimensions,
        confidence=min(item.confidence for item in agreeing),
        provenance=provenance,
        start_chunk_index=int(chunk_index),
        status="active",
    )


def build_narrator_voice_context(
    translation_id: str, db: Any, *, chunk_index: int,
    target_language: str,
) -> str:
    """Render profiles effective at a chunk, with locked profiles first."""

    if not translation_id or db is None or not hasattr(db, "get_narrator_voice_profiles"):
        return ""
    profiles = [
        item for item in db.get_narrator_voice_profiles(
            translation_id, effective_chunk_index=int(chunk_index),
            include_inactive=True,
        )
        if item.get("status") in {"active", "provisional"}
    ]
    if not profiles:
        return ""
    dimensions = ", ".join(get_language_profile(target_language).narrator_voice_dimensions)
    lines = [
        "ESTABLISHED NARRATOR VOICE TIMELINE (not dialogue addressing):",
        f"Applicable {target_language} dimensions: {dimensions}.",
    ]
    for item in sorted(profiles, key=lambda row: (not bool(row.get("is_locked")), row.get("narrator_key", ""))):
        values = {
            "point_of_view": item.get("point_of_view"),
            "self_reference": item.get("self_reference"),
            "formality": item.get("formality"),
            "speech_level": item.get("speech_level"),
            "gender": item.get("gender"),
            "number": item.get("number"),
            "dialect": item.get("dialect"),
            "tense": item.get("tense"),
        }
        values.update(item.get("dimensions") or {})
        rendered = ", ".join(f"{key}={value}" for key, value in values.items() if value and value != "unknown")
        lock = (
            "locked user profile" if item.get("is_locked")
            else "provisional language policy" if item.get("status") == "provisional"
            else "accepted evidence profile"
        )
        lines.append(
            f"- {item.get('narrator_key', 'default')} ({item.get('narrator_identity', 'unknown')}): "
            f"{rendered or 'style continuity only'} [{lock}; confidence={float(item.get('confidence', 0)):.2f}]"
        )
    lines.append(
        "Apply these profiles only to their narrator/discourse scope. Pair-specific addressing never overrides narration. "
        "Do not invent a transition; uncertain change requires review."
    )
    return "\n".join(lines)


def voice_observation_contract() -> str:
    """Compact JSON contract fragment shared by context and Editor prompts."""

    return json.dumps({
        "segment_id": "SEG-0001",
        "discourse_mode": "narration|dialogue|thought|letter|embedded_story",
        "narrator_key": "stable narrator key",
        "narrator_identity": "name or unknown",
        "point_of_view": "first|second|third|mixed|unknown",
        "dimensions": {"language_specific_dimension": "observed value"},
        "source_quote": "exact unique source span",
        "target_quote": "exact unique target span",
        "transition_type": "none|chapter|scene|explicit",
        "transition_evidence": "exact evidence when explicit",
        "confidence": 0.0,
    }, ensure_ascii=False)


def persist_voice_observations(
    db: Any, translation_id: str, chunk_index: int,
    observations: Sequence[VoiceObservation], *, chapter_index: Optional[int] = None,
    scene_key: str = "", provenance: str = "senior_editor",
) -> Dict[str, int]:
    """Persist evidence, activating only stable profiles and safe transitions."""

    stats = {"observations": 0, "profiles": 0, "conflicts": 0, "backfill": 0}
    narrator_keys = {item.narrator_key for item in observations}
    if len(narrator_keys) > 1:
        for narrator_key in sorted(narrator_keys):
            nested = persist_voice_observations(
                db, translation_id, chunk_index,
                [item for item in observations if item.narrator_key == narrator_key],
                chapter_index=chapter_index, scene_key=scene_key,
                provenance=provenance,
            )
            for key, value in nested.items():
                stats[key] += value
        return stats
    for item in observations:
        db.add_narrator_voice_observation(
            translation_id, chunk_index, item.to_dict(),
            chapter_index=chapter_index, scene_key=scene_key,
            provenance=provenance,
        )
        stats["observations"] += 1

    historical: List[VoiceObservation] = []
    earliest_chunk = int(chunk_index)
    if hasattr(db, "get_narrator_voice_timeline"):
        for row in db.get_narrator_voice_timeline(translation_id).get("observations", []):
            if row.get("status") != "accepted":
                continue
            item = normalize_voice_observation(row)
            if item is None or item.narrator_key not in narrator_keys:
                continue
            row_chunk = int(row.get("chunk_index", chunk_index) or 0)
            earliest_chunk = min(earliest_chunk, row_chunk)
            historical.append(replace(
                item, segment_id=f"CHUNK-{row_chunk}:{item.segment_id}"
            ))
    candidate = profile_from_observations(
        observations, chunk_index=chunk_index, provenance=provenance,
    )
    if candidate is None:
        candidate = profile_from_observations(
            historical, chunk_index=earliest_chunk, provenance=provenance,
        )
    if candidate is None:
        return stats
    current = [
        item for item in db.get_narrator_voice_profiles(
            translation_id, effective_chunk_index=max(0, chunk_index - 1),
        )
        if item.get("narrator_key") == candidate.narrator_key
    ]
    if not current:
        if db.upsert_narrator_voice_profile(translation_id, candidate.to_dict()):
            stats["profiles"] += 1
            if hasattr(db, "mark_narrator_voice_chunks_stale") and chunk_index > 0:
                stats["backfill"] += db.mark_narrator_voice_chunks_stale(
                    translation_id, candidate.start_chunk_index,
                    end_chunk_index=chunk_index - 1,
                )
        return stats
    active = current[0]
    comparable = (
        str(active.get("point_of_view") or "unknown"),
        active.get("dimensions") or {},
    )
    proposed = (candidate.point_of_view, candidate.dimensions)
    if comparable == proposed:
        return stats
    transition_items = [item for item in observations if item.transition_type != "none"]
    safe_transition = any(
        item.transition_type in {"chapter", "scene"}
        and ((item.transition_type == "chapter" and chapter_index is not None)
             or (item.transition_type == "scene" and bool(scene_key)))
        or item.transition_type == "explicit" and bool(item.transition_evidence)
        for item in transition_items
    )
    if not safe_transition or active.get("is_locked"):
        db.add_narrator_voice_conflict(
            translation_id,
            narrator_key=candidate.narrator_key,
            chunk_index=chunk_index,
            chapter_index=chapter_index,
            scene_key=scene_key,
            reason=(
                "A locked narrator profile conflicts with new evidence."
                if active.get("is_locked")
                else "Narrator change lacks a supported chapter, scene, or explicit transition."
            ),
            candidate=candidate.to_dict(),
        )
        stats["conflicts"] += 1
        return stats
    # Close the inferred prior interval. User locks are never modified here.
    db._get_connection().execute("""
        UPDATE context_narrator_profiles SET end_chunk_index = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE translation_id = ? AND id = ? AND is_locked = 0
    """, (chunk_index - 1, translation_id, active["id"]))
    db._commit_connection(db._get_connection())
    created = db.upsert_narrator_voice_profile(translation_id, candidate.to_dict())
    if created:
        transition = transition_items[0]
        conn = db._get_connection()
        conn.execute("""
            INSERT INTO context_narrator_transitions (
                translation_id, narrator_key, from_profile_id, to_profile_id,
                chunk_index, chapter_index, scene_key, transition_type,
                evidence_quote, confidence, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (translation_id, candidate.narrator_key, active["id"], created["id"],
              chunk_index, chapter_index, scene_key, transition.transition_type,
              transition.transition_evidence or transition.source_quote,
              transition.confidence, provenance))
        db._commit_connection(conn)
        stats["profiles"] += 1
    return stats


async def _preflight_narrator_voice(
    *, db: Any, translation_id: str, chunks: Sequence[Dict[str, Any]],
    target_language: str, model_name: str, llm_client: Any,
    file_type: str,
) -> Dict[str, Any]:
    """Create a conservative source-grounded provisional language policy."""

    boundary_key = f"v{VOICE_CONTRACT_VERSION}:source"
    candidates = [
        item for item in chunks
        if str(item.get("original_text") or "").strip()
    ][:3]
    if not candidates:
        return {"status": "not_ready", "next_boundary": 2}
    sampled_indices = [int(item.get("chunk_index", 0)) for item in candidates]
    if not db.claim_narrator_bootstrap(
        translation_id, attempt_kind="preflight", boundary_key=boundary_key,
        sampled_chunks=sampled_indices,
    ):
        return {"status": "preflight_deduplicated", "next_boundary": 2}
    payloads = [{
        "chunk_index": int(item.get("chunk_index", 0)),
        "source": str(item.get("original_text") or "")[:8000],
    } for item in candidates]
    prompt = (
        "Classify narrator point of view from source evidence only. Do not infer "
        "from dialogue pronouns. Return JSON with source_narrator containing: "
        "narrative_detected (boolean), point_of_view (first|second|third|mixed|unknown), "
        "narrator_key, narrator_identity, chunk_index, exact unique source_quote, "
        "confidence, and voice_over (boolean). For SRT, narrative_detected may be "
        "true only for an explicit narrator or voice-over cue. File type: "
        + str(file_type or "unknown") + "\n" + json.dumps(payloads, ensure_ascii=False)
    )
    try:
        from src.core.translator import _generate_editor_response
        response = await _generate_editor_response(
            llm_client=llm_client, prompt=prompt,
            system_prompt="You are an evidence-driven narrator analyst. Return JSON only.",
            model_name=model_name, temperature=0.0, max_output_tokens=800,
            response_schema=None, stage="narrator_preflight",
        )
        raw = str(getattr(response, "content", "") or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
        summary = data.get("source_narrator") or {}
        try:
            chunk_index = int(summary.get("chunk_index", -1))
            confidence = float(summary.get("confidence", 0.0))
        except (TypeError, ValueError):
            chunk_index, confidence = -1, 0.0
        sample = next((item for item in payloads if item["chunk_index"] == chunk_index), None)
        quote = str(summary.get("source_quote") or "")
        accepted = bool(
            sample
            and summary.get("narrative_detected")
            and str(summary.get("point_of_view") or "").casefold() == "first"
            and confidence >= 0.80
            and _exact_span(sample["source"], quote)
            and (
                str(file_type or "").casefold() != "srt"
                or bool(summary.get("voice_over"))
            )
        )
        if accepted:
            db.upsert_narrator_voice_profile(
                translation_id,
                _provisional_profile(
                    target_language,
                    narrator_key=str(summary.get("narrator_key") or "default"),
                    narrator_identity=str(summary.get("narrator_identity") or "unknown"),
                    confidence=confidence,
                ).to_dict(),
            )
        status = "provisional" if accepted else "ambiguous"
        db.finish_narrator_bootstrap(
            translation_id, status, attempt_kind="preflight",
            boundary_key=boundary_key,
            details={
                "accepted": int(accepted), "sampled": sampled_indices,
                "contract_version": VOICE_CONTRACT_VERSION,
            },
        )
        return {"status": status, "next_boundary": 2}
    except Exception as exc:
        db.finish_narrator_bootstrap(
            translation_id, "failed", attempt_kind="preflight",
            boundary_key=boundary_key,
            details={"error": type(exc).__name__},
        )
        return {"status": "failed", "error": type(exc).__name__, "next_boundary": 2}


async def bootstrap_narrator_voice(
    *, db: Any, translation_id: str, chunks: Sequence[Dict[str, Any]],
    target_language: str, model_name: str, llm_client: Any,
    file_type: str = "",
) -> Dict[str, Any]:
    """Run retryable source preflight and completed-output voice analysis."""

    if not all(hasattr(db, name) for name in (
        "get_narrator_voice_profiles", "claim_narrator_bootstrap",
        "finish_narrator_bootstrap",
    )):
        return {"status": "unsupported_database"}
    if db.get_narrator_voice_profiles(translation_id):
        return {"status": "not_needed"}
    completed = [
        item for item in chunks
        if item.get("status") in {"completed", "partial"}
        and str(item.get("original_text") or "").strip()
        and str(item.get("translated_text") or "").strip()
    ]
    boundary = eligible_bootstrap_boundary(len(completed))
    if boundary is None:
        return await _preflight_narrator_voice(
            db=db, translation_id=translation_id, chunks=chunks,
            target_language=target_language, model_name=model_name,
            llm_client=llm_client, file_type=file_type,
        )
    boundary_key = f"v{VOICE_CONTRACT_VERSION}:completed:{boundary}"
    eligible = completed[:boundary]
    indices = sorted({
        0, len(eligible) - 1, len(eligible) // 5,
        (2 * len(eligible)) // 5, (3 * len(eligible)) // 5,
        (4 * len(eligible)) // 5,
    })[:6]
    sampled = [eligible[index] for index in indices]
    sampled_indices = [int(item.get("chunk_index", 0)) for item in sampled]
    if not db.claim_narrator_bootstrap(
        translation_id, boundary_key=boundary_key,
        sampled_chunks=sampled_indices,
    ):
        return {"status": "deduplicated", "boundary": boundary}
    payloads = []
    budget = 24000  # conservative four-characters-per-token input estimate
    for item in sampled:
        source = str(item.get("original_text") or "")
        target = str(item.get("translated_text") or "")
        allowance = max(1000, budget // max(1, len(sampled) - len(payloads)))
        source = source[:allowance // 2]
        target = target[:allowance // 2]
        budget -= len(source) + len(target)
        payloads.append({
            "chunk_index": int(item.get("chunk_index", 0)),
            "source": source,
            "target": target,
        })
    prompt = (
        "Analyze only narrator voice in these aligned completed translation chunks. "
        "Do not count pronouns or infer from majority frequency. Distinguish narration, "
        "dialogue, thoughts, letters, and embedded stories. Return one JSON object with "
        "voice_observations and source_narrator. source_narrator must contain "
        "narrative_detected, point_of_view, narrator_key, narrator_identity, "
        "chunk_index, source_quote, and confidence. Every observation must include "
        "chunk_index and this contract: "
        + voice_observation_contract()
        + "\nUse exact unique source_quote and target_quote evidence. Dialogue alone "
        "never establishes a narrator. For SRT, require an explicit narration or "
        "voice-over cue. Return [] when uncertain.\nFile type: "
        + str(file_type or "unknown") + "\n"
        + json.dumps(payloads, ensure_ascii=False)
    )
    try:
        from src.core.translator import _generate_editor_response
        response = await _generate_editor_response(
            llm_client=llm_client, prompt=prompt,
            system_prompt=(
                "You are an evidence-driven literary narrator analyst. Return JSON only."
            ),
            model_name=model_name, temperature=0.0, max_output_tokens=3000,
            response_schema=None, stage="narrator_bootstrap",
        )
        raw = str(getattr(response, "content", "") or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
        raw_items = data.get("voice_observations") or []
        accepted_all: List[VoiceObservation] = []
        rejected_all: List[Dict[str, Any]] = []
        for sample in payloads:
            raw_for_chunk = [
                item for item in raw_items
                if int(item.get("chunk_index", -1)) == sample["chunk_index"]
            ]
            accepted, rejected = validate_voice_observations(
                raw_for_chunk, source_text=sample["source"],
                target_text=sample["target"], target_language=target_language,
            )
            rejected_all.extend(rejected)
            for item in accepted:
                item = replace(
                    item,
                    segment_id=f"CHUNK-{sample['chunk_index']}:{item.segment_id}",
                )
                db.add_narrator_voice_observation(
                    translation_id, sample["chunk_index"], item.to_dict(),
                    provenance="bootstrap",
                )
            accepted_all.extend(accepted)
        profile = profile_from_observations(
            accepted_all, chunk_index=min(sampled_indices), provenance="bootstrap",
        )
        if profile:
            db.upsert_narrator_voice_profile(translation_id, profile.to_dict())
            if hasattr(db, "mark_narrator_voice_chunks_stale"):
                db.mark_narrator_voice_chunks_stale(
                    translation_id, profile.start_chunk_index,
                    end_chunk_index=max(sampled_indices),
                )
        elif not db.get_narrator_voice_profiles(
            translation_id, include_inactive=True,
        ):
            source_summary = data.get("source_narrator") or {}
            try:
                summary_chunk = int(source_summary.get("chunk_index", -1))
                summary_confidence = float(source_summary.get("confidence", 0.0))
            except (TypeError, ValueError):
                summary_chunk, summary_confidence = -1, 0.0
            sample = next((
                item for item in payloads if item["chunk_index"] == summary_chunk
            ), None)
            source_quote = str(source_summary.get("source_quote") or "")
            is_narrative = bool(source_summary.get("narrative_detected"))
            is_first = str(source_summary.get("point_of_view") or "").casefold() == "first"
            srt_allowed = str(file_type or "").casefold() != "srt" or bool(
                source_summary.get("voice_over")
            )
            if (
                sample and is_narrative and is_first and srt_allowed
                and summary_confidence >= 0.80
                and _exact_span(sample["source"], source_quote)
            ):
                provisional = _provisional_profile(
                    target_language,
                    narrator_key=str(source_summary.get("narrator_key") or "default"),
                    narrator_identity=str(
                        source_summary.get("narrator_identity") or "unknown"
                    ),
                    confidence=summary_confidence,
                )
                db.upsert_narrator_voice_profile(
                    translation_id, provisional.to_dict()
                )
        status = "profiled" if profile else "ambiguous"
        db.finish_narrator_bootstrap(
            translation_id, status, boundary_key=boundary_key,
            details={
                "accepted": len(accepted_all),
                "rejected": len(rejected_all),
                "rejection_reasons": [item["reason"] for item in rejected_all],
                "sampled": sampled_indices,
                "boundary": boundary,
                "contract_version": VOICE_CONTRACT_VERSION,
            },
        )
        return {
            "status": status, "accepted": len(accepted_all),
            "rejected": len(rejected_all), "boundary": boundary,
            "next_boundary": boundary + BOOTSTRAP_INTERVAL,
        }
    except Exception as exc:
        db.finish_narrator_bootstrap(
            translation_id, "failed", boundary_key=boundary_key,
            details={"error": type(exc).__name__, "boundary": boundary},
        )
        return {"status": "failed", "error": type(exc).__name__}
