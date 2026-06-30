"""Scene-local dialogue speaker attribution helpers.

Dialogue attribution is intentionally separate from persistent novel lore.
The model may infer who speaks a turn, but an inference must never create or
modify a canonical character entry. The structured map produced here is stored
with a translation checkpoint and injected into translation/refinement prompts
as internal metadata only.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional


MIN_SPEAKER_CONFIDENCE = 0.65
MAX_DIALOGUE_CANDIDATES = 80
MAX_CUE_LENGTH = 240

_PLACEHOLDER_RE = re.compile(
    r"(?:\[id\d+\]|\[\[\d+\]\]|/\d+|\$\d+\$|__TEMP_[A-Z0-9_]+__)",
    re.IGNORECASE,
)
_SUBTITLE_MARKER_RE = re.compile(r"^\s*\[(\d+)\]\s*")
_DIALOGUE_DASH_RE = re.compile(r"^\s*[—–-]\s+\S")
_QUOTED_SPAN_RE = re.compile(
    r"""
    “[^”\n]{1,800}”
    |‘[^’\n]{1,800}’
    |「[^」\n]{1,800}」
    |『[^』\n]{1,800}』
    |《[^》\n]{1,800}》
    |«[^»\n]{1,800}»
    |‹[^›\n]{1,800}›
    |"[^"\n]{1,800}"
    """,
    re.VERBOSE,
)
_UNKNOWN_VALUES = {
    "",
    "unknown",
    "unspecified",
    "none",
    "null",
    "n/a",
    "na",
    "?",
}


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = _PLACEHOLDER_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def _stable_turn_id(
    cue: str,
    previous_line: str,
    next_line: str,
    occurrence: int,
) -> str:
    material = "\n".join(
        (
            _normalize(previous_line)[-100:],
            _normalize(cue),
            _normalize(next_line)[:100],
            str(occurrence),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:14]
    return f"dlg-{digest}"


def detect_dialogue_turns(source_text: str) -> List[Dict[str, str]]:
    """Return conservative, stable dialogue candidates from source text.

    Quoted spans, dialogue-dash lines, and indexed subtitle lines are supported.
    The LLM still decides whether each candidate is actual dialogue. Pure prose
    is not sent for attribution, which keeps prompts and logs compact.
    """
    lines = str(source_text or "").splitlines()
    candidates: List[Dict[str, str]] = []
    occurrences: Dict[str, int] = {}
    represented_lines = set()

    def append_candidate(
        cue: str,
        line_index: int,
        marker: str = "",
    ) -> None:
        signature = _normalize(cue).casefold()
        occurrence = occurrences.get(signature, 0)
        occurrences[signature] = occurrence + 1
        candidates.append(
            {
                "id": _stable_turn_id(
                    cue,
                    lines[line_index - 1] if line_index > 0 else "",
                    lines[line_index + 1]
                    if line_index + 1 < len(lines)
                    else "",
                    occurrence,
                ),
                "cue": cue[:MAX_CUE_LENGTH],
                "line": marker or str(line_index + 1),
            }
        )
        represented_lines.add(line_index)

    for line_index, raw_line in enumerate(lines):
        clean_line = _normalize(raw_line)
        if not clean_line:
            continue

        subtitle_match = _SUBTITLE_MARKER_RE.match(clean_line)
        quoted_spans = [
            _normalize(match.group(0))
            for match in _QUOTED_SPAN_RE.finditer(clean_line)
        ]

        if subtitle_match:
            cue_entries = [clean_line]
            marker = subtitle_match.group(1)
        elif quoted_spans:
            cue_entries = quoted_spans
            marker = ""
        elif _DIALOGUE_DASH_RE.match(clean_line):
            cue_entries = [clean_line]
            marker = ""
        else:
            continue

        for cue in cue_entries:
            append_candidate(cue, line_index, marker)
            if len(candidates) >= MAX_DIALOGUE_CANDIDATES:
                return candidates

    # Some books and scripts format conversation as short alternating lines
    # without quotation marks or speaker labels. Treat compact runs as
    # candidates only when the run contains conversational punctuation. The
    # model may still classify any candidate as non-dialogue/Unknown.
    line_index = 0
    while line_index < len(lines):
        if not _normalize(lines[line_index]):
            line_index += 1
            continue
        run_start = line_index
        while line_index < len(lines) and _normalize(lines[line_index]):
            line_index += 1
        run_indices = list(range(run_start, line_index))
        clean_run = [_normalize(lines[index]) for index in run_indices]
        compact = (
            2 <= len(clean_run) <= 8
            and all(1 <= len(line) <= 180 for line in clean_run)
            and any(re.search(r"[?!？！…]$", line) for line in clean_run)
            and not any(index in represented_lines for index in run_indices)
            and not any(
                re.match(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)", line)
                for line in clean_run
            )
        )
        if compact:
            for candidate_index, cue in zip(run_indices, clean_run):
                if candidate_index in represented_lines:
                    continue
                append_candidate(cue, candidate_index)
                if len(candidates) >= MAX_DIALOGUE_CANDIDATES:
                    return candidates

    return candidates


def dialogue_candidates_prompt(turns: Iterable[Dict[str, str]]) -> str:
    """Serialize candidates for the context-analysis prompt."""
    payload = [
        {
            "id": str(turn.get("id") or ""),
            "cue": str(turn.get("cue") or ""),
            "line": str(turn.get("line") or ""),
        }
        for turn in turns
        if turn.get("id") and turn.get("cue")
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _canonical_character(
    value: Any,
    character_aliases: Dict[str, str],
) -> str:
    normalized = _normalize(str(value or ""))
    if normalized.casefold() in _UNKNOWN_VALUES:
        return "Unknown"

    key = normalized.casefold().strip(" .,:;[]()")
    if key in character_aliases:
        return character_aliases[key]

    for alias, canonical in character_aliases.items():
        if key == alias.casefold():
            return canonical
    return "Unknown"


def canonicalize_dialogue_state(
    state: Optional[Dict[str, Any]],
    character_aliases: Dict[str, str],
) -> Dict[str, str]:
    """Resolve persisted scene state through the latest canonical registry."""
    result: Dict[str, str] = {}
    for field in ("speaker", "addressee"):
        canonical = _canonical_character(
            (state or {}).get(field),
            character_aliases,
        )
        if canonical != "Unknown":
            result[field] = canonical
    return result


def canonicalize_dialogue_attribution(
    attribution: Optional[Dict[str, Any]],
    character_aliases: Dict[str, str],
) -> Dict[str, Any]:
    """Migrate a saved dialogue map to current canonical character names."""
    source = attribution or {}
    result: Dict[str, Any] = {
        "version": source.get("version", 1),
        "turns": [],
        "state_after": {},
    }
    for raw_turn in source.get("turns") or []:
        if not isinstance(raw_turn, dict):
            continue
        turn = dict(raw_turn)
        turn["speaker"] = _canonical_character(
            raw_turn.get("speaker"),
            character_aliases,
        )
        turn["addressee"] = _canonical_character(
            raw_turn.get("addressee"),
            character_aliases,
        )
        turn["confidence"] = _confidence(raw_turn.get("confidence"))
        result["turns"].append(turn)
    result["state_after"] = _state_after_from_confident_turns(result["turns"])
    if source.get("scene_key") is not None:
        result["scene_key"] = source.get("scene_key")
    return result


def _confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return max(0.0, min(numeric, 1.0))


def _state_after_from_confident_turns(
    turns: Iterable[Dict[str, Any]],
) -> Dict[str, str]:
    """Persist only source-current, high-confidence dialogue state."""
    state: Dict[str, str] = {}
    for turn in turns:
        speaker = str(turn.get("speaker") or "")
        addressee = str(turn.get("addressee") or "")
        if (
            speaker.casefold() in _UNKNOWN_VALUES
            or speaker == "Unknown"
            or _confidence(turn.get("confidence")) < MIN_SPEAKER_CONFIDENCE
        ):
            continue
        state = {"speaker": speaker}
        if addressee.casefold() not in _UNKNOWN_VALUES and addressee != "Unknown":
            state["addressee"] = addressee
    return state


def parse_dialogue_attribution(
    raw_block: str,
    candidates: List[Dict[str, str]],
    character_aliases: Dict[str, str],
    previous_state: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate model attribution against candidate IDs and canonical lore."""
    del previous_state
    candidate_by_id = {
        str(candidate.get("id")): candidate
        for candidate in candidates
        if candidate.get("id")
    }
    result: Dict[str, Any] = {
        "version": 1,
        "turns": [],
        "state_after": {},
    }
    if not raw_block.strip() or not candidate_by_id:
        return result

    text = raw_block.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return result

    if isinstance(payload, list):
        payload = {"turns": payload}
    if not isinstance(payload, dict):
        return result

    seen = set()
    for item in payload.get("turns") or []:
        if not isinstance(item, dict):
            continue
        turn_id = str(item.get("id") or "")
        if turn_id not in candidate_by_id or turn_id in seen:
            continue
        seen.add(turn_id)
        candidate = candidate_by_id[turn_id]
        result["turns"].append(
            {
                "id": turn_id,
                "cue": candidate.get("cue", ""),
                "line": candidate.get("line", ""),
                "speaker": _canonical_character(
                    item.get("speaker"),
                    character_aliases,
                ),
                "addressee": _canonical_character(
                    item.get("addressee"),
                    character_aliases,
                ),
                "confidence": _confidence(item.get("confidence")),
            }
        )

    result["state_after"] = _state_after_from_confident_turns(result["turns"])
    return result


def dialogue_attribution_stats(
    attribution: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    turns = list((attribution or {}).get("turns") or [])
    assigned = sum(
        1
        for turn in turns
        if (
            turn.get("speaker") not in _UNKNOWN_VALUES | {"Unknown"}
            and _confidence(turn.get("confidence")) >= MIN_SPEAKER_CONFIDENCE
        )
    )
    return {
        "identified": len(turns),
        "assigned": assigned,
        "uncertain": max(0, len(turns) - assigned),
    }


def format_dialogue_attribution_for_prompt(
    attribution: Optional[Dict[str, Any]],
) -> str:
    """Render only confident assignments as hidden prompt metadata."""
    confident = []
    for turn in (attribution or {}).get("turns") or []:
        speaker = str(turn.get("speaker") or "")
        confidence = _confidence(turn.get("confidence"))
        if speaker.casefold() in _UNKNOWN_VALUES or confidence < MIN_SPEAKER_CONFIDENCE:
            continue
        confident.append(
            {
                "id": str(turn.get("id") or ""),
                "speaker": speaker,
                "addressee": (
                    str(turn.get("addressee") or "Unknown")
                    if str(turn.get("addressee") or "").casefold()
                    not in _UNKNOWN_VALUES
                    else "Unknown"
                ),
                "cue": str(turn.get("cue") or "")[:MAX_CUE_LENGTH],
            }
        )

    if not confident:
        return ""

    payload = json.dumps(
        confident,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "# SCENE-LOCAL DIALOGUE ATTRIBUTION (INTERNAL METADATA)\n\n"
        "Use this map only to choose pronouns, addressing forms, register, and "
        "character voice. Match entries to dialogue cues in source order. "
        "Never print IDs, speaker labels, addressee labels, confidence values, "
        "or this metadata in the translated/refined text.\n\n"
        f"{payload}"
    )


def empty_dialogue_attribution(
    state_after: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "turns": [],
        "state_after": dict(state_after or {}),
    }
