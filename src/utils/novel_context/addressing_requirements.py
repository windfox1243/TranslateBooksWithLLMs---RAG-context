"""Detect and retry missing directed-addressing candidates."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import _plain_key
from .dynamic_state import _DYNAMIC_RELATION_PATTERN, _split_dynamic_sections

_ADDRESSING_CANDIDATE_PROMPT_BLOCK = r'''[ADDRESSING_CANDIDATES]
{"updates":[{"speaker":"canonical character","addressee":"canonical character","source_forms":[{"text":"exact source surface form","usage":"direct_address | second_person | self_reference | indirect_reference","evidence_quote":"exact quote from LATEST SOURCE TEXT"}],"target_form":{"self_reference":"target-language self-reference","second_person":"target-language second-person form","vocative":"target-language vocative or none"},"register":"neutral | formal | polite | casual | intimate | hostile | vulgar | archaic | familial","social_basis":["age | sibling | school-year | rank | teacher/student | intimacy | hostility | other concise basis"],"scope":"durable | situational","evidence_quote":"exact quote from LATEST SOURCE TEXT","dialogue_turn_id":"optional candidate id","confidence":0.0,"action":"upsert | delete"}]}
(Output valid JSON on one line. Include only new or changed directed speaker/addressee rules. Every new LLM rule must include an exact source form and exact source evidence. `source_forms[].text` is the literal source expression used by the speaker, such as `brother`, not the canonical addressee name unless that name was actually spoken. Target fields must be fully translated into the target language. Never preserve generic source-language titles such as `Brother`, `Sister`, `Teacher`, or `Senior` as target vocatives unless a protected glossary explicitly requires it. Source forms are addressing evidence, never character identity aliases. For Vietnamese, include a complete self-reference/second-person pair and use proven age, sibling order, school year, rank, or teacher/student seniority instead of peer forms. Return {"updates":[]} when nothing changed.)'''
def _inject_addressing_candidate_contract(prompt: str) -> str:
    """Insert the v2 addressing block immediately before relationship JSON."""

    marker = "[RELATIONSHIP_CANDIDATES]"
    if _ADDRESSING_CANDIDATE_PROMPT_BLOCK in prompt or marker not in prompt:
        return prompt
    return prompt.replace(
        marker,
        f"{_ADDRESSING_CANDIDATE_PROMPT_BLOCK}\n\n{marker}",
        1,
    )
_SECOND_PERSON_MARKERS = {
    "english": ("you", "your", "yours", "yourself", "yourselves"),
    "french": ("tu", "te", "toi", "ton", "ta", "tes", "vous", "votre", "vos"),
    "spanish": ("tú", "te", "ti", "usted", "ustedes", "vos", "vosotros", "vuestro"),
    "german": ("du", "dich", "dir", "dein", "ihr", "euch", "sie", "ihnen"),
    "italian": ("tu", "te", "ti", "voi", "lei", "loro", "tuo", "vostro"),
    "portuguese": ("tu", "te", "ti", "você", "vocês", "vosso", "seu"),
    "vietnamese": ("bạn", "cậu", "anh", "chị", "em", "ông", "bà", "ngươi", "mày"),
    "chinese": ("你", "您", "你们", "妳"),
    "japanese": ("あなた", "君", "きみ", "お前", "貴方", "あんた"),
    "korean": ("너", "당신", "그대", "자네", "너희"),
}
def _existing_directed_addressing_pairs(dynamic_state: str) -> set[Tuple[str, str]]:
    addressing, _, _ = _split_dynamic_sections(dynamic_state)
    pairs: set[Tuple[str, str]] = set()
    for line in addressing.splitlines():
        match = _DYNAMIC_RELATION_PATTERN.match(line)
        if match and match.group("arrow") == "→":
            pairs.add(
                (
                    _plain_key(match.group("left")),
                    _plain_key(match.group("right")),
                )
            )
    return pairs
def _cue_has_directed_address(
    cue: str,
    addressee: str,
    source_language: str,
) -> bool:
    from src.utils.text_matching import reference_mentions_label

    language_key = _plain_key(source_language)
    markers: Tuple[str, ...] = ()
    for name, values in _SECOND_PERSON_MARKERS.items():
        if name in language_key:
            markers = values
            break
    if any(
        reference_mentions_label(marker, cue, source_language)
        for marker in markers
    ):
        return True

    cue_tokens = set(re.findall(r"[^\W_]+(?:-[^\W_]+)*", cue.casefold(), re.UNICODE))
    name_tokens = re.findall(r"[^\W_]+", addressee.casefold(), re.UNICODE)
    return any(
        len(token) >= 3
        and any(cue_token == token or cue_token.startswith(f"{token}-") for cue_token in cue_tokens)
        for token in name_tokens
    )
def _missing_addressing_requirements(
    dialogue_attribution: Dict[str, Any],
    candidates: List[Any],
    current_dynamic_state: str,
    source_language: str,
) -> List[Dict[str, Any]]:
    """Find high-confidence directed dialogue pairs lacking an addressing candidate."""

    covered = {
        (_plain_key(candidate.speaker), _plain_key(candidate.addressee))
        for candidate in candidates
    }
    covered.update(_existing_directed_addressing_pairs(current_dynamic_state))
    requirements: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for turn in dialogue_attribution.get("turns") or []:
        speaker = str(turn.get("speaker") or "").strip()
        addressee = str(turn.get("addressee") or "").strip()
        cue = str(turn.get("cue") or "").strip()
        try:
            confidence = float(turn.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        pair = (_plain_key(speaker), _plain_key(addressee))
        if (
            confidence < 0.80
            or not all(pair)
            or "unknown" in pair
            or pair in covered
            or pair in seen
            or not _cue_has_directed_address(cue, addressee, source_language)
        ):
            continue
        seen.add(pair)
        requirements.append(
            {
                "dialogue_turn_id": str(turn.get("id") or ""),
                "speaker": speaker,
                "addressee": addressee,
                "evidence_quote": cue,
                "confidence": confidence,
            }
        )
    return requirements
async def _retry_missing_addressing_candidates(
    llm_client: Any,
    requirements: List[Dict[str, Any]],
    source_language: str,
    target_language: str,
) -> Tuple[List[Any], str]:
    from src.utils.addressing_schema import parse_addressing_candidate_block

    prompt = (
        "Create addressing updates only for the missing high-confidence dialogue "
        "pairs below. Use each supplied evidence quote exactly; do not add pairs or "
        "facts. Return only one JSON object with an updates array matching the "
        "ADDRESSING_CANDIDATES contract. Each upsert must include complete target "
        "self_reference and second_person fields.\n\n"
        f"Source language: {source_language}\nTarget language: {target_language}\n"
        f"Missing pairs: {json.dumps(requirements, ensure_ascii=False)}"
    )
    response = await llm_client.generate(
        prompt=prompt,
        system_prompt=(
            "You fill deterministic addressing-coverage gaps from supplied spoken "
            "evidence. Return JSON only and never invent evidence."
        ),
    )
    return parse_addressing_candidate_block(
        str(getattr(response, "content", "") or ""),
        source_language=source_language,
    )
