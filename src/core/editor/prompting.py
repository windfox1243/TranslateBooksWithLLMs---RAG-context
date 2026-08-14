"""Prompt composition for the Senior Editor reflection pass.

The reflection pass sends the editor either a complete audit prompt or, when
resuming a job whose earlier local patches did not land, a focused prompt that
carries only the unresolved issues. Both shapes are built here so the caller
deals in one composed result.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.editor.unit_repair import unit_mode_enabled
from src.prompts.prompts import (
    generate_chunk_reflection_prompt,
    generate_unit_reflection_prompt,
)
from src.utils.translation_quality import build_editor_segments


def _candidate_segments_for_issue(
    segments: List[Dict[str, Any]],
    by_id: Dict[str, int],
    issue: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Collect the draft neighborhoods an issue could plausibly be talking about."""

    replacement = issue.get("draft_replacement") or {}
    needles = [
        str(replacement.get("draft") or "").strip(),
        str(issue.get("draft_quote") or "").strip(),
    ]
    candidate_indexes = set()
    requested = str(issue.get("segment_id") or "").upper()
    if requested in by_id:
        candidate_indexes.add(by_id[requested])
    for index, segment in enumerate(segments):
        folded = str(segment.get("text") or "").casefold()
        if any(needle and needle.casefold() in folded for needle in needles):
            candidate_indexes.add(index)
    if not candidate_indexes:
        terms = {
            token.casefold()
            for needle in needles
            for token in re.findall(r"\w{3,}", needle, re.UNICODE)
        }
        scored = []
        for index, segment in enumerate(segments):
            folded = str(segment.get("text") or "").casefold()
            score = sum(1 for term in terms if term in folded)
            if score:
                scored.append((score, index))
        candidate_indexes.update(
            index for _score, index in sorted(scored, reverse=True)[:3]
        )
    expanded = set()
    for index in candidate_indexes:
        expanded.update(
            candidate for candidate in (index - 1, index, index + 1)
            if 0 <= candidate < len(segments)
        )
    return [segments[index] for index in sorted(expanded)]


def _build_focused_locator_retry_prompt(
    draft_text: str,
    issues: List[Dict[str, Any]],
    invalid_ids: set[str],
    locator_errors: List[str],
) -> str:
    """Build a compact locator-only request from candidate draft neighborhoods."""

    from src.utils.translation_quality import build_editor_segments

    segments = build_editor_segments(draft_text)
    by_id = {str(item.get("segment_id") or "").upper(): index for index, item in enumerate(segments)}
    payload = []
    for issue in issues:
        issue_id = str(issue.get("issue_id") or "")
        if issue_id not in invalid_ids:
            continue
        payload.append({
            "issue": issue,
            "candidate_segments": _candidate_segments_for_issue(
                segments, by_id, issue
            ),
        })
    return (
        "Correct only the invalid exact-span locators below. Return the same "
        "reflection JSON schema with status needs_repair, only the corrected "
        "issues, and voice_observations as an empty list. Preserve issue IDs "
        "and repair instructions. Each segment_id must name one candidate "
        "segment; draft_quote must occur exactly once inside it and contain "
        "draft_replacement.draft. If no candidate supports an issue, change "
        "that issue to review_only with no draft_replacement.\n\n"
        "LOCATOR ERRORS:\n"
        + json.dumps(locator_errors, ensure_ascii=False, separators=(",", ":"))
        + "\n\nINVALID ISSUES AND CANDIDATE SEGMENTS:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _build_missing_replacement_retry_prompt(
    draft_text: str,
    issues: List[Dict[str, Any]],
    invalid_ids: set[str],
) -> str:
    """Ask for the corrected wording an issue left out, not for its locator.

    An issue that quotes its span correctly but supplies no replacement was
    being sent through the locator retry, which opens by asking the editor to
    correct locators that were never wrong -- so it answered with the same
    issues unchanged and every one of them was demoted to review. One measured
    chunk raised eight major findings this way and repaired none of them.
    """

    from src.utils.translation_quality import build_editor_segments

    segments = build_editor_segments(draft_text)
    by_id = {
        str(item.get("segment_id") or "").upper(): index
        for index, item in enumerate(segments)
    }
    payload = []
    for issue in issues:
        if str(issue.get("issue_id") or "") not in invalid_ids:
            continue
        payload.append({
            "issue": issue,
            "candidate_segments": _candidate_segments_for_issue(
                segments, by_id, issue
            ),
        })
    return (
        "Each issue below names a defect but supplies no replacement text. "
        "Their locators are not in question: keep segment_id and draft_quote "
        "exactly as given. Return the same reflection JSON schema with status "
        "needs_repair, only these issues, and voice_observations as an empty "
        "list. For each issue set draft_replacement.draft to the exact "
        "substring of draft_quote that is wrong, and "
        "draft_replacement.replacement to the corrected wording. The "
        "replacement must be non-empty and must be a rewrite: never an empty "
        "string, and never a deletion of the quoted text. If the only repair "
        "you can name is removing text, or you cannot write a corrected "
        "wording, change that issue to review_only with no draft_replacement.\n"
        "\nISSUES MISSING A REPLACEMENT:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _render_reflection_novel_context(
    novel_context: str,
    prompt_options: Optional[Dict[str, Any]],
    source_chunk: str,
    draft_translation: str,
) -> str:
    """Render the same selective novel-context view for reflection and repair."""
    options = dict(prompt_options or {})
    if novel_context and not options.get("novel_context"):
        options["novel_context"] = novel_context

    raw_context = str(options.get("novel_context") or novel_context or "")
    active_speaker = None
    attribution = options.get("dialogue_attribution") or {}
    if isinstance(attribution, dict):
        state_after = attribution.get("state_after") or {}
        if isinstance(state_after, dict):
            active_speaker = state_after.get("speaker")

    active_context = ""
    if raw_context.strip():
        try:
            from src.utils.novel_context import render_novel_context_for_prompt

            rendered = render_novel_context_for_prompt(
                raw_context,
                reference_text="\n".join(
                    part for part in (source_chunk, draft_translation) if part
                ),
                max_tokens=options.get("novel_context_prompt_max_tokens"),
                selective=options.get("novel_context_selective_injection", True),
                active_speaker=active_speaker,
            )
            active_context = rendered.strip() or raw_context.strip()
        except Exception:
            active_context = raw_context.strip()
    directed_context = str(options.get("directed_addressing_context") or "").strip()
    relationship_context = str(options.get("relationship_context") or "").strip()
    prompt_context_bundle = str(options.get("prompt_context_bundle") or "").strip()
    context_contract_version = int(
        options.get("context_contract_version", 1) or 1
    )
    blocks = []
    if prompt_context_bundle:
        blocks.append(prompt_context_bundle)
    elif directed_context:
        blocks.append(
            "# STRUCTURED DIRECTED ADDRESSING RULES\n"
            f"{directed_context}"
        )
    if relationship_context and not prompt_context_bundle:
        blocks.append(
            "# STRUCTURED RELATIONSHIP CONTEXT\n"
            f"{relationship_context}"
        )
    if active_context and not (
        context_contract_version >= 5 and prompt_context_bundle
    ):
        blocks.append(
            "# ACTIVE MARKDOWN NOVEL CONTEXT\n"
            f"{active_context}"
        )
    neighbor_context = str(options.get("editor_neighbor_context") or "").strip()
    if neighbor_context:
        blocks.append(
            "# ADJACENT WINDOW CONTEXT (READ-ONLY; DO NOT REWRITE)\n"
            f"{neighbor_context}"
        )
    return "\n\n".join(blocks).strip()


@dataclass
class ReflectionPrompts:
    """The prompt pair to send, plus what went into it."""

    reflection_pair: Any
    reflection_fallback_pair: Any
    active_novel_context: str
    retry_seed_issues: List[Dict[str, Any]]
    use_focused_manual_retry: bool
    components: Dict[str, Any] = field(default_factory=dict)


def _source_blocks_complete(source_chunk: str, prompt: str) -> bool:
    """Report whether every source paragraph reached the prompt."""

    from src.core.editor.units import split_blocks

    blocks = split_blocks(source_chunk)
    return all(block.text in prompt for block in blocks) if blocks else True


def compose_reflection_prompts(
    source_chunk: str,
    draft_translation: str,
    target_language: str,
    novel_context: str,
    custom_instructions: str,
    glossary_block: str,
    deterministic_findings: str,
    narrative_voice_context: str,
    source_available: bool,
    options: Dict[str, Any],
    prompt_options: Optional[Dict[str, Any]] = None,
) -> ReflectionPrompts:
    """Build the reflection prompt pair and the metrics describing it."""
    active_novel_context = _render_reflection_novel_context(
        novel_context=novel_context,
        prompt_options=prompt_options,
        source_chunk=source_chunk,
        draft_translation=draft_translation,
    )

    retry_seed_issues = []
    for seed_index, item in enumerate(
        list(options.get("editor_retry_unresolved_issues") or [])[:12], start=1
    ):
        if (
            not isinstance(item, dict)
            or str(item.get("repair_kind") or "").casefold() != "local_replace"
        ):
            continue
        seed = dict(item)
        seed.setdefault("issue_id", f"retry-{seed_index}")
        retry_seed_issues.append(seed)
    use_focused_manual_retry = bool(
        retry_seed_issues
        and not deterministic_findings
    )
    use_unit_mode = unit_mode_enabled(options) and not use_focused_manual_retry
    if use_focused_manual_retry:
        from src.prompts.prompts import PromptPair

        retry_ids = {str(item.get("issue_id")) for item in retry_seed_issues}
        retry_reasons = [
            str(item) for item in list(options.get("editor_retry_reason_codes") or [])
            if item
        ] or [f"local_patch_unresolved:{item}" for item in sorted(retry_ids)]
        focused_prompt = _build_focused_locator_retry_prompt(
            draft_translation,
            retry_seed_issues,
            retry_ids,
            retry_reasons,
        )
        focused_system = (
            "You are retrying previously unresolved local translation edits. "
            "Use only the supplied issue evidence and candidate draft segments. "
            "Return one canonical reflection JSON object; never rewrite the "
            "complete chunk or introduce unrelated edits."
        )
        reflection_pair = PromptPair(focused_system, focused_prompt)
        reflection_fallback_pair = reflection_pair
    elif use_unit_mode:
        # The focused retry above exists to re-locate a span the editor quoted
        # wrongly, so it has nothing to retry here: a unit id either names a
        # unit or does not, and the answer is the same the second time.
        reflection_pair = generate_unit_reflection_prompt(
            source_chunk=source_chunk,
            draft_translation=draft_translation,
            target_language=target_language,
            novel_context=active_novel_context,
            custom_instructions=custom_instructions,
            glossary_block=glossary_block,
            deterministic_findings=deterministic_findings,
            narrative_voice_context=narrative_voice_context,
            source_available=source_available,
            native_schema=True,
        )
        reflection_fallback_pair = generate_unit_reflection_prompt(
            source_chunk=source_chunk,
            draft_translation=draft_translation,
            target_language=target_language,
            novel_context=active_novel_context,
            custom_instructions=custom_instructions,
            glossary_block=glossary_block,
            deterministic_findings=deterministic_findings,
            narrative_voice_context=narrative_voice_context,
            source_available=source_available,
            native_schema=False,
        )
    else:
        reflection_pair = generate_chunk_reflection_prompt(
            source_chunk=source_chunk,
            draft_translation=draft_translation,
            target_language=target_language,
            novel_context=active_novel_context,
            custom_instructions=custom_instructions,
            glossary_block=glossary_block,
            deterministic_findings=deterministic_findings,
            narrative_voice_context=narrative_voice_context,
            source_available=source_available,
            native_schema=True,
        )
        reflection_fallback_pair = generate_chunk_reflection_prompt(
            source_chunk=source_chunk,
            draft_translation=draft_translation,
            target_language=target_language,
            novel_context=active_novel_context,
            custom_instructions=custom_instructions,
            glossary_block=glossary_block,
            deterministic_findings=deterministic_findings,
            narrative_voice_context=narrative_voice_context,
            source_available=source_available,
            native_schema=False,
        )
    components = {
        "source_chars": len(source_chunk),
        "draft_chars": len(draft_translation),
        "context_chars": len(active_novel_context),
        "glossary_chars": len(glossary_block),
        "custom_instruction_chars": len(custom_instructions),
        "deterministic_finding_chars": len(deterministic_findings),
        "narrator_context_chars": len(narrative_voice_context),
        "fixed_system_chars": len(reflection_pair.system),
        "source_sha256": hashlib.sha256(source_chunk.encode("utf-8")).hexdigest(),
        "draft_sha256": hashlib.sha256(draft_translation.encode("utf-8")).hexdigest(),
        # The bitext carries the source a paragraph at a time, so the whole
        # chunk is never one substring of the prompt. Asking whether every
        # paragraph arrived is the same question, and the only one this metric
        # was ever answering.
        "source_complete": (
            _source_blocks_complete(source_chunk, reflection_pair.user)
            if use_unit_mode
            else source_chunk.strip() in reflection_pair.user
        ),
        "input_mode": (
            "focused_manual_retry" if use_focused_manual_retry
            else "unit_bitext" if use_unit_mode
            else "complete_audit"
        ),
        "retry_source_run_id": options.get("editor_retry_source_run_id") or 0,
        "draft_segment_chars": sum(
            len(str(item.get("text") or ""))
            for item in build_editor_segments(draft_translation)
        ),
    }

    return ReflectionPrompts(
        reflection_pair=reflection_pair,
        reflection_fallback_pair=reflection_fallback_pair,
        active_novel_context=active_novel_context,
        retry_seed_issues=retry_seed_issues,
        use_focused_manual_retry=use_focused_manual_retry,
        components=components,
    )
