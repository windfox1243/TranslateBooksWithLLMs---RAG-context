"""Deterministic analysis run before the editor LLM sees the draft.

Everything here is computed without an LLM call: which terms are protected
from rewriting, which source-language fragments leaked into the draft, and
whether the narrator's self-reference conforms to the job's policy. The
findings become the "deterministic findings" block handed to the editor
prompt, so this pass fixes what it can locally and reports the rest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.utils.addressing_schema import context_contract_version
from src.utils.progress_logging import emit_progress_log
from src.utils.translation_quality import (
    find_source_residue,
    identity_preserving_proper_names,
)


@dataclass(frozen=True)
class EditorPreflight:
    """What the deterministic pass established about a draft chunk.

    draft_translation is the draft after exact narrator-form patches were
    applied, so callers must continue from this text rather than the one they
    passed in.
    """

    draft_translation: str
    source_available: bool
    contract_v2: bool
    source_language: str
    glossary_terms: Dict[str, Any]
    protected_terms: List[str]
    residue_findings: List[Any]
    narrator_conformance: Dict[str, Any]
    narrator_patches: List[Any]
    remaining_narrator_blockers: List[Dict[str, Any]]
    initial_narrator_finding_count: int
    deterministic_findings: str
    narrative_voice_context: str


def run_editor_preflight(
    source_chunk: str,
    draft_translation: str,
    target_language: str,
    novel_context: str,
    options: Dict[str, Any],
    log_callback: Optional[Callable] = None,
) -> EditorPreflight:
    """Run the deterministic checks that precede the editor reflection pass."""
    source_available = bool(source_chunk and source_chunk.strip()) and str(
        options.get("editor_source_mode") or "checkpoint"
    ).casefold() != "monolingual"
    contract_v2 = context_contract_version(options) >= 2
    source_language = str(options.get("source_language") or "")
    glossary_terms = (
        options.get("glossary_terms")
        if isinstance(options.get("glossary_terms"), dict)
        else {}
    )
    protected_terms = [
        str(item) for item in options.get("active_character_names") or [] if item
    ]
    protected_terms.extend(
        str(item) for item in options.get("preserved_terms") or [] if item
    )
    protected_terms.extend(
        identity_preserving_proper_names(source_chunk, draft_translation)
    )
    try:
        from src.utils.novel_context import (
            GLOSSARY_SECTION,
            _character_profile_map,
            _find_lore_section,
            _parse_bullet_entries,
            character_alias_map,
            extract_global_lore,
        )

        raw_lore = extract_global_lore(
            str(options.get("novel_context") or novel_context or "")
        )
        protected_terms.extend(
            str(profile.get("name") or key)
            for key, profile in _character_profile_map(raw_lore).items()
        )
        aliases = character_alias_map(raw_lore)
        protected_terms.extend(str(item) for item in aliases.keys())
        protected_terms.extend(str(item) for item in aliases.values())
        glossary_bounds = _find_lore_section(raw_lore, GLOSSARY_SECTION)
        if glossary_bounds:
            for source_term, target_term in _parse_bullet_entries(
                raw_lore[glossary_bounds[1]:glossary_bounds[2]]
            ):
                if source_term and target_term:
                    glossary_terms.setdefault(source_term, target_term)
                    protected_terms.extend((source_term, target_term))
    except Exception as exc:
        # Whatever was collected before the failure is kept, but the rest of the
        # character names, aliases and glossary pairs are now absent from
        # protected_terms -- so the editor is free to rewrite them. That is a
        # silent quality regression unless it is reported.
        emit_progress_log(
            log_callback,
            "editor_protected_terms_incomplete",
            f"Could not derive the full protected-term set from the novel "
            f"context: {type(exc).__name__}: {exc}. Proper names and glossary "
            f"pairs may not be protected for this unit.",
            level="warning",
        )
    residue_findings = []
    if source_available and bool(options.get("source_residue_validation", contract_v2)):
        residue_findings = find_source_residue(
            source_chunk,
            draft_translation,
            source_language=source_language,
            target_language=target_language,
            protected_terms=protected_terms,
            glossary_terms=glossary_terms,
        )
    from src.core.editor import (
        apply_narrator_conformance_patches,
        audit_narrator_conformance,
    )

    narrator_conformance = audit_narrator_conformance(
        source_text=source_chunk,
        target_text=draft_translation,
        source_language=source_language,
        target_language=target_language,
        file_type=str(options.get("file_type") or "txt"),
        dialogue_attribution=options.get("dialogue_attribution") or {},
        db=options.get("_checkpoint_db"),
        translation_id=str(options.get("translation_id") or ""),
        chunk_index=int(options.get("chunk_index", 0) or 0),
        explicit_override=str(
            options.get("narrator_self_reference_override") or ""
        ),
    )
    draft_translation, narrator_patches = apply_narrator_conformance_patches(
        draft_translation, narrator_conformance,
    )
    if narrator_patches:
        if log_callback:
            emit_progress_log(
                log_callback,
                "narrator_conformance_locally_patched",
                f"Applied {len(narrator_patches)} exact narrator-form patch(es).",
                layer="narrator_voice",
            )
        narrator_conformance = audit_narrator_conformance(
            source_text=source_chunk,
            target_text=draft_translation,
            source_language=source_language,
            target_language=target_language,
            file_type=str(options.get("file_type") or "txt"),
            dialogue_attribution=options.get("dialogue_attribution") or {},
            db=options.get("_checkpoint_db"),
            translation_id=str(options.get("translation_id") or ""),
            chunk_index=int(options.get("chunk_index", 0) or 0),
            explicit_override=str(
                options.get("narrator_self_reference_override") or ""
            ),
        )
        if source_available and bool(
            options.get("source_residue_validation", contract_v2)
        ):
            residue_findings = find_source_residue(
                source_chunk,
                draft_translation,
                source_language=source_language,
                target_language=target_language,
                protected_terms=protected_terms,
                glossary_terms=glossary_terms,
            )
    remaining_narrator_blockers = [
        item for item in narrator_conformance.get("violating_segments") or []
        if item.get("blocking")
    ]
    initial_narrator_finding_count = (
        len(narrator_patches) + len(remaining_narrator_blockers)
    )
    deterministic_payloads = [
        finding.to_dict() for finding in residue_findings
    ] + remaining_narrator_blockers
    deterministic_findings = json.dumps(
        deterministic_payloads,
        ensure_ascii=False,
        indent=2,
    ) if deterministic_payloads else ""
    narrative_voice_context = str(
        options.get("narrative_voice_context") or ""
    ).strip()

    return EditorPreflight(
        draft_translation=draft_translation,
        source_available=source_available,
        contract_v2=contract_v2,
        source_language=source_language,
        glossary_terms=glossary_terms,
        protected_terms=protected_terms,
        residue_findings=residue_findings,
        narrator_conformance=narrator_conformance,
        narrator_patches=narrator_patches,
        remaining_narrator_blockers=remaining_narrator_blockers,
        initial_narrator_finding_count=initial_narrator_finding_count,
        deterministic_findings=deterministic_findings,
        narrative_voice_context=narrative_voice_context,
    )
