"""Stateful session objects wrapping context load, update and save."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .characters import _character_gender_map, _character_profile_map
from .constants import logger
from .document import (
    _dynamic_state_has_entries,
    compress_dynamic_state,
    extract_dynamic_state_from_text,
    extract_global_lore,
)
from .gating import set_bypass_context_gating
from .glossary import (
    _character_alias_map,
    _is_inverted_target_to_source_glossary_pair,
    _is_valid_glossary_term,
    _normalized_character_alias_map,
)
from .lore_merge import merge_new_lore
from .merge import (
    _compose_novel_context_from_parts,
    _sanitize_vietnamese_dynamic_state,
    build_novel_context,
    merge_dynamic_state,
)
from .refinement import (
    decode_context_snapshot,
    make_novel_context_filename,
    normalize_refinement_context,
)
from .rendering import (
    _bounded_source_memory,
    _clean_source_memory_chunk,
    _source_memory_budget_chars,
)
from .storage import load_novel_context, resolve_novel_context_path
from .vietnamese import _is_vietnamese_target_language


def _package_update_chunk(*args: Any, **kwargs: Any):
    """Call update_novel_context_chunk as the package currently exposes it.

    novel_context used to be a single module, so a session resolved this from
    the module globals on every call -- which is what lets a pipeline test
    intercept the work by patching src.utils.novel_context.update_novel_context_chunk
    without reaching the session the pipeline builds several layers down.
    Importing the name directly here would bind it once and silently defeat
    that, so the lookup stays late.

    It is the default of an injectable field rather than a lookup written into
    each call site: anyone holding the session can hand it a different updater
    outright, and the one place that still needs late binding says why.
    """
    from src.utils import novel_context

    return novel_context.update_novel_context_chunk(*args, **kwargs)


@dataclass
class RefinementContextTracker:
    """Resolve historical or source-first context for sequential refinement."""

    prompt_options: Dict[str, Any]
    historical_contexts: List[Optional[str]]
    historical_dialogue_attributions: List[Optional[Dict[str, Any]]] = field(
        default_factory=list
    )
    log_callback: Optional[Callable] = None
    # A default_factory rather than a plain default: a function stored as a
    # class attribute would be handed self as its first argument.
    update_chunk: Callable = field(default_factory=lambda: _package_update_chunk)
    cursor: int = 0
    # Derived in __post_init__ from prompt_options, but declared here so that
    # repr, equality and type checking see the whole object. Assigning them in
    # __post_init__ alone left half the tracker's state invisible.
    global_lore: str = ""
    dynamic_state: str = ""
    auto_analyze: bool = False
    dialogue_state: Dict[str, str] = field(default_factory=dict)
    dialogue_scene_key: Optional[str] = None
    current_dialogue_attribution: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from src.utils.dialogue_attribution import empty_dialogue_attribution

        base_context = self.prompt_options.get("novel_context", "")
        self.global_lore = extract_global_lore(base_context)
        self.dynamic_state = extract_dynamic_state_from_text(base_context) or ""
        if any(self.historical_contexts):
            # Refinement replays historical states from the beginning. Never
            # seed that replay with the final end-of-book dynamic state.
            self.dynamic_state = ""
        self.auto_analyze = bool(self.prompt_options.get("auto_update_context"))
        self.current_dialogue_attribution = empty_dialogue_attribution()

    async def next_context(
        self,
        *,
        text: str,
        llm_client: Any,
        model_name: str,
        target_language: str,
        display_index: int,
        total_chunks: int,
        scene_key: Optional[Any] = None,
    ) -> str:
        """Return context for the next refinement unit without mutating its file."""
        historical = (
            self.historical_contexts[self.cursor]
            if self.cursor < len(self.historical_contexts)
            else None
        )
        historical_dialogue = (
            self.historical_dialogue_attributions[self.cursor]
            if self.cursor < len(self.historical_dialogue_attributions)
            else None
        )
        from src.utils.dialogue_attribution import (
            canonicalize_dialogue_attribution,
            detect_dialogue_turns,
            dialogue_attribution_or_carry,
            dialogue_attribution_stats,
            empty_dialogue_attribution,
        )
        normalized_scene_key = (
            str(scene_key) if scene_key is not None else None
        )
        if (
            normalized_scene_key is not None
            and self.dialogue_scene_key is not None
            and normalized_scene_key != self.dialogue_scene_key
        ):
            self.dialogue_state = {}
        if normalized_scene_key is not None:
            self.dialogue_scene_key = normalized_scene_key
        current_aliases = _character_alias_map(self.global_lore)
        self.current_dialogue_attribution = (
            canonicalize_dialogue_attribution(
                historical_dialogue,
                current_aliases,
            )
            if historical_dialogue
            else empty_dialogue_attribution()
        )
        if historical_dialogue:
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after") or {}
            )

        if historical:
            historical_context = normalize_refinement_context(
                historical,
                build_novel_context(self.global_lore, self.dynamic_state),
            )
            self.global_lore = extract_global_lore(historical_context)
            refreshed_aliases = _character_alias_map(self.global_lore)
            self.current_dialogue_attribution = (
                canonicalize_dialogue_attribution(
                    self.current_dialogue_attribution,
                    refreshed_aliases,
                )
            )
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after") or {}
            )
            historical_dynamic = (
                extract_dynamic_state_from_text(historical_context) or ""
            )
            self.dynamic_state = merge_dynamic_state(
                self.dynamic_state,
                historical_dynamic,
                _character_alias_map(self.global_lore),
                target_language=target_language,
                character_genders=_character_gender_map(self.global_lore),
                character_profiles=_character_profile_map(self.global_lore),
            )
            full_context = build_novel_context(
                self.global_lore,
                self.dynamic_state,
            )
            if self.log_callback:
                self.log_callback(
                    "refinement_context_snapshot",
                    f"📚 Restored historical context for refinement unit {display_index}/{total_chunks}.",
                )
        elif self.auto_analyze and text.strip():
            if self.log_callback:
                self.log_callback(
                    "refinement_context_analyzing",
                    f"🧭 Analyzing context for refinement unit {display_index}/{total_chunks}...",
                )
            dialogue_sink: Dict[str, Any] = {}
            dialogue_turns = detect_dialogue_turns(text)
            self.global_lore, self.dynamic_state, change_logs = await self.update_chunk(
                llm_client=llm_client,
                model_name=model_name,
                current_global_lore=self.global_lore,
                current_dynamic_state=self.dynamic_state,
                source_chunk=text,
                translated_chunk=None,
                source_language=target_language,
                target_language=target_language,
                chunk_index=display_index,
                total_chunks=total_chunks,
                dialogue_turns=dialogue_turns,
                current_dialogue_state=self.dialogue_state,
                dialogue_attribution_sink=dialogue_sink,
                selective_context_view=self.prompt_options.get(
                    "novel_context_selective_update",
                    True,
                ),
                context_view_max_tokens=self.prompt_options.get(
                    "novel_context_update_prompt_max_tokens",
                ),
                custom_instructions=str(self.prompt_options.get("custom_instructions") or ""),
                glossary_block=str(self.prompt_options.get("glossary_block") or ""),
                log_callback=self.log_callback,
            )
            self.current_dialogue_attribution = dialogue_attribution_or_carry(
                dialogue_sink, self.dialogue_state
            )
            self.dialogue_state = dict(
                self.current_dialogue_attribution.get("state_after") or {}
            )
            full_context = build_novel_context(
                self.global_lore,
                self.dynamic_state,
            )
            if self.log_callback:
                self.log_callback(
                    "refinement_context_ready",
                    f"✅ Context prepared for refinement unit {display_index}/{total_chunks}.",
                )
                for change_log in change_logs:
                    self.log_callback("novel_context_log", change_log)
                if dialogue_turns:
                    stats = dialogue_attribution_stats(
                        self.current_dialogue_attribution
                    )
                    message = (
                        "Dialogue context: "
                        f"{stats['identified']} turns identified, "
                        f"{stats['assigned']} assigned, "
                        f"{stats['uncertain']} uncertain."
                    )
                    self.log_callback(
                        "dialogue_attribution",
                        message,
                    )
        else:
            full_context = build_novel_context(
                self.global_lore,
                self.dynamic_state,
            )

        self.cursor += 1
        if self.log_callback and (self.global_lore or self.dynamic_state):
            self.log_callback(
                "novel_context_state",
                f"Refinement context ready for unit {display_index}/{total_chunks}",
                {
                    "type": "novel_context_state",
                    "content": full_context,
                    "filename": self.prompt_options.get("novel_context_file", ""),
                    "phase": "refinement",
                    "chunk_index": display_index - 1,
                    "ephemeral": not bool(historical),
                },
            )
        return full_context
@dataclass
class NovelContextSession:
    """Own the mutable context state shared by translation pipelines."""

    path: Path
    prompt_options: Dict[str, Any]
    global_lore: str
    dynamic_state: str
    log_callback: Optional[Callable] = None
    # See RefinementContextTracker.update_chunk for why this is a factory.
    update_chunk: Callable = field(default_factory=lambda: _package_update_chunk)
    dialogue_state: Dict[str, str] = field(default_factory=dict)
    dialogue_attribution: Dict[str, Any] = field(default_factory=dict)
    dialogue_scene_key: Optional[str] = None
    source_memory: List[str] = field(default_factory=list)
    relationship_candidates: List[Dict[str, Any]] = field(default_factory=list)
    relationship_parse_status: str = "absent"
    addressing_candidates: List[Dict[str, Any]] = field(default_factory=list)
    addressing_parse_status: str = "absent"
    # What the context file held the last time this session read or wrote it.
    # Anything else on disk at save time was put there by another job, and has
    # to be merged in rather than replaced. See reconcile.py.
    disk_baseline: str = ""

    @property
    def content(self) -> str:
        return build_novel_context(self.global_lore, self.dynamic_state)

    def sync_prompt(self) -> str:
        content = self.content
        self.prompt_options["novel_context"] = content
        return content

    def save(self) -> str:
        from .reconcile import save_novel_context_merged

        content = self.sync_prompt()
        written, change_logs = save_novel_context_merged(
            self.path.name,
            self.path.parent,
            content,
            self.disk_baseline,
            self._target_language(),
        )
        self.disk_baseline = written
        if change_logs:
            # The session adopts what it just wrote, so the prompt carries the
            # other job's findings from the next chunk onward instead of
            # re-proposing state the file already moved past.
            self.global_lore = extract_global_lore(written)
            self.dynamic_state = extract_dynamic_state_from_text(written) or ""
            self.sync_prompt()
            if self.log_callback:
                self.log_callback(
                    "novel_context_reconciled",
                    "🔀 Context file changed while this job ran; merged "
                    f"{len(change_logs)} update(s) from it.",
                )
        return written

    def _target_language(self) -> str:
        return str(
            self.prompt_options.get("target_language")
            or self.prompt_options.get("target_lang")
            or self.prompt_options.get("language")
            or ""
        )

    def snapshot(self) -> str:
        """Return a compressed full-context snapshot."""
        return compress_dynamic_state(self.content)

    def analysis_result(self):
        """Expose the latest source analysis through the v5 typed contract."""
        from src.core.context import ContextAnalysisResult

        return ContextAnalysisResult(
            dialogue_attribution=dict(self.dialogue_attribution or {}),
            relationship_candidates=list(self.relationship_candidates),
            addressing_observations=list(self.addressing_candidates),
            diagnostics={
                "relationship_parse_status": self.relationship_parse_status,
                "addressing_parse_status": self.addressing_parse_status,
            },
        )

    def register_editor_terms(self, term_pairs: List[Tuple[str, str]]) -> List[str]:
        """Register term replacements ordered by Senior Editor critique into global lore & glossary."""
        if not term_pairs:
            return []
        valid_pairs = []
        for src, tgt in term_pairs:
            if src and tgt and _is_valid_glossary_term(src) and _is_valid_glossary_term(tgt):
                if _is_inverted_target_to_source_glossary_pair(src, tgt):
                    src, tgt = tgt, src
                valid_pairs.append((src, tgt))
        if not valid_pairs:
            return []
        bullet_lines = [f"- {src}: {tgt}" for src, tgt in valid_pairs]
        new_glossary = "\n".join(bullet_lines)
        self.global_lore, change_logs = merge_new_lore(
            global_lore=self.global_lore,
            new_characters="",
            new_glossary=new_glossary,
            source_text="",
        )
        if "glossary_terms" not in self.prompt_options or not isinstance(self.prompt_options.get("glossary_terms"), dict):
            self.prompt_options["glossary_terms"] = {}
        for src, tgt in valid_pairs:
            self.prompt_options["glossary_terms"][src] = tgt

        self.save()
        if self.log_callback and change_logs:
            for log_msg in change_logs:
                self.log_callback("novel_context_term_registered", f"📌 Senior Editor term registered: {log_msg}")
        return change_logs

    def remember_source(self, source_chunk: str) -> None:
        """Retain source text for later context updates without calling the LLM."""
        clean_source = _clean_source_memory_chunk(source_chunk)
        if not clean_source:
            return
        self.source_memory.append(clean_source)
        self.source_memory = [
            chunk for chunk in self.source_memory
            if chunk.strip()
        ]
        bounded = _bounded_source_memory(self.source_memory)
        self.source_memory = (
            bounded.split("\n\n--- Previous source chunk ---\n\n")
            if bounded
            else []
        )

    async def analyze_source(
        self,
        llm_client: Any,
        model_name: str,
        source_chunk: str,
        source_language: str,
        target_language: str,
        chunk_index: int,
        total_chunks: int,
        scene_key: Optional[Any] = None,
    ) -> List[str]:
        """Analyze source text before translating it and expose the new context."""
        from src.utils.dialogue_attribution import (
            detect_dialogue_turns,
            dialogue_attribution_or_carry,
            dialogue_attribution_stats,
        )

        normalized_scene_key = (
            str(scene_key) if scene_key is not None else None
        )
        if (
            normalized_scene_key is not None
            and self.dialogue_scene_key is not None
            and normalized_scene_key != self.dialogue_scene_key
        ):
            self.dialogue_state = {}
        if normalized_scene_key is not None:
            if (
                self.dialogue_scene_key is not None
                and normalized_scene_key != self.dialogue_scene_key
            ):
                self.source_memory = []
            self.dialogue_scene_key = normalized_scene_key

        source_context = _bounded_source_memory(self.source_memory)
        if self.log_callback:
            budget = _source_memory_budget_chars()
            self.log_callback(
                "novel_context_source_memory",
                "Source-analysis memory prepared for the current scene.",
                {
                    "used_chars": len(source_context),
                    "budget_chars": budget,
                    "truncated": bool(budget and len(source_context) >= budget),
                    "scope": "source_analysis",
                },
            )
        dialogue_turns = detect_dialogue_turns(source_chunk)
        dialogue_sink: Dict[str, Any] = {}
        relationship_sink: Dict[str, Any] = {}
        addressing_sink: Dict[str, Any] = {}
        self.global_lore, self.dynamic_state, change_logs = await self.update_chunk(
            llm_client=llm_client,
            model_name=model_name,
            current_global_lore=self.global_lore,
            current_dynamic_state=self.dynamic_state,
            source_chunk=source_chunk,
            translated_chunk=None,
            source_language=source_language,
            target_language=target_language,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            source_context=source_context,
            dialogue_turns=dialogue_turns,
            current_dialogue_state=self.dialogue_state,
            dialogue_attribution_sink=dialogue_sink,
            relationship_candidate_sink=relationship_sink,
            addressing_candidate_sink=addressing_sink,
            context_contract_version=int(
                self.prompt_options.get("context_contract_version", 1)
            ),
            selective_context_view=self.prompt_options.get(
                "novel_context_selective_update",
                True,
            ),
            context_view_max_tokens=self.prompt_options.get(
                "novel_context_update_prompt_max_tokens",
            ),
            custom_instructions=str(self.prompt_options.get("custom_instructions") or ""),
            glossary_block=str(self.prompt_options.get("glossary_block") or ""),
            log_callback=self.log_callback,
        )
        self.relationship_candidates = list(
            relationship_sink.get("candidates") or []
        )
        self.relationship_parse_status = str(
            relationship_sink.get("parse_status") or "absent"
        )
        self.addressing_candidates = list(
            addressing_sink.get("candidates") or []
        )
        self.addressing_parse_status = str(
            addressing_sink.get("parse_status") or "absent"
        )
        self.remember_source(source_chunk)
        self.dialogue_attribution = dialogue_attribution_or_carry(
            dialogue_sink, self.dialogue_state
        )
        self.dialogue_state = dict(
            self.dialogue_attribution.get("state_after") or {}
        )
        if normalized_scene_key is not None:
            self.dialogue_attribution["scene_key"] = normalized_scene_key
        if self.dialogue_attribution.get("turns"):
            self.prompt_options["dialogue_attribution"] = self.dialogue_attribution
        else:
            self.prompt_options.pop("dialogue_attribution", None)
        if dialogue_turns:
            # Same line either way; the branch is only about where it goes.
            stats = dialogue_attribution_stats(self.dialogue_attribution)
            message = (
                "Dialogue context: "
                f"{stats['identified']} turns identified, "
                f"{stats['assigned']} assigned, "
                f"{stats['uncertain']} uncertain."
            )
            if self.log_callback:
                self.log_callback("dialogue_attribution", message)
            else:
                logger.info(message)
        self.save()
        return change_logs

    async def sync_translated_output(
        self,
        translated_chunk: str,
        source_chunk: str = "",
        target_language: Optional[str] = None,
        source_language: Optional[str] = None,
        llm_client: Optional[Any] = None,
        model_name: str = "",
        chunk_index: int = 0,
        total_chunks: int = 0,
    ) -> bool:
        """Sync final polished translation output (from Senior Editor pass) back into dynamic context."""
        if not translated_chunk or not translated_chunk.strip():
            return False

        target_lang = target_language or self.prompt_options.get("target_language") or "Vietnamese"
        source_lang = source_language or self.prompt_options.get("source_language") or "English"

        contract_version = int(
            self.prompt_options.get("context_contract_version", 1) or 1
        )

        # Contract v5 treats source analysis as authoritative for lore and
        # relationships.  The final translation is only allowed to contribute
        # deterministic target-language observations after structural success;
        # it must not trigger a second full lore/relationship analysis request.
        if contract_version >= 5:
            alias_map = _character_alias_map(self.global_lore)
            sanitized_dynamic = _sanitize_vietnamese_dynamic_state(
                dynamic_state=self.dynamic_state,
                alias_map=alias_map,
                character_genders=_character_gender_map(self.global_lore),
                character_profiles=_character_profile_map(self.global_lore),
                translated_chunk=translated_chunk,
                dialogue_attribution=self.dialogue_attribution,
                target_language=target_lang,
            )
            if sanitized_dynamic and sanitized_dynamic.strip() != self.dynamic_state.strip():
                self.dynamic_state = sanitized_dynamic
                self.save()
                return True
            return False

        if llm_client and source_chunk:
            from src.utils.dialogue_attribution import detect_dialogue_turns
            dialogue_turns = detect_dialogue_turns(source_chunk)
            dialogue_sink: Dict[str, Any] = {}
            relationship_sink: Dict[str, Any] = {}
            addressing_sink: Dict[str, Any] = {}
            self.global_lore, self.dynamic_state, change_logs = await self.update_chunk(
                llm_client=llm_client,
                model_name=model_name,
                current_global_lore=self.global_lore,
                current_dynamic_state=self.dynamic_state,
                source_chunk=source_chunk,
                translated_chunk=translated_chunk,
                source_language=source_lang,
                target_language=target_lang,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                dialogue_turns=dialogue_turns,
                current_dialogue_state=self.dialogue_state,
                dialogue_attribution_sink=dialogue_sink,
                relationship_candidate_sink=relationship_sink,
                addressing_candidate_sink=addressing_sink,
                context_contract_version=int(
                    self.prompt_options.get("context_contract_version", 1)
                ),
                selective_context_view=self.prompt_options.get("novel_context_selective_update", True),
                context_view_max_tokens=self.prompt_options.get("novel_context_update_prompt_max_tokens"),
                custom_instructions=str(self.prompt_options.get("custom_instructions") or ""),
                glossary_block=str(self.prompt_options.get("glossary_block") or ""),
                log_callback=self.log_callback,
            )
            self.relationship_candidates = list(
                relationship_sink.get("candidates") or []
            )
            self.relationship_parse_status = str(
                relationship_sink.get("parse_status") or "absent"
            )
            self.addressing_candidates = list(
                addressing_sink.get("candidates") or []
            )
            self.addressing_parse_status = str(
                addressing_sink.get("parse_status") or "absent"
            )
            from src.utils.dialogue_attribution import dialogue_attribution_or_carry
            self.dialogue_attribution = dialogue_attribution_or_carry(
                dialogue_sink, self.dialogue_state
            )
            self.dialogue_state = dict(
                self.dialogue_attribution.get("state_after") or {}
            )
            self.save()
            return True
        else:
            alias_map = _character_alias_map(self.global_lore)
            sanitized_dynamic = _sanitize_vietnamese_dynamic_state(
                dynamic_state=self.dynamic_state,
                alias_map=alias_map,
                character_genders=_character_gender_map(self.global_lore),
                character_profiles=_character_profile_map(self.global_lore),
                translated_chunk=translated_chunk,
                dialogue_attribution=self.dialogue_attribution,
                target_language=target_lang,
            )
            if sanitized_dynamic and sanitized_dynamic.strip() != self.dynamic_state.strip():
                self.dynamic_state = sanitized_dynamic
                self.save()
                return True
        return False
def open_novel_context_session(
    prompt_options: Dict[str, Any],
    novel_contexts_dir: Path,
    input_filename: str = "",
    fallback_name: str = "translation",
    resume_snapshot: Optional[str] = None,
    resume_dialogue_state: Optional[Dict[str, str]] = None,
    resume_dialogue_scene_key: Optional[Any] = None,
    log_callback: Optional[Callable] = None,
) -> Optional[NovelContextSession]:
    """Load/create context state, restore a snapshot, and inject it into prompts."""
    # Scoped to this job's flow rather than assigned onto src.config, which is
    # shared by every concurrent job in the process. See gating.py.
    if "bypass_context_gating" in prompt_options:
        set_bypass_context_gating(bool(prompt_options["bypass_context_gating"]))

    novel_context_file = prompt_options.get("novel_context_file")
    auto_update_context = bool(prompt_options.get("auto_update_context", False))

    if auto_update_context and not novel_context_file:
        novel_context_file = make_novel_context_filename(input_filename, fallback_name)
        prompt_options["novel_context_file"] = novel_context_file
        if log_callback:
            log_callback(
                "novel_context_created",
                f"Auto-created new novel context file: {novel_context_file}",
            )

    if not novel_context_file:
        return None

    path = resolve_novel_context_path(novel_context_file, novel_contexts_dir)
    current_content = load_novel_context(path.name, path.parent)
    disk_baseline = current_content
    file_global_lore = extract_global_lore(current_content)
    file_dynamic_state = extract_dynamic_state_from_text(current_content) or ""
    global_lore = file_global_lore
    dynamic_state = file_dynamic_state

    if resume_snapshot:
        snapshot_content, snapshot_global_lore, snapshot_dynamic_state = (
            decode_context_snapshot(
                resume_snapshot,
                current_content,
                canonicalize_full_snapshot=False,
            )
        )
        prefer_resume_snapshot = bool(
            prompt_options.get("prefer_resume_snapshot")
        )
        if prefer_resume_snapshot or not _dynamic_state_has_entries(
            file_dynamic_state
        ):
            current_content = snapshot_content
            global_lore = snapshot_global_lore or file_global_lore
            dynamic_state = snapshot_dynamic_state
        elif log_callback and _dynamic_state_has_entries(snapshot_dynamic_state):
            log_callback(
                "novel_context_file_preferred",
                "Using the current context file instead of an older resume snapshot.",
            )

    character_aliases = (
        _normalized_character_alias_map(global_lore)
        if resume_snapshot
        else _character_alias_map(global_lore)
    )
    target_language = (
        prompt_options.get("target_language")
        or prompt_options.get("target_lang")
        or prompt_options.get("language")
    )
    if _is_vietnamese_target_language(target_language):
        sanitized_dynamic_state = _sanitize_vietnamese_dynamic_state(
            dynamic_state,
            character_aliases,
            character_genders=_character_gender_map(global_lore),
            character_profiles=_character_profile_map(global_lore),
            target_language=target_language,
        )
        if sanitized_dynamic_state != str(dynamic_state or "").strip():
            dynamic_state = sanitized_dynamic_state
            current_content = _compose_novel_context_from_parts(
                global_lore,
                dynamic_state,
            )

    from src.utils.dialogue_attribution import canonicalize_dialogue_state
    resume_dialogue_state = canonicalize_dialogue_state(
        resume_dialogue_state,
        character_aliases,
    )

    session = NovelContextSession(
        path=path,
        prompt_options=prompt_options,
        global_lore=global_lore,
        dynamic_state=dynamic_state,
        log_callback=log_callback,
        dialogue_state=dict(resume_dialogue_state or {}),
        disk_baseline=disk_baseline,
        dialogue_scene_key=(
            str(resume_dialogue_scene_key)
            if resume_dialogue_scene_key is not None
            else None
        ),
    )
    if resume_snapshot:
        content = current_content
        session.prompt_options["novel_context"] = content
    else:
        content = session.sync_prompt()
    if log_callback:
        log_callback(
            "novel_context_state",
            "Context loaded",
            {
                "type": "novel_context_state",
                "content": content,
                "filename": path.name,
            },
        )
    return session
