"""Merge a context document against whatever another job left on disk.

`save_novel_context` makes one write atomic. That is not the same as making a
caller's read, edit and write-back atomic, and the gap between the two is where
a whole job's worth of lore can vanish.

A session loads the context file once when it opens, keeps the state in memory
for the length of the job, and writes the whole document back. Two jobs on one
context file -- normal when translating several volumes of the same novel --
each hold a copy taken at their own start time, so whichever saves last replaces
everything the other one learned. Not one lost edit: every edit.

`save_novel_context_merged` closes that by re-reading under the same lock that
guards the write. When the file still matches the baseline the caller started
from, nothing has changed underneath and the save is a plain write -- which is
the single-job case, and therefore almost every case. When it has changed, the
two documents are reconciled before the write so both jobs' findings survive.

Reconciliation reuses the merge the translation pipeline already uses for LLM
output rather than inventing a second set of rules: the other job's entries
arrive as candidate lore, and `merge_new_lore` / `merge_dynamic_state` decide
how they combine. That has one known edge: `merge_new_lore` takes characters,
aliases and glossary, but not the name-translation map, so name-map entries
written only by the other job are not carried over. Every other section is.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .characters import _character_gender_map, _character_profile_map
from .constants import ALIASES_SECTION, CHARACTERS_SECTION, GLOSSARY_SECTION
from .document import extract_dynamic_state_from_text, extract_global_lore
from .glossary import character_alias_map
from .lore_merge import merge_new_lore
from .merge import (
    build_novel_context,
    merge_dynamic_state,
    normalize_novel_context_content,
)
from .rendering import _section_body
from .storage import _resolve_inside, writer_lock


def reconcile_context_documents(
    mine: str,
    theirs: str,
    target_language: str = "",
) -> Tuple[str, List[str]]:
    """Combine two full context documents, keeping what each one knows.

    `mine` is the base and `theirs` supplies the candidate entries, so a genuine
    disagreement about the same character resolves the way it would if the other
    job's lore had come back from the model on this job's next chunk.
    """
    my_lore = extract_global_lore(mine)
    their_lore = extract_global_lore(theirs)

    merged_lore, change_logs = merge_new_lore(
        my_lore,
        _section_body(their_lore, CHARACTERS_SECTION),
        _section_body(their_lore, GLOSSARY_SECTION),
        _section_body(their_lore, ALIASES_SECTION),
        "",
    )
    merged_dynamic = merge_dynamic_state(
        extract_dynamic_state_from_text(mine) or "",
        extract_dynamic_state_from_text(theirs) or "",
        character_aliases=character_alias_map(merged_lore),
        target_language=target_language or None,
        character_genders=_character_gender_map(merged_lore),
        character_profiles=_character_profile_map(merged_lore),
    )
    return build_novel_context(merged_lore, merged_dynamic), change_logs


def save_novel_context_merged(
    filename: str,
    novel_contexts_dir: Path,
    content: str,
    baseline: str = "",
    target_language: str = "",
) -> Tuple[str, List[str]]:
    """Save `content`, folding in any change made since `baseline` was read.

    Returns the document actually written together with the reconciliation log,
    which is empty when nothing had to be reconciled. Callers should keep the
    returned document as their new baseline: it is what the file now holds, and
    saving stale state over it would reopen the very gap this closes.

    Whenever the file cannot be located or read, this degrades to exactly the
    plain write it replaced. Reading is how the merge case is detected, so
    failing to read means there is nothing to detect -- not that the save should
    fail. `save_novel_context` still validates the filename and still raises.
    """
    file_path = _writable_path(filename, novel_contexts_dir)
    if file_path is None:
        return _write(filename, novel_contexts_dir, content), []

    # Held across the read and the write, so no other writer in this process can
    # land between deciding what to merge and writing the result. It is the same
    # lock `save_novel_context` takes, which is why it is re-entrant.
    with writer_lock(file_path):
        current = _read(file_path)
        if current is None or current == normalize_novel_context_content(baseline):
            return _write(filename, novel_contexts_dir, content), []

        merged, change_logs = reconcile_context_documents(
            content, current, target_language
        )
        return _write(filename, novel_contexts_dir, merged), change_logs


def _writable_path(filename: str, novel_contexts_dir: Path) -> Optional[Path]:
    try:
        return _resolve_inside(novel_contexts_dir, filename)
    except Exception:
        return None


def _read(file_path: Path) -> Optional[str]:
    """Return the file's normalized content, or None if it cannot be read."""
    try:
        if not file_path.exists():
            return None
        return normalize_novel_context_content(
            file_path.read_text(encoding="utf-8-sig")
        )
    except Exception:
        return None


def _write(filename: str, novel_contexts_dir: Path, content: str) -> str:
    # Resolved through the package namespace at call time rather than imported
    # directly, so that patching src.utils.novel_context.save_novel_context
    # intercepts this write the same way it intercepts every other one.
    from src.utils import novel_context

    novel_context.save_novel_context(filename, novel_contexts_dir, content)
    return normalize_novel_context_content(content)
