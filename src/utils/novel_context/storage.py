"""Filesystem layout for novel context files: naming, listing, load and save."""
from __future__ import annotations

import os
import re
import threading
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.atomic_replace import replace_atomically

from .constants import (
    ADDRESSING_SECTION,
    ALIASES_SECTION,
    CHARACTERS_SECTION,
    GLOSSARY_SECTION,
    NAME_MAP_SECTION,
    RELATIONSHIP_SECTION,
    SAFE_FILENAME_PUNCTUATION,
    WINDOWS_RESERVED_FILENAMES,
)
from .merge import build_novel_context, normalize_novel_context_content

_WRITER_LOCKS: Dict[str, "threading.RLock"] = {}
_WRITER_LOCKS_GUARD = threading.Lock()


def writer_lock(file_path: Path):
    """Return the lock guarding writes to one context file.

    Keyed by resolved path so two spellings of the same file share a lock. The
    table is never pruned: there is one entry per context file the process has
    written, which is bounded by how many novels the user is working on.

    Re-entrant because a caller that needs its read and its write to be one step
    -- see reconcile.py -- takes this lock and then calls `save_novel_context`,
    which takes it again.
    """
    key = str(file_path.resolve())
    with _WRITER_LOCKS_GUARD:
        lock = _WRITER_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _WRITER_LOCKS[key] = lock
        return lock


_writer_lock = writer_lock
def is_safe_filename(filename: str) -> bool:
    """Return whether a context filename is safe while preserving Unicode names."""
    if not filename or filename != filename.strip():
        return False
    if not filename.lower().endswith(".txt"):
        return False
    if any(
        ord(char) < 32 or not (char.isalnum() or char in SAFE_FILENAME_PUNCTUATION)
        for char in filename
    ):
        return False
    if filename in {".", ".."}:
        return False
    stem = filename[:-4].rstrip(". ")
    if not stem or stem in {".", ".."}:
        return False
    if stem.split(".", 1)[0].lower() in WINDOWS_RESERVED_FILENAMES:
        return False
    return True
def _safe_context_filename_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    safe_chars = [
        char if char.isalnum() or char in SAFE_FILENAME_PUNCTUATION else "_"
        for char in normalized
    ]
    return "".join(safe_chars).strip(".")
def _resolve_inside(directory: Path, filename: str) -> Optional[Path]:
    """Return the file path if it resolves inside `directory`, else None."""
    candidate = directory / filename
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate
def list_novel_contexts(novel_contexts_dir: Path) -> List[Dict[str, Any]]:
    """List text files in the directory.

    Returns a list of dicts:
        {
            "filename": "my_novel.txt",
            "display_name": "my_novel",
            "format": "txt"
        }
    """
    if not novel_contexts_dir.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for file_path in novel_contexts_dir.glob("*.txt"):
        try:
            file_path.resolve().relative_to(novel_contexts_dir.resolve())
        except ValueError:
            continue

        entries.append(
            {
                "filename": file_path.name,
                "display_name": file_path.stem,
                "format": "txt",
            }
        )

    entries.sort(key=lambda e: e["display_name"].lower())
    return entries
def load_novel_context(filename: str, novel_contexts_dir: Path) -> str:
    """Load context content from a safe filename. Creates empty template if missing."""
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: Unicode letters/numbers, `_`, `-`, `.`; extension must be .txt."
        )

    file_path = _resolve_inside(novel_contexts_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Novel_Contexts directory."
        )

    if not file_path.exists():
        # Empty sections are intentional. Example bullets used to leak into
        # real context files as fake characters such as "[Name]" and "[None]".
        template = build_novel_context(
            (
                "# GLOBAL LORE\n"
                "(Characters, genders, and terminology; canonical names only.)\n\n"
                f"{CHARACTERS_SECTION}\n\n"
                f"{ALIASES_SECTION}\n\n"
                f"{NAME_MAP_SECTION}\n\n"
                f"{GLOSSARY_SECTION}"
            ),
            (
                f"{ADDRESSING_SECTION}\n\n"
                f"{RELATIONSHIP_SECTION}"
            ),
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(template, encoding="utf-8")
        return template

    return normalize_novel_context_content(
        file_path.read_text(encoding="utf-8-sig")
    )
def save_novel_context(filename: str, novel_contexts_dir: Path, content: str) -> None:
    """Atomically save context content to a safe filename."""
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: Unicode letters/numbers, `_`, `-`, `.`; extension must be .txt."
        )

    file_path = _resolve_inside(novel_contexts_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Novel_Contexts directory."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_novel_context_content(content)

    # Two things make this safe under concurrent jobs, and they cover different
    # failures. The temporary name is unique per writer, so two jobs sharing one
    # context file -- normal when translating several volumes of the same novel
    # -- cannot write into each other's staging file and leave a blend of both
    # behind. The lock then serializes writers to the same path within this
    # process, so the last replace wins whole rather than racing.
    #
    # This makes each write atomic. It does not make a caller's read, edit and
    # write-back atomic; two jobs interleaving there still lose one edit.
    with writer_lock(file_path):
        temporary_path = file_path.with_name(
            f"{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(normalized, encoding="utf-8")
            replace_atomically(temporary_path, file_path)
        finally:
            # The replace consumed it on success; on failure it is a stray file.
            temporary_path.unlink(missing_ok=True)


def resolve_novel_context_path(filename: str, novel_contexts_dir: Path) -> Path:
    """Resolve novel context file path. Checks directory first, then absolute/relative."""
    if is_safe_filename(filename):
        resolved = _resolve_inside(novel_contexts_dir, filename)
        if resolved:
            return resolved

    # Check if the filename's basename is a safe filename.
    # If the app is frozen, or if the path contains 'Novel_Contexts' / 'TranslateBook_Data',
    # redirect the file resolution to the current local novel_contexts_dir.
    # This prevents absolute paths from old/unbuilt directories leaking in when the executable is moved.
    # os.path.basename only understands the host separator. Check both so
    # Windows checkpoints migrate correctly on Linux or macOS, and vice versa.
    base_name = re.split(r"[\\/]", str(filename))[-1]
    if is_safe_filename(base_name):
        import sys
        is_frozen = getattr(sys, 'frozen', False)
        path_str = str(filename).replace('\\', '/')
        if is_frozen or 'novel_contexts' in path_str.lower() or 'translatebook_data' in path_str.lower():
            resolved = _resolve_inside(novel_contexts_dir, base_name)
            if resolved:
                return resolved

    if os.path.isabs(filename):
        return Path(filename).resolve()

    # Try relative to current working directory
    return Path(filename).resolve()
