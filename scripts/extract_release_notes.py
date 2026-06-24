"""Extract one version section from CHANGELOG.md for GitHub releases."""

from __future__ import annotations

import re
import sys
from pathlib import Path


_ORDERED_LIST = re.compile(r"^\d+\.\s+")
_UNORDERED_LIST = re.compile(r"^[-*+]\s+")
_FENCE = re.compile(r"^(```|~~~)")


def _is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(_UNORDERED_LIST.match(stripped) or _ORDERED_LIST.match(stripped))


def _unwrap_markdown_for_release(markdown_text: str) -> str:
    """Remove source hard-wraps from release notes.

    CHANGELOG.md is wrapped for diff readability, but GitHub release pages show
    those soft line breaks. This formatter keeps Markdown structure while
    joining paragraph and list-item continuation lines into normal prose.
    """
    output: list[str] = []
    pending: str | None = None
    in_fence = False

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            output.append(pending)
            pending = None

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if _FENCE.match(stripped):
            flush_pending()
            output.append(line)
            in_fence = not in_fence
            continue

        if in_fence:
            output.append(line)
            continue

        if not stripped:
            flush_pending()
            if output and output[-1] != "":
                output.append("")
            continue

        if stripped.startswith(("#", ">")):
            flush_pending()
            output.append(stripped)
            continue

        if _is_list_line(stripped):
            flush_pending()
            pending = stripped
            continue

        if pending is None:
            pending = stripped
        else:
            pending = f"{pending} {stripped}"

    flush_pending()
    return "\n".join(output).strip() + "\n"


def extract_release_notes(changelog_text: str, tag_name: str) -> str:
    """Return the changelog section matching a release tag.

    Args:
        changelog_text: Full changelog Markdown.
        tag_name: Release tag such as ``v1.4.14`` or ``1.4.14``.

    Raises:
        ValueError: If the matching changelog heading is missing.
    """
    return _unwrap_markdown_for_release(
        extract_release_notes_raw(changelog_text, tag_name)
    )


def extract_release_notes_raw(changelog_text: str, tag_name: str) -> str:
    """Return the matching changelog section without release-page formatting."""
    version = tag_name.removeprefix("v").strip()
    if not version:
        raise ValueError("Release tag is empty.")

    heading = re.compile(rf"^##\s+v?{re.escape(version)}(?:\s+-|\s*$)")
    lines = changelog_text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if heading.match(line)),
        None,
    )
    if start is None:
        raise ValueError(f"No changelog section found for {tag_name}.")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break

    return "\n".join(lines[start:end]).strip() + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage: extract_release_notes.py CHANGELOG.md TAG OUTPUT.md",
            file=sys.stderr,
        )
        return 2

    changelog_path = Path(argv[1])
    tag_name = argv[2]
    output_path = Path(argv[3])

    notes = extract_release_notes(
        changelog_path.read_text(encoding="utf-8"),
        tag_name,
    )
    output_path.write_text(notes, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
