"""Extract one version section from CHANGELOG.md for GitHub releases."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_release_notes(changelog_text: str, tag_name: str) -> str:
    """Return the changelog section matching a release tag.

    Args:
        changelog_text: Full changelog Markdown.
        tag_name: Release tag such as ``v1.4.14`` or ``1.4.14``.

    Raises:
        ValueError: If the matching changelog heading is missing.
    """
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
