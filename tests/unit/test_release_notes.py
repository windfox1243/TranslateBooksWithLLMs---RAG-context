from scripts.extract_release_notes import extract_release_notes


def test_extract_release_notes_only_current_section_and_unwraps_lines():
    changelog = """# Changelog

## 1.4.14 - 2026-06-24

This stable release has a hard-wrapped
intro paragraph.

### Improved

- First bullet is wrapped
  onto another source line.
- Second bullet stays short.

## 1.4.13 - 2026-06-23

- Older release note.
"""

    notes = extract_release_notes(changelog, "v1.4.14")

    assert "## 1.4.14 - 2026-06-24" in notes
    assert "## 1.4.13" not in notes
    assert "hard-wrapped intro paragraph" in notes
    assert "wrapped onto another source line" in notes
    assert "wrapped\nintro" not in notes
    assert "wrapped\n  onto" not in notes
