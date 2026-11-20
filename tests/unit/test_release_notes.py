import pytest

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


_PRERELEASE_CHANGELOG = """# Changelog

## 2.0.0 - Unreleased

- Release candidate notes.

## 1.18.6 - 2026-08-17

- Older release note.
"""


def test_prerelease_tag_uses_the_section_of_the_release_it_leads_to():
    notes = extract_release_notes(_PRERELEASE_CHANGELOG, "v2.0.0-rc.1")
    assert "Release candidate notes." in notes
    assert "Older release note." not in notes


def test_an_exact_prerelease_section_still_wins():
    changelog = _PRERELEASE_CHANGELOG.replace(
        "## 2.0.0 - Unreleased", "## 2.0.0-rc.1 - 2026-11-20\n\n- Exact.\n\n## 2.0.0 - Unreleased",
    )
    notes = extract_release_notes(changelog, "v2.0.0-rc.1")
    assert "Exact." in notes
    assert "Release candidate notes." not in notes


def test_missing_stable_section_is_still_an_error():
    with pytest.raises(ValueError):
        extract_release_notes(_PRERELEASE_CHANGELOG, "v2.1.0")
