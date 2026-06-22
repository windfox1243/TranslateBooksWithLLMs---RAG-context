"""Static contract for reactive localization of resumable-job cards."""

from pathlib import Path


def test_resume_cards_refresh_when_locale_changes():
    project_root = Path(__file__).resolve().parents[2]
    source = (
        project_root
        / "src"
        / "web"
        / "static"
        / "js"
        / "translation"
        / "resume-manager.js"
    ).read_text(encoding="utf-8")

    assert "window.addEventListener('localeChanged'" in source
    assert "this.loadResumableJobs();" in source
