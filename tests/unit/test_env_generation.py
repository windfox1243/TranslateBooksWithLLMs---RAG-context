import os
import sys
from pathlib import Path

import launcher
from src.utils.env_helper import create_env_file, render_compact_env


def test_compact_env_contains_current_knobs_without_reference_wall():
    text = render_compact_env()

    assert "NOVEL_CONTEXT_PROMPT_MAX_TOKENS=1800" in text
    assert "NOVEL_CONTEXT_UPDATE_INTERVAL=1" in text
    assert "NOVEL_CONTEXT_SOURCE_MEMORY_CHARS=6000" in text
    assert "MAX_TOKENS_PER_CHUNK=450" in text
    assert "GEMINI_MODEL=" not in text
    assert "Multi-key support" not in text
    assert "Translation Attribution" not in text
    assert len(text.splitlines()) <= 30


def test_create_env_file_does_not_copy_env_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text(
        "# Very long reference\nMULTI_KEY_SUPPORT_DOCS=1\n",
        encoding="utf-8",
    )

    assert create_env_file() is True
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "NOVEL_CONTEXT_PROMPT_MAX_TOKENS=1800" in text
    assert "NOVEL_CONTEXT_UPDATE_INTERVAL=1" in text
    assert "NOVEL_CONTEXT_SOURCE_MEMORY_CHARS=6000" in text
    assert "MULTI_KEY_SUPPORT_DOCS=1" not in text


def test_launcher_first_run_writes_compact_env_with_reference_copy(
    tmp_path,
    monkeypatch,
):
    exe_dir = tmp_path / "exe"
    bundle_dir = tmp_path / "bundle"
    exe_dir.mkdir()
    bundle_dir.mkdir()
    (bundle_dir / ".env.example").write_text(
        "# Full reference wall\nREFERENCE_ONLY=1\n",
        encoding="utf-8",
    )

    old_cwd = Path.cwd()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "TranslateBook.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    try:
        launcher.setup_working_directory()
    finally:
        os.chdir(old_cwd)

    data_dir = exe_dir / "TranslateBook_Data"
    env_text = (data_dir / ".env").read_text(encoding="utf-8")
    example_text = (data_dir / ".env.example").read_text(encoding="utf-8")

    assert "NOVEL_CONTEXT_PROMPT_MAX_TOKENS=1800" in env_text
    assert "NOVEL_CONTEXT_UPDATE_INTERVAL=1" in env_text
    assert "NOVEL_CONTEXT_SOURCE_MEMORY_CHARS=6000" in env_text
    assert "REFERENCE_ONLY=1" not in env_text
    assert "REFERENCE_ONLY=1" in example_text
