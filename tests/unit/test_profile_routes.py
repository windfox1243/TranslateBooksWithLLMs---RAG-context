import json
from pathlib import Path

import pytest
from flask import Flask

from src.api.blueprints import profile_routes


@pytest.fixture
def profile_client(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_routes, "TRANSLATION_PROFILES_DIR", tmp_path)
    app = Flask(__name__)
    app.register_blueprint(profile_routes.create_profile_blueprint())
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, tmp_path


def test_profile_round_trip_preserves_all_non_secret_settings(profile_client):
    client, profile_dir = profile_client
    payload = {
        "source_language": "English",
        "target_language": "French",
        "llm_provider": "openai",
        "model": "example-model",
        "llm_api_endpoint": "https://example.test/v1/chat/completions",
        "glossary": "12",
        "custom_instruction_file": "literary.txt",
        "novel_context_file": "novel_context.txt",
        "bilingual_output": True,
        "text_cleanup": True,
        "auto_update_context": True,
        "plain_text_mode": True,
        "chapter_mode": True,
        "auto_pause_on_rate_limit": False,
        "parallel_workers": 4,
        "tts_enabled": True,
        "tts_voice": "voice",
        "tts_rate": "+10%",
        "tts_format": "mp3",
        "tts_bitrate": "128k",
        "output_filename_pattern": "{originalName}-{targetLang}.{ext}",
        "reflection_mode": True,
        "bypass_context_gating": True,
        "editor_provider": "gemini",
        "editor_model": "gemini-3-flash-preview",
        "draft_thinking_level": "low",
        "editor_thinking_level": "minimal",
        "editor_max_output_tokens": "8192",
        "auto_review_repair_threshold": 3,
        "translation_id": "old-job",
        "resume_from_index": 42,
        "context_snapshot": "old-snapshot",
        "dialogue_attribution": {"state_after": {"speaker": "Old Speaker"}},
    }

    response = client.post("/api/profiles/Novel Project", json=payload)
    assert response.status_code == 200
    assert not list(profile_dir.glob("*.tmp"))

    saved = json.loads((profile_dir / "Novel Project.json").read_text(encoding="utf-8"))
    assert saved["profile_version"] == 1
    assert "api_key" not in saved
    assert "translation_id" not in saved
    assert "resume_from_index" not in saved
    assert "context_snapshot" not in saved
    assert "dialogue_attribution" not in saved

    response = client.get("/api/profiles/Novel Project")
    assert response.status_code == 200
    returned = response.get_json()
    assert returned == saved
    assert returned["plain_text_mode"] is True
    assert returned["chapter_mode"] is True
    assert returned["parallel_workers"] == 4
    assert returned["draft_thinking_level"] == "low"
    assert returned["editor_thinking_level"] == "minimal"
    assert returned["editor_max_output_tokens"] == "8192"
    assert returned["auto_review_repair_threshold"] == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"source_language": "English", "openai_api_key": "sk-xxxxxxxx"},
        {"source_language": "English", "nested": {"token": "secret"}},
        {
            "source_language": "English",
            "llm_api_endpoint": "https://user:password@example.test/v1",
        },
        {
            "source_language": "English",
            "llm_api_endpoint": "https://example.test/v1?api_key=secret",
        },
    ],
)
def test_profiles_reject_credentials(profile_client, payload):
    client, _ = profile_client
    response = client.post("/api/profiles/Unsafe", json=payload)
    assert response.status_code == 400


def test_profile_names_and_values_are_validated(profile_client):
    client, _ = profile_client
    assert client.post("/api/profiles/..", json={"source_language": "English"}).status_code == 400
    assert client.post(
        "/api/profiles/Valid",
        json={"source_language": "English", "parallel_workers": 99},
    ).status_code == 400
    assert client.post(
        "/api/profiles/Valid",
        json={"source_language": "English", "auto_review_repair_threshold": 21},
    ).status_code == 400


def test_profile_listing_is_sorted_and_corrupt_files_are_reported(profile_client):
    client, profile_dir = profile_client
    (profile_dir / "zeta.json").write_text("{}", encoding="utf-8")
    (profile_dir / "Alpha.json").write_text("{broken", encoding="utf-8")

    response = client.get("/api/profiles")
    assert response.status_code == 200
    assert response.get_json() == ["Alpha", "zeta"]

    response = client.get("/api/profiles/Alpha")
    assert response.status_code == 422


def test_profile_frontend_restores_complete_state_without_timer_race():
    project_root = Path(__file__).resolve().parents[2]
    form_manager = (
        project_root / "src" / "web" / "static" / "js" / "ui" / "form-manager.js"
    ).read_text(encoding="utf-8")
    api_client = (
        project_root / "src" / "web" / "static" / "js" / "core" / "api-client.js"
    ).read_text(encoding="utf-8")

    assert "await ProviderManager.waitForCurrentModelLoad()" in form_manager
    assert "await GlossaryManager.refreshDropdown()" in form_manager
    assert "await FormManager.loadNovelContexts()" in form_manager
    assert "await FormManager.loadCustomInstructions()" in form_manager
    assert "option.value === String(data.glossary)" in form_manager
    assert "option.value === data.novel_context_file" in form_manager
    assert "option.value === data.custom_instruction_file" in form_manager
    assert "ApiClient.getProfiles()" in form_manager
    assert "ApiClient.getProfile(name)" in form_manager
    assert "ApiClient.saveProfile(name, profileData)" in form_manager
    assert "-- Custom Settings --" not in form_manager

    for field in (
        "plain_text_mode",
        "chapter_mode",
        "auto_pause_on_rate_limit",
        "parallel_workers",
        "tts_enabled",
        "output_filename_pattern",
    ):
        assert field in form_manager

    for method in ("getProfiles", "getProfile", "saveProfile", "deleteProfile"):
        assert f"async {method}(" in api_client
