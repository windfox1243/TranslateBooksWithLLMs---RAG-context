"""The web UI can read, save and profile the cost preset."""

from pathlib import Path

import pytest
from dotenv import dotenv_values
from flask import Flask

import src.config as cfg
from src.api.blueprints import config_routes, profile_routes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings_client(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.setattr(config_routes, "get_config_path", lambda: str(tmp_path))
    monkeypatch.setattr(
        config_routes,
        "reload_config",
        lambda: monkeypatch.setattr(
            cfg, "COST_PRESET", dotenv_values(env_file).get("COST_PRESET") or "balanced",
        ),
    )
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint(server_session_id=1))
    with app.test_client() as client:
        client.env_file = env_file
        yield client


@pytest.mark.parametrize("stored, shown", [("economy", "economy"), ("Quality", "quality"), ("junk", "balanced")])
def test_config_reports_the_preset_the_ui_should_select(settings_client, monkeypatch, stored, shown):
    monkeypatch.setattr(cfg, "COST_PRESET", stored, raising=False)
    assert settings_client.get("/api/config").get_json()["cost_preset"] == shown


def test_saving_a_preset_writes_it_to_env(settings_client):
    resp = settings_client.post("/api/settings", json={"COST_PRESET": "Economy"})
    assert resp.status_code == 200
    assert dotenv_values(settings_client.env_file)["COST_PRESET"] == "economy"


def test_an_unknown_preset_is_not_saved(settings_client):
    resp = settings_client.post("/api/settings", json={"COST_PRESET": "cheapest"})
    assert resp.status_code == 400
    assert not settings_client.env_file.exists()


def test_profiles_keep_a_valid_preset_and_reject_others():
    assert profile_routes._normalize_profile({"cost_preset": "Quality"})["cost_preset"] == "quality"
    with pytest.raises(ValueError):
        profile_routes._normalize_profile({"cost_preset": "cheapest"})


def test_every_entry_point_sends_the_selected_preset():
    js = ROOT / "src" / "web" / "static" / "js"
    snippet = "cost_preset: DomHelpers.getValue('costPreset') || 'balanced'"
    assert snippet in (js / "ui" / "form-manager.js").read_text(encoding="utf-8")
    assert snippet in (js / "translation" / "batch-controller.js").read_text(encoding="utf-8")
    template = (ROOT / "src" / "web" / "templates" / "translation_interface.html").read_text(encoding="utf-8")
    for value in ("economy", "balanced", "quality"):
        assert f'value="{value}" data-i18n="settings:cost_preset_{value}"' in template
