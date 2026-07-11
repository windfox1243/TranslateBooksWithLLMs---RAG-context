"""Regression coverage for packaged .env and web-default initialization."""

import json
import os
from pathlib import Path
import subprocess
import sys

from flask import Flask

from src import config
from src.api.blueprints.config_routes import create_config_blueprint


ROOT = Path(__file__).resolve().parents[2]


def test_explicit_config_root_env_overrides_inherited_process_values(tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=gemini\n"
        "DEFAULT_MODEL=gemini-test-model\n"
        "GEMINI_API_KEY=YOUR_API_KEY_HERE\n"
        "ENABLE_CHUNK_REFLECTION=true\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "TRANSLATEBOOK_CONFIG_DIR": str(tmp_path),
            "LLM_PROVIDER": "ollama",
            "DEFAULT_MODEL": "stale-process-model",
            "GEMINI_API_KEY": "",
            "ENABLE_CHUNK_REFLECTION": "false",
        }
    )
    script = (
        "import json; from src import config; "
        "print(json.dumps({"
        "'root': str(config.CONFIG_DIR), "
        "'provider': config.LLM_PROVIDER, "
        "'model': config.DEFAULT_MODEL, "
        "'key_configured': bool(config.GEMINI_API_KEY), "
        "'reflection': config.ENABLE_CHUNK_REFLECTION"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "root": str(tmp_path.resolve()),
        "provider": "gemini",
        "model": "gemini-test-model",
        "key_configured": True,
        "reflection": "true",
    }


def test_api_config_exposes_provider_reflection_and_masked_key_count(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "ENABLE_CHUNK_REFLECTION", "true")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "FIRST_PLACEHOLDER,SECOND_PLACEHOLDER")

    app = Flask(__name__)
    app.register_blueprint(create_config_blueprint(server_session_id=1))
    with app.test_client() as client:
        response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["llm_provider"] == "gemini"
    assert payload["enable_chunk_reflection"] is True
    assert payload["gemini_api_key_configured"] is True
    assert payload["gemini_api_key_count"] == 2
    assert "FIRST_PLACEHOLDER" not in response.get_data(as_text=True)
    assert "SECOND_PLACEHOLDER" not in response.get_data(as_text=True)
