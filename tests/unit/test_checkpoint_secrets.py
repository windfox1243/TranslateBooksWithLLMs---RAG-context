"""Unit tests for issue #213 follow-up: the jobs database must never store API keys.

Covers `sanitize_config_secrets` (the persistence-side strip), the actual DB
write paths (`create_job`, `update_job_config`), and the resume-time credential
validation that replaces reading keys back from the checkpoint.
"""
import json
import sqlite3

import pytest
from flask import Flask

from src.persistence.database import Database, sanitize_config_secrets
from src.api.blueprints.translation_routes import (
    _available_context_chunk_indices,
    _apply_resume_overrides,
    _prompt_options_from_start_request,
    _rehydrate_resume_credentials,
    _strip_api_keys,
)


SECRET_CONFIG = {
    'model': 'gemini-2.0-flash',
    'llm_provider': 'gemini',
    'llm_api_endpoint': '',
    'output_filename': 'book_fr.epub',
    'api_key': 'generic-secret',
    'gemini_api_key': 'AIza-secret',
    'openai_api_key': 'sk-secret',
    'openrouter_api_key': 'sk-or-1,sk-or-2',
    'mistral_api_key': 'mistral-secret',
    'deepseek_api_key': 'ds-secret',
    'poe_api_key': 'poe-secret',
    'nim_api_key': 'nim-secret',
    'prompt_options': {'glossary_id': 3},
}


def _read_raw_config(db_path, translation_id):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT config FROM translation_jobs WHERE translation_id = ?",
            (translation_id,),
        ).fetchone()
        return json.loads(row[0])
    finally:
        conn.close()


def _assert_no_secrets(config):
    leaked = [k for k in config if k == 'api_key' or k.endswith('_api_key')]
    assert leaked == [], f"API keys persisted in config: {leaked}"


class TestSanitizeConfigSecrets:
    def test_removes_every_api_key_field(self):
        sanitized = sanitize_config_secrets(SECRET_CONFIG)
        _assert_no_secrets(sanitized)

    def test_preserves_non_secret_fields(self):
        sanitized = sanitize_config_secrets(SECRET_CONFIG)
        assert sanitized['model'] == 'gemini-2.0-flash'
        assert sanitized['llm_provider'] == 'gemini'
        assert sanitized['output_filename'] == 'book_fr.epub'
        assert sanitized['prompt_options'] == {'glossary_id': 3}

    def test_does_not_mutate_input(self):
        config = dict(SECRET_CONFIG)
        sanitize_config_secrets(config)
        assert config['gemini_api_key'] == 'AIza-secret'


class TestDatabaseNeverStoresKeys:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "jobs.db")
        database = Database(db_path)
        yield database
        database.close()

    def test_create_job_strips_keys_from_persisted_config(self, db):
        assert db.create_job('trans_1', 'epub', dict(SECRET_CONFIG))
        raw = _read_raw_config(db.db_path, 'trans_1')
        _assert_no_secrets(raw)
        assert raw['model'] == 'gemini-2.0-flash'

    def test_create_job_leaves_caller_config_intact(self, db):
        config = dict(SECRET_CONFIG)
        db.create_job('trans_2', 'epub', config)
        # The live job still needs its keys in memory.
        assert config['gemini_api_key'] == 'AIza-secret'

    def test_update_job_config_strips_keys_too(self, db):
        db.create_job('trans_3', 'txt', {'model': 'llama3'})
        assert db.update_job_config('trans_3', dict(SECRET_CONFIG))
        raw = _read_raw_config(db.db_path, 'trans_3')
        _assert_no_secrets(raw)

    def test_get_job_returns_config_without_keys(self, db):
        db.create_job('trans_4', 'srt', dict(SECRET_CONFIG))
        job = db.get_job('trans_4')
        _assert_no_secrets(job['config'])


class TestResumeCredentialValidation:
    """Checkpoints carry no keys, so resume must validate .env/request instead."""

    @pytest.fixture
    def app_ctx(self):
        app = Flask(__name__)
        with app.app_context():
            yield

    def _restored_config(self):
        # What load_checkpoint now yields: a cloud-provider config, no keys.
        return {
            'model': 'gemini-2.0-flash',
            'llm_provider': 'gemini',
            'llm_api_endpoint': '',
        }

    def test_empty_body_resume_without_env_key_is_rejected(self, app_ctx, monkeypatch):
        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        result = _apply_resume_overrides(self._restored_config(), {})
        assert result is not None
        _response, status = result
        assert status == 400

    def test_empty_body_resume_with_env_key_passes(self, app_ctx, monkeypatch):
        monkeypatch.setenv('GEMINI_API_KEY', 'env-key')
        assert _apply_resume_overrides(self._restored_config(), {}) is None

    def test_background_resume_uses_live_config_key(self, monkeypatch):
        import src.api.blueprints.translation_routes as routes

        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        monkeypatch.setattr(routes._config, 'GEMINI_API_KEY', 'live-config-key')
        config = self._restored_config()

        _rehydrate_resume_credentials(config)

        assert config['gemini_api_key'] == 'live-config-key'

    def test_key_in_resume_request_passes_without_env(self, app_ctx, monkeypatch):
        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        config = self._restored_config()
        assert _apply_resume_overrides(config, {'api_key': 'user-key'}) is None
        assert config['gemini_api_key'] == 'user-key'

    def test_local_openai_compatible_resume_needs_no_key(self, app_ctx, monkeypatch):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        config = {
            'model': 'qwen3-14b',
            'llm_provider': 'openai',
            'llm_api_endpoint': 'http://localhost:1234/v1/chat/completions',
        }
        assert _apply_resume_overrides(config, {}) is None

    def test_official_openai_resume_without_key_is_rejected(self, app_ctx, monkeypatch):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        config = {
            'model': 'gpt-4o',
            'llm_provider': 'openai',
            'llm_api_endpoint': 'https://api.openai.com/v1/chat/completions',
        }
        result = _apply_resume_overrides(config, {})
        assert result is not None
        _response, status = result
        assert status == 400

    def test_cross_provider_editor_rehydrates_its_standard_key(self, app_ctx, monkeypatch):
        monkeypatch.setenv('GEMINI_API_KEY', 'YOUR_DRAFT_API_KEY_HERE')
        monkeypatch.setenv('OPENROUTER_API_KEY', 'YOUR_EDITOR_API_KEY_HERE')
        config = self._restored_config()
        config['prompt_options'] = {
            'editor_provider': 'openrouter',
            'editor_model': 'editor-model',
        }
        assert _apply_resume_overrides(config, {}) is None
        assert config['gemini_api_key'] == 'YOUR_DRAFT_API_KEY_HERE'
        assert config['openrouter_api_key'] == 'YOUR_EDITOR_API_KEY_HERE'

    def test_cross_provider_editor_accepts_provider_specific_override(self, app_ctx, monkeypatch):
        monkeypatch.setenv('GEMINI_API_KEY', 'YOUR_DRAFT_API_KEY_HERE')
        monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
        config = self._restored_config()
        config['prompt_options'] = {
            'editor_provider': 'openrouter',
            'editor_model': 'editor-model',
        }
        assert _apply_resume_overrides(
            config, {'openrouter_api_key': 'YOUR_EDITOR_API_KEY_HERE'}
        ) is None
        assert config['openrouter_api_key'] == 'YOUR_EDITOR_API_KEY_HERE'


class TestResponseStripping:
    def test_strip_api_keys_removes_all_keys_in_place(self):
        config = dict(SECRET_CONFIG)
        _strip_api_keys(config)
        _assert_no_secrets(config)
        assert config['model'] == 'gemini-2.0-flash'

    def test_strip_api_keys_tolerates_none(self):
        assert _strip_api_keys(None) is None


class TestStartPromptOptions:
    def test_reflection_falls_back_only_when_start_request_omits_option(self, monkeypatch):
        import src.api.blueprints.translation_routes as routes

        monkeypatch.setattr(routes._config, 'ENABLE_CHUNK_REFLECTION', 'true')

        explicit_false = _prompt_options_from_start_request({
            'prompt_options': {'reflection_mode': False}
        })
        assert explicit_false['reflection_mode'] is False

        omitted = _prompt_options_from_start_request({'prompt_options': {}})
        assert omitted['reflection_mode'] is True

        monkeypatch.setattr(routes._config, 'ENABLE_CHUNK_REFLECTION', 'false')
        no_options = _prompt_options_from_start_request({})
        assert no_options['reflection_mode'] is False


def test_available_context_indices_come_from_real_checkpoint_rows():
    checkpoint = {
        'chunks': [
            {
                'chunk_index': 0,
                'status': 'completed',
                'chunk_data': {'context_snapshot': 'snapshot-0'},
            },
            {
                'chunk_index': 2,
                'status': 'partial',
                'chunk_data': {'context_snapshot': 'snapshot-2'},
            },
            {
                'chunk_index': 4,
                'status': 'completed',
                'chunk_data': {},
            },
        ]
    }

    assert _available_context_chunk_indices(checkpoint) == [0, 2]
