"""Unit tests for the LLM endpoint allowlist and the endpoint/key pairing rule.

Two guards are covered:

1. `EndpointValidator` — a request-supplied endpoint must be loopback/private
   or an allowlisted host.
2. The pairing rule in `translation_routes` — an endpoint that differs from the
   server default must carry its own API key, so the `.env` key never travels
   to a host the request chose.
"""
from types import SimpleNamespace

import pytest
from flask import Flask

import src.config as _config
from src.api.api_keys import resolve_api_key
from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.services.endpoint_validator import EndpointValidator

# Inert placeholder — never a real credential.
FAKE_KEY = 'sk-xxxxxxxx'
OPENAI_DEFAULT_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
# Allowlisted by the subdomain rule, but not the configured default.
OPENAI_ALT_ENDPOINT = 'https://eu.api.openai.com/v1/chat/completions'


# ---------------------------------------------------------------------------
# EndpointValidator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('endpoint', [None, '', '   '])
def test_absent_endpoint_is_accepted(endpoint):
    """No endpoint means 'use the server default', which is not an override."""
    assert EndpointValidator.validate(endpoint) == (True, None)


@pytest.mark.parametrize('endpoint', [
    'http://localhost:11434/api/generate',
    'http://127.0.0.1:1234/v1/chat/completions',
    'http://192.168.1.50:11434/api/generate',
    'http://host.docker.internal:11434/api/generate',
    'http://10.0.0.7:8080/v1/chat/completions',
    'http://[::1]:11434/api/generate',
    'http://ollama.localhost/api/generate',
])
def test_local_and_private_endpoints_are_allowed(endpoint):
    assert EndpointValidator.validate(endpoint) == (True, None)


@pytest.mark.parametrize('endpoint', [
    OPENAI_DEFAULT_ENDPOINT,
    OPENAI_ALT_ENDPOINT,  # subdomain rule
    'https://openrouter.ai/api/v1/chat/completions',
    'https://integrate.api.nvidia.com/v1/chat/completions',
])
def test_known_provider_hosts_are_allowed(endpoint):
    assert EndpointValidator.validate(endpoint) == (True, None)


def test_unknown_public_host_is_rejected():
    ok, message = EndpointValidator.validate('https://evil.example.com/v1/chat/completions')
    assert ok is False
    assert 'evil.example.com' in message
    assert 'LLM_ENDPOINT_ALLOWLIST' in message


def test_non_http_scheme_is_rejected():
    ok, message = EndpointValidator.validate('ftp://api.openai.com/x')
    assert ok is False
    assert 'http or https' in message


def test_embedded_credentials_are_rejected():
    ok, message = EndpointValidator.validate('https://user:pw@api.openai.com/v1')
    assert ok is False
    assert 'credentials' in message


def test_env_allowlist_extends_accepted_hosts(monkeypatch):
    """The allowlist is read at call time so reload_config() is honoured."""
    rejected, _ = EndpointValidator.validate('https://llm.internal.example/v1')
    assert rejected is False

    monkeypatch.setattr(_config, 'LLM_ENDPOINT_ALLOWLIST', ('llm.internal.example',))
    assert EndpointValidator.validate('https://llm.internal.example/v1') == (True, None)
    # Subdomains of an allowlisted host are covered too.
    assert EndpointValidator.validate('https://eu.llm.internal.example/v1') == (True, None)


def test_configured_server_default_is_always_allowed(monkeypatch):
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', 'https://gateway.example.org/v1/chat/completions')
    assert EndpointValidator.validate('https://gateway.example.org/v1/chat/completions') == (True, None)


# ---------------------------------------------------------------------------
# resolve_api_key(allow_env_fallback=...)
# ---------------------------------------------------------------------------

def test_env_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', FAKE_KEY)
    assert resolve_api_key('__USE_ENV__', 'OPENAI_API_KEY', allow_env_fallback=False) == ''
    assert resolve_api_key('', 'OPENAI_API_KEY', allow_env_fallback=False) == ''


def test_env_fallback_is_on_by_default(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', FAKE_KEY)
    assert resolve_api_key('__USE_ENV__', 'OPENAI_API_KEY') == FAKE_KEY


def test_explicit_key_survives_disabled_fallback(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    assert resolve_api_key(FAKE_KEY, 'OPENAI_API_KEY', allow_env_fallback=False) == FAKE_KEY


# ---------------------------------------------------------------------------
# POST /api/translate
# ---------------------------------------------------------------------------

class _RecordingStateManager:
    """Minimal stand-in that records the config a job was created with."""

    def __init__(self):
        self.created = []
        # The fork's blueprint factory reconciles interrupted operations
        # through the checkpoint database on creation.
        self.checkpoint_manager = SimpleNamespace(db=SimpleNamespace())

    def create_translation(self, translation_id, config):
        self.created.append((translation_id, dict(config)))


@pytest.fixture
def translate_app(tmp_path):
    """Flask app exposing only the translation blueprint (no auth gate)."""
    state_manager = _RecordingStateManager()
    started = []

    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager,
        lambda translation_id, config: started.append(translation_id),
        str(tmp_path),
    ))

    with app.test_client() as client:
        yield client, state_manager, started


def _payload(**overrides):
    body = {
        'text': 'Hello world.',
        'source_language': 'English',
        'target_language': 'French',
        'model': 'gpt-4o',
        'llm_api_endpoint': OPENAI_DEFAULT_ENDPOINT,
        'output_filename': 'out.txt',
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def deterministic_openai_endpoint(monkeypatch):
    """Pin the OpenAI default so 'override' is unambiguous regardless of .env."""
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', OPENAI_DEFAULT_ENDPOINT)


def test_disallowed_endpoint_is_rejected_before_job_creation(translate_app):
    client, state_manager, started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint='https://evil.example.com/v1/chat/completions',
        openai_api_key=FAKE_KEY,
    ))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'Endpoint not allowed'
    assert state_manager.created == []
    assert started == []


def test_endpoint_override_without_own_key_is_rejected(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint=OPENAI_ALT_ENDPOINT,
        openai_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 400
    # The allowlist accepted the host; it is the pairing rule that refused.
    assert resp.get_json()['error'] == 'Endpoint override requires its own API key'
    assert state_manager.created == []


def test_endpoint_override_with_own_key_is_accepted(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint=OPENAI_ALT_ENDPOINT,
        openai_api_key=FAKE_KEY,
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1
    _translation_id, config = state_manager.created[0]
    assert config['openai_api_key'] == FAKE_KEY


def test_gemini_endpoint_field_is_inert(translate_app, monkeypatch):
    """Gemini ignores llm_api_endpoint, so the pairing guard must not fire."""
    client, state_manager, _started = translate_app
    monkeypatch.setenv('GEMINI_API_KEY', 'env-gemini-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='gemini',
        model='gemini-2.0-flash',
        llm_api_endpoint=OPENAI_ALT_ENDPOINT,
        gemini_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1


def test_default_ollama_endpoint_needs_no_key(translate_app):
    """The common local path must not regress."""
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint=_config.API_ENDPOINT,
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1


def test_keyless_local_openai_server_is_accepted(translate_app, monkeypatch):
    """LM Studio / llama.cpp / vLLM run as provider 'openai' with no stored key.

    Nothing can leak, so the pairing rule must not fire.
    """
    client, state_manager, _started = translate_app
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        model='qwen3-14b',
        llm_api_endpoint='http://localhost:1234/v1/chat/completions',
        openai_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['openai_api_key'] == ''


def test_stored_key_never_reaches_a_local_override(translate_app, monkeypatch):
    """A private-range host is allowlisted, so the pairing rule is the only
    thing standing between the .env key and the operator's LAN."""
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint='http://192.168.1.50:1234/v1/chat/completions',
        openai_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'Endpoint override requires its own API key'
    assert state_manager.created == []


def test_custom_ollama_endpoint_needs_no_key(translate_app):
    """Ollama has no stored credential to leak, so an override is fine."""
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint='http://192.168.1.50:11434/api/generate',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1


# ---------------------------------------------------------------------------
# Senior Editor endpoint (fork-specific)
#
# The editor can run on its own provider and endpoint
# (prompt_options.editor_provider / editor_api_endpoint). That endpoint is a
# second client-chosen host, so the pairing rule must hold for it too.
# ---------------------------------------------------------------------------

def _editor_options(endpoint, provider='openai', model='gpt-4o'):
    return {
        'editor_provider': provider,
        'editor_model': model,
        'editor_api_endpoint': endpoint,
    }


def test_disallowed_editor_endpoint_is_rejected(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint=_config.API_ENDPOINT,
        openai_api_key=FAKE_KEY,
        prompt_options=_editor_options('https://evil.example.com/v1/chat/completions'),
    ))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'Endpoint not allowed'
    assert state_manager.created == []


def test_editor_endpoint_override_without_own_key_is_rejected(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint=_config.API_ENDPOINT,
        openai_api_key='__USE_ENV__',
        prompt_options=_editor_options('http://192.168.1.50:1234/v1/chat/completions'),
    ))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'Endpoint override requires its own API key'
    assert state_manager.created == []


def test_editor_endpoint_override_with_own_key_never_carries_env_key(
    translate_app, monkeypatch
):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint=_config.API_ENDPOINT,
        openai_api_key=FAKE_KEY,
        prompt_options=_editor_options(OPENAI_ALT_ENDPOINT),
    ))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['openai_api_key'] == FAKE_KEY


def test_editor_on_default_endpoint_may_use_env_key(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint=_config.API_ENDPOINT,
        openai_api_key='__USE_ENV__',
        prompt_options=_editor_options(OPENAI_DEFAULT_ENDPOINT),
    ))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['openai_api_key'] == 'env-only-key'


def test_resume_with_editor_override_and_no_key_is_rejected(app_context, monkeypatch):
    from src.api.blueprints.translation_routes.helpers import _apply_resume_overrides

    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    config = {
        'llm_provider': 'ollama',
        'llm_api_endpoint': _config.API_ENDPOINT,
        'prompt_options': _editor_options('http://192.168.1.50:1234/v1/chat/completions'),
    }
    result = _apply_resume_overrides(config, {})
    assert result is not None
    response, status = result
    assert status == 400
    assert response.get_json()['error'] == 'Endpoint override requires its own API key'
    assert 'openai_api_key' not in config


def test_resume_with_editor_override_and_own_key_drops_env_key(app_context, monkeypatch):
    from src.api.blueprints.translation_routes.helpers import _apply_resume_overrides

    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    config = {
        'llm_provider': 'ollama',
        'llm_api_endpoint': _config.API_ENDPOINT,
        'prompt_options': _editor_options(OPENAI_ALT_ENDPOINT),
    }
    assert _apply_resume_overrides(config, {'editor_api_key': FAKE_KEY}) is None
    assert config['openai_api_key'] == FAKE_KEY


@pytest.fixture
def app_context():
    app = Flask(__name__)
    with app.app_context():
        yield app
