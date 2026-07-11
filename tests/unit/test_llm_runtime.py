"""Provider runtime routing for independent draft and editor clients."""

from src.core.llm.runtime import build_draft_and_editor_clients, build_runtime_spec


def test_same_provider_and_model_reuses_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    draft, editor, draft_spec, editor_spec = build_draft_and_editor_clients(
        draft_provider="gemini",
        draft_model="same-model",
        draft_endpoint="",
        prompt_options={},
        credentials={},
    )
    assert editor is draft
    assert editor_spec == draft_spec


def test_same_provider_different_model_uses_isolated_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    draft, editor, draft_spec, editor_spec = build_draft_and_editor_clients(
        draft_provider="gemini",
        draft_model="draft-model",
        draft_endpoint="",
        prompt_options={"editor_model": "editor-model"},
        credentials={},
    )
    assert editor is not draft
    assert editor_spec.api_key == draft_spec.api_key


def test_cross_provider_uses_editor_key_and_endpoint():
    draft, editor, draft_spec, editor_spec = build_draft_and_editor_clients(
        draft_provider="gemini",
        draft_model="draft-model",
        draft_endpoint="http://draft.invalid",
        prompt_options={
            "editor_provider": "openai",
            "editor_model": "editor-model",
            "editor_api_endpoint": "https://editor.invalid/v1/chat/completions",
        },
        credentials={
            "gemini_api_key": "YOUR_DRAFT_API_KEY_HERE",
            "openai_api_key": "YOUR_EDITOR_API_KEY_HERE",
        },
    )
    assert editor is not draft
    assert draft_spec.api_key == "YOUR_DRAFT_API_KEY_HERE"
    assert editor_spec.api_key == "YOUR_EDITOR_API_KEY_HERE"
    assert editor_spec.api_endpoint == "https://editor.invalid/v1/chat/completions"


def test_nim_runtime_accepts_standard_nim_key():
    spec = build_runtime_spec(
        "nim", "model", credentials={"nim_api_key": "YOUR_NIM_API_KEY_HERE"}
    )
    assert spec.api_key == "YOUR_NIM_API_KEY_HERE"
    assert spec.key_required is True
