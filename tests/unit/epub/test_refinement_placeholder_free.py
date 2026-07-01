import pytest

from src.core.epub.xhtml_translator import _refine_epub_chunks


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.context_used = 15
        self.context_limit = 1000


class _PlaceholderDroppingClient:
    def __init__(self):
        self.seen_user_prompt = ""

    async def make_request(self, user_prompt, model_name, system_prompt=None):
        self.seen_user_prompt = user_prompt
        return _FakeResponse("Xin chao polished world")

    def extract_translation(self, content):
        return content


class _NoopContextTracker:
    current_dialogue_attribution = None

    async def next_context(self, **kwargs):
        return ""


@pytest.mark.asyncio
async def test_epub_refinement_hides_and_reinserts_placeholders():
    client = _PlaceholderDroppingClient()
    translated_chunks = ["[id4]Xin chao[id5] world[id6]"]
    chunks = [
        {
            "local_tag_map": {
                "[id0]": "<p>",
                "[id1]": "<em>",
                "[id2]": "</p>",
            },
            "global_indices": [4, 5, 6],
        }
    ]

    refined = await _refine_epub_chunks(
        translated_chunks=translated_chunks,
        chunks=chunks,
        target_language="Vietnamese",
        model_name="fake-model",
        llm_client=client,
        context_manager=None,
        placeholder_format=("[id", "]"),
        log_callback=None,
        prompt_options={"structured_refinement_hide_placeholders": True},
        context_tracker=_NoopContextTracker(),
    )

    assert "[id0]" not in client.seen_user_prompt
    assert refined != translated_chunks
    assert "polished" in refined[0]
    assert all(placeholder in refined[0] for placeholder in ("[id4]", "[id5]", "[id6]"))
