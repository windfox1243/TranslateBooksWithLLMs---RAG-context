"""Language-matrix tests for deterministic source-residue validation."""

from types import SimpleNamespace

import pytest

from src.core.translator import ReflectionValidationError, run_chunk_reflection_pass
from src.utils.translation_quality import find_source_residue, validate_editor_repair


@pytest.mark.parametrize(
    ("source_language", "target_language", "source", "draft", "span"),
    [
        ("English", "Vietnamese", "Brother, come here.", "Brother, lại đây.", "Brother"),
        ("Chinese", "English", "他拔出了圣剑。", "He drew 圣剑.", "圣剑"),
        ("Korean", "English", "그녀는 선생님을 불렀다.", "She called 선생님.", "선생님"),
        ("Arabic", "English", "قال يا أخي.", "He said أخي.", "أخي"),
        ("Thai", "English", "เขาเรียกพี่ชาย", "He called พี่ชาย", "พี่ชาย"),
        ("English", "French", "They walked into the room.", "Ils walked into the room.", "walked into the room"),
    ],
)
def test_source_residue_language_matrix(
    source_language,
    target_language,
    source,
    draft,
    span,
):
    findings = find_source_residue(
        source,
        draft,
        source_language=source_language,
        target_language=target_language,
    )
    assert any(span.casefold() in item.draft_span.casefold() for item in findings)


def test_source_residue_excludes_names_markup_and_preserved_gram():
    findings = find_source_residue(
        "Gram <i>struck</i> [[0]] Tomio Momozawa.",
        "Gram <i>đánh trúng</i> [[0]] Tomio Momozawa.",
        source_language="English",
        target_language="Vietnamese",
        protected_terms=["Tomio Momozawa"],
        glossary_terms={"Gram": "Gram"},
    )
    assert findings == []


def test_source_residue_excludes_fragments_of_protected_name():
    findings = find_source_residue(
        "Frondier de Roach entered the room.",
        "Frondier de Roach bước vào phòng.",
        source_language="English",
        target_language="Vietnamese",
        protected_terms=["Frondier de Roach"],
    )
    assert findings == []


def test_source_residue_does_not_block_generic_two_word_loan_phrase():
    findings = find_source_residue(
        "The recent game over was final.",
        "Lần game over vừa rồi là cuối cùng.",
        source_language="English",
        target_language="Vietnamese",
    )
    assert findings == []


def test_editor_repair_validates_only_the_located_pronoun_occurrence():
    errors = validate_editor_repair(
        "Ông nói với anh ấy.",
        [{
            "draft_quote": "Anh nói với anh ấy.",
            "draft_replacement": {"draft": "Anh", "replacement": "Ông"},
        }],
        draft_text="Anh nói với anh ấy.",
        source_text="He spoke with him.",
        source_language="English",
        target_language="Vietnamese",
    )
    assert errors == []


def test_editor_repair_rejects_ambiguous_issue_locator():
    errors = validate_editor_repair(
        "Ông nói với anh ấy.",
        [{
            "draft_quote": "anh",
            "draft_replacement": {"draft": "anh", "replacement": "ông"},
        }],
        draft_text="Anh nói với anh ấy.",
        source_text="He spoke with him.",
        source_language="English",
        target_language="Vietnamese",
    )
    assert any(item.startswith("issue_locator_ambiguous") for item in errors)


@pytest.mark.asyncio
async def test_no_issues_cannot_bypass_brother_residue():
    class Client:
        def __init__(self):
            self.responses = [
                '{"status":"no_issues","issues":[]}',
                "<TRANSLATION>Anh, lại đây.</TRANSLATION>",
            ]
            self.calls = 0

        async def generate_async(self, **_kwargs):
            response = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=response)

    client = Client()
    repaired = await run_chunk_reflection_pass(
        source_chunk="Brother, come here.",
        draft_translation="Brother, lại đây.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={
            "context_contract_version": 2,
            "source_language": "English",
            "source_residue_validation": True,
        },
    )
    assert repaired == "Anh, lại đây."
    assert client.calls == 2


@pytest.mark.asyncio
async def test_incomplete_editor_replacement_contract_retries_once():
    complete = (
        '{"status":"needs_repair","issues":[{'
        '"category":"mistranslation","severity":"major",'
        '"source_quote":"Brother","draft_quote":"Brother",'
        '"instruction":"Translate the address term.",'
        '"draft_replacement":{"draft":"Brother","replacement":"Anh"},'
        '"glossary_update":null}]}'
    )

    class Client:
        def __init__(self):
            self.responses = [
                complete.replace(
                    '"draft_replacement":{"draft":"Brother","replacement":"Anh"},',
                    '"draft_replacement":null,',
                ),
                complete,
                "<TRANSLATION>Anh, lại đây.</TRANSLATION>",
            ]
            self.calls = 0

        async def generate_async(self, **_kwargs):
            response = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=response)

    client = Client()
    repaired = await run_chunk_reflection_pass(
        source_chunk="Brother, come here.",
        draft_translation="Brother, lại đây.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={
            "context_contract_version": 2,
            "source_language": "English",
            "source_residue_validation": True,
        },
    )
    assert repaired == "Anh, lại đây."
    # The corrected exact issue is applied locally; no full-chunk repair call.
    assert client.calls == 2


@pytest.mark.asyncio
async def test_unresolved_semantic_repair_preserves_valid_draft_for_review():
    critique = (
        '{"status":"needs_repair","issues":[{'
        '"category":"omission","severity":"major",'
        '"source_quote":"Come here.","draft_quote":"",'
        '"instruction":"Restore an omitted detail.",'
        '"draft_replacement":null,'
        '"glossary_update":null}]}'
    )

    class Client:
        def __init__(self):
            self.responses = [
                critique,
                "",
                "",
            ]
            self.calls = 0

        async def generate_async(self, **_kwargs):
            response = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=response)

    result = await run_chunk_reflection_pass(
        source_chunk="Come here.",
        draft_translation="Lại đây.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=Client(),
        prompt_options={
            "context_contract_version": 2,
            "source_language": "English",
            "source_residue_validation": True,
        },
    )
    assert result == "Lại đây."


@pytest.mark.asyncio
async def test_ambiguous_locator_is_corrected_before_any_repair_call():
    ambiguous = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"pronoun-1","category":"pronoun","severity":"major",'
        '"confidence":0.95,"source_quote":"He spoke with him.",'
        '"draft_quote":"anh","instruction":"Use the senior form.",'
        '"draft_replacement":{"draft":"anh","replacement":"ông"},'
        '"glossary_update":null}]}'
    )
    corrected = ambiguous.replace(
        '"draft_quote":"anh"',
        '"draft_quote":"Anh nói với anh ấy."',
    ).replace('"draft":"anh"', '"draft":"Anh"')

    class Client:
        def __init__(self):
            self.responses = [ambiguous, corrected]
            self.calls = []

        async def generate_async(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content=self.responses[len(self.calls) - 1])

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk="He spoke with him.",
        draft_translation="Anh nói với anh ấy.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={"source_language": "English"},
    )
    assert result == "ông nói với anh ấy."
    assert len(client.calls) == 2
    assert client.calls[0]["temperature"] == 0.2
    assert client.calls[1]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_no_op_editor_issue_is_review_only_without_rewrite():
    critique = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"noop-1","category":"style","severity":"major",'
        '"source_quote":"Wait.","draft_quote":"Chờ đã.",'
        '"instruction":"Keep the existing wording.",'
        '"draft_replacement":{"draft":"Chờ đã.","replacement":"Chờ đã."},'
        '"glossary_update":null}]}'
    )

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=critique)

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk="Wait.",
        draft_translation="Chờ đã.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={"source_language": "English"},
    )
    assert result == "Chờ đã."
    assert client.calls == 1


@pytest.mark.asyncio
async def test_local_patch_is_not_revalidated_with_stale_locator():
    critique = (
        '{"status":"needs_repair","issues":['
        '{"issue_id":"pronoun-1","category":"pronoun","severity":"major",'
        '"source_quote":"You can go.","draft_quote":"Cậu có thể đi.",'
        '"instruction":"Use the senior form.",'
        '"draft_replacement":{"draft":"Cậu","replacement":"Anh"},'
        '"glossary_update":null},'
        '{"issue_id":"omission-1","category":"omission","severity":"major",'
        '"source_quote":"Please return.","draft_quote":"",'
        '"instruction":"Restore the omitted sentence.",'
        '"draft_replacement":null,"glossary_update":null}]}'
    )

    class EditorClient:
        def __init__(self):
            self.responses = [
                critique,
                "<TRANSLATION>Anh có thể đi. Xin hãy quay lại.</TRANSLATION>",
            ]
            self.calls = 0

        async def generate_async(self, **_kwargs):
            value = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=value)

        def extract_translation(self, raw):
            return raw.removeprefix("<TRANSLATION>").removesuffix("</TRANSLATION>")

    class DraftClient:
        def extract_translation(self, _raw):
            raise AssertionError("repair extraction must use the editor client")

    editor = EditorClient()
    result = await run_chunk_reflection_pass(
        source_chunk="You can go. Please return.",
        draft_translation="Cậu có thể đi.",
        target_language="Vietnamese",
        model_name="draft",
        llm_client=DraftClient(),
        prompt_options={
            "source_language": "English",
            "_editor_llm_client": editor,
            "editor_model_resolved": "editor",
        },
    )
    assert result == "Anh có thể đi. Xin hãy quay lại."
    assert editor.calls == 2


def test_nonblocking_same_script_residue_is_not_a_repair_error():
    errors = validate_editor_repair(
        "Đây là game over cuối cùng.",
        [],
        draft_text="Đây là game over cuối cùng.",
        source_text="This is the final game over.",
        source_language="English",
        target_language="Vietnamese",
    )
    assert errors == []
