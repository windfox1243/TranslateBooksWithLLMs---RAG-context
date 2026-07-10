"""Language-matrix tests for deterministic source-residue validation."""

from types import SimpleNamespace

import pytest

from src.core.translator import ReflectionValidationError, run_chunk_reflection_pass
from src.utils.translation_quality import find_source_residue


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
    assert client.calls == 3


@pytest.mark.asyncio
async def test_unresolved_major_repair_fails_contract_v2_chunk():
    critique = (
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
                critique,
                "<TRANSLATION>Brother, lại đây.</TRANSLATION>",
                "<TRANSLATION>Brother, lại đây.</TRANSLATION>",
            ]
            self.calls = 0

        async def generate_async(self, **_kwargs):
            response = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=response)

    with pytest.raises(ReflectionValidationError):
        await run_chunk_reflection_pass(
            source_chunk="Brother, come here.",
            draft_translation="Brother, lại đây.",
            target_language="Vietnamese",
            model_name="test",
            llm_client=Client(),
            prompt_options={
                "context_contract_version": 2,
                "source_language": "English",
                "source_residue_validation": True,
            },
        )
