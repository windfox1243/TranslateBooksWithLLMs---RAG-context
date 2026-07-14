"""Language-matrix tests for deterministic source-residue validation."""

from types import SimpleNamespace

import pytest

from src.core.translator import ReflectionValidationError, run_chunk_reflection_pass
from src.persistence.database import Database
from src.utils.translation_quality import (
    apply_local_editor_patches,
    filter_protected_span_editor_issues,
    find_source_residue,
    format_editor_segments,
    identity_preserving_proper_names,
    normalize_unique_issue_locators,
    validate_editor_repair,
    validate_issue_locators,
)


def test_segment_locator_disambiguates_repeated_text():
    draft = "Wait here.\nWait here."
    issue = {
        "issue_id": "issue-1",
        "segment_id": "SEG-0002",
        "repair_kind": "local_replace",
        "draft_quote": "Wait here.",
        "draft_replacement": {"draft": "Wait", "replacement": "Stay"},
    }
    assert "[SEG-0001] Wait here." in format_editor_segments(draft)
    assert "[SEG-0002] Wait here." in format_editor_segments(draft)
    assert validate_issue_locators(draft, [issue]) == []
    repaired, unresolved, errors = apply_local_editor_patches(draft, [issue])
    assert repaired == "Wait here.\nStay here."
    assert unresolved == []
    assert errors == []


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
        '"repair_kind":"local_replace",'
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
    # Only the malformed issue is corrected; the exact edit is then local.
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
async def test_optional_editor_escalation_runs_once_after_repeated_fingerprint():
    critique = (
        '{"status":"needs_repair","issues":[{'
        '"category":"omission","severity":"major",'
        '"source_quote":"Come here now.","draft_quote":"",'
        '"instruction":"Restore the omitted detail.",'
        '"draft_replacement":null,"glossary_update":null}]}'
    )

    class BaseClient:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=critique if self.calls == 1 else "")

    class EscalationClient:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                content="<TRANSLATION>Lại đây ngay.</TRANSLATION>"
            )

    base = BaseClient()
    escalation = EscalationClient()
    result = await run_chunk_reflection_pass(
        source_chunk="Come here now.",
        draft_translation="Lại đây.",
        target_language="Vietnamese",
        model_name="small",
        llm_client=base,
        prompt_options={
            "context_contract_version": 5,
            "source_language": "English",
            "editor_escalation_enabled": True,
            "editor_escalation_model": "large",
            "_editor_escalation_llm_client": escalation,
        },
    )

    assert result == "Lại đây ngay."
    assert base.calls == 3
    assert escalation.calls == 1


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
        '"draft_quote":"Anh nói với"',
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


def test_unique_replacement_span_repairs_model_locator_deterministically():
    issue = {
        "issue_id": "issue-1",
        "repair_kind": "local_replace",
        "draft_quote": "wrong quote",
        "draft_replacement": {"draft": "Cậu", "replacement": "Anh"},
    }
    normalized, repaired = normalize_unique_issue_locators(
        "Cậu có thể đi. Một câu khác.", [issue],
    )
    assert repaired == ["issue-1"]
    assert normalized[0]["draft_quote"] == "Cậu"
    assert normalized[0]["segment_id"] == "SEG-0001"
    assert validate_issue_locators(
        "Cậu có thể đi. Một câu khác.", normalized,
    ) == []


def test_complete_glossary_entity_masks_embedded_social_residue():
    findings = find_source_residue(
        "Lord Axe entered the hall.",
        "Lord Axe bước vào đại sảnh.",
        source_language="English",
        target_language="Vietnamese",
        glossary_terms={"Lord Axe": "Lord Axe"},
    )
    assert findings == []


def test_unchanged_multiword_proper_name_is_protected_without_glossary():
    assert identity_preserving_proper_names(
        "Lord Axe entered beside Special Week.",
        "Lord Axe bước vào cạnh Special Week.",
    ) == ["Lord Axe", "Special Week"]


def test_model_edit_inside_complete_protected_entity_is_rejected():
    issue = {
        "issue_id": "title-1",
        "draft_quote": "Lord Axe",
        "draft_replacement": {"draft": "Lord", "replacement": "Ngài"},
    }
    retained, rejected = filter_protected_span_editor_issues(
        "Lord Axe entered. Lord Axe won.",
        [issue],
        glossary_terms={"Lord Axe": "Lord Axe"},
    )
    assert retained == []
    assert rejected == ["title-1"]


def test_full_rewrite_cannot_remove_identity_preserving_glossary_term():
    errors = validate_editor_repair(
        "Ngài Axe bước vào.",
        [],
        draft_text="Lord Axe bước vào.",
        source_text="Lord Axe entered.",
        source_language="English",
        target_language="Vietnamese",
        glossary_terms={"Lord Axe": "Lord Axe"},
    )
    assert "protected_term_removed: Lord Axe" in errors


@pytest.mark.asyncio
async def test_editor_cannot_translate_token_inside_protected_name():
    critique = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"title-1","category":"terminology",'
        '"severity":"major","confidence":0.99,"repair_kind":"local_replace",'
        '"source_quote":"Lord Axe","draft_quote":"Lord Axe",'
        '"instruction":"Translate the title.",'
        '"draft_replacement":{"draft":"Lord","replacement":"Ngài"},'
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
        source_chunk="Lord Axe entered.",
        draft_translation="Lord Axe bước vào.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={
            "source_language": "English",
            "glossary_terms": {"Lord Axe": "Lord Axe"},
        },
    )
    assert result == "Lord Axe bước vào."
    assert client.calls == 1


@pytest.mark.asyncio
async def test_partial_local_patch_preserves_review_required_metadata(monkeypatch):
    import src.core.translator as translator_module

    critique = (
        '{"status":"needs_repair","issues":['
        '{"issue_id":"valid","category":"pronoun","severity":"major",'
        '"confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"You may go.","draft_quote":"Cậu có thể đi.",'
        '"instruction":"Use the senior form.",'
        '"draft_replacement":{"draft":"Cậu","replacement":"Anh"},'
        '"glossary_update":null},'
        '{"issue_id":"ambiguous","category":"style","severity":"major",'
        '"confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"Wait.","draft_quote":"đợi",'
        '"instruction":"Change one occurrence.",'
        '"draft_replacement":{"draft":"đợi","replacement":"chờ"},'
        '"glossary_update":null}]}'
    )

    original_parse = translator_module.parse_reflection_result

    def parse_with_deterministic_finding(value):
        parsed = original_parse(value)
        for issue in parsed.issues:
            if issue.get("issue_id") == "ambiguous":
                issue["deterministic"] = True
        return parsed

    monkeypatch.setattr(
        translator_module, "parse_reflection_result", parse_with_deterministic_finding,
    )

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=critique)

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk="You may go. Wait. Wait.",
        draft_translation="Cậu có thể đi. Hãy đợi. Rồi đợi.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={"source_language": "English"},
    )

    assert result == "Anh có thể đi. Hãy đợi. Rồi đợi."
    assert result.quality_status == "review_required"
    assert result.editor_validation["stage"] == "locator_validation"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_large_editor_input_is_reviewed_in_complete_bounded_windows(monkeypatch):
    from src.core.editor import EditorService

    source = ("Source sentence. " * 900).strip()
    draft = ("Câu bản thảo. " * 900).strip()
    calls = []

    async def review_window(_self, **kwargs):
        calls.append(kwargs)
        return kwargs["draft_translation"]

    monkeypatch.setattr(EditorService, "review_chunk", review_window)
    result = await run_chunk_reflection_pass(
        source_chunk=source,
        draft_translation=draft,
        target_language="Vietnamese",
        model_name="test",
        llm_client=object(),
        prompt_options={
            "source_language": "English",
            "file_type": "txt",
            "editor_max_input_tokens": 5000,
        },
    )

    assert result == draft
    assert len(calls) > 1
    assert "".join(call["source_chunk"] for call in calls) == source
    assert "".join(call["draft_translation"] for call in calls) == draft
    assert all(call["prompt_options"]["_editor_windowed"] for call in calls)


@pytest.mark.asyncio
async def test_large_editor_input_stays_whole_without_configured_model_limit(monkeypatch):
    from src.core.editor import EditorService

    source = ("Source sentence. " * 900).strip()
    draft = ("Câu bản thảo. " * 900).strip()
    calls = []

    async def review_unit(_self, **kwargs):
        calls.append(kwargs)
        return kwargs["draft_translation"]

    monkeypatch.setattr(EditorService, "review_chunk", review_unit)
    result = await run_chunk_reflection_pass(
        source_chunk=source,
        draft_translation=draft,
        target_language="Vietnamese",
        model_name="test",
        llm_client=object(),
        prompt_options={"source_language": "English", "file_type": "txt"},
    )

    assert result == draft
    assert len(calls) == 1
    assert calls[0]["source_chunk"] == source
    assert calls[0]["draft_translation"] == draft
    assert not calls[0]["prompt_options"].get("_editor_windowed")


@pytest.mark.asyncio
async def test_unsupported_model_locator_is_warning_not_review():
    critique = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"unsupported-1","category":"style","severity":"major",'
        '"confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"Wait.","draft_quote":"missing phrase",'
        '"instruction":"Change the phrase.",'
        '"draft_replacement":{"draft":"missing phrase","replacement":"replacement"},'
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
    assert not hasattr(result, "quality_status")
    assert client.calls == 2


@pytest.mark.asyncio
async def test_unique_replacement_outside_quote_is_grounded_without_rewrite():
    malformed = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"bad-1","category":"style","severity":"major",'
        '"confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"Source evidence.","draft_quote":"Short quote",'
        '"instruction":"Replace the longer phrase.",'
        '"draft_replacement":{"draft":"phrase outside quote",'
        '"replacement":"correct phrase"},"glossary_update":null}]}'
    )

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=malformed)

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk="Source evidence.",
        draft_translation="Short quote and phrase outside quote.",
        target_language="English",
        model_name="test",
        llm_client=client,
        prompt_options={"source_language": "English"},
    )
    assert result == "Short quote and correct phrase."
    assert client.calls == 1


@pytest.mark.asyncio
async def test_isolated_issue_retry_preserves_valid_sibling():
    initial = (
        '{"status":"needs_repair","issues":['
        '{"issue_id":"valid-1","category":"terminology","severity":"major",'
        '"confidence":0.95,"repair_kind":"local_replace",'
        '"source_quote":"Coach","draft_quote":"Huấn luyện viên",'
        '"instruction":"Use the glossary term.",'
        '"draft_replacement":{"draft":"Huấn luyện viên","replacement":"HLV"},'
        '"glossary_update":null},'
        '{"issue_id":"bad-2","category":"style","severity":"major",'
        '"confidence":0.90,"repair_kind":"local_replace",'
        '"source_quote":"spoke","draft_quote":"nói nhỏ",'
        '"instruction":"Improve the verb.","draft_replacement":null,'
        '"glossary_update":null}]}'
    )
    corrected_only = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"bad-2","category":"style","severity":"major",'
        '"confidence":0.90,"repair_kind":"local_replace",'
        '"source_quote":"spoke","draft_quote":"nói nhỏ",'
        '"instruction":"Improve the verb.",'
        '"draft_replacement":{"draft":"nói nhỏ","replacement":"thì thầm"},'
        '"glossary_update":null}]}'
    )

    class Client:
        def __init__(self):
            self.responses = [initial, corrected_only]
            self.calls = 0

        async def generate_async(self, **_kwargs):
            value = self.responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=value)

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk="Coach spoke.",
        draft_translation="Huấn luyện viên nói nhỏ.",
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={"source_language": "English"},
    )
    assert result == "HLV thì thầm."
    assert client.calls == 2


@pytest.mark.asyncio
async def test_minor_issue_is_warning_without_rewrite(tmp_path):
    critique = (
        '{"status":"needs_repair","issues":[{'
        '"issue_id":"minor-1","category":"style","severity":"minor",'
        '"confidence":0.99,"repair_kind":"local_replace",'
        '"source_quote":"Wait.","draft_quote":"Chờ đã.",'
        '"instruction":"Optional stylistic preference.",'
        '"draft_replacement":{"draft":"Chờ đã.","replacement":"Đợi nhé."},'
        '"glossary_update":null}]}'
    )

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=critique)

    client = Client()
    db_path = str(tmp_path / "jobs.db")
    db = Database(db_path)
    assert db.create_job("warning-job", "txt", {})
    result = await run_chunk_reflection_pass(
        source_chunk="Wait.", draft_translation="Chờ đã.",
        target_language="Vietnamese", model_name="test", llm_client=client,
        prompt_options={
            "source_language": "English",
            "translation_id": "warning-job",
            "jobs_db_path": db_path,
            "chunk_index": 0,
        },
    )
    assert result == "Chờ đã."
    assert client.calls == 1
    diagnostics = db.get_editor_diagnostics("warning-job")
    assert diagnostics["summary"]["outcomes"] == {"warnings_only": 1}
    assert diagnostics["summary"]["warnings"] == 1


def test_editor_repair_validates_capitalization_only_correction():
    errors = validate_editor_repair(
        "Đây là các Cuộc đua Tuyển chọn.",
        [{
            "issue_id": "issue-1",
            "draft_quote": "các cuộc đua tuyển chọn",
            "draft_replacement": {
                "draft": "các cuộc đua tuyển chọn",
                "replacement": "các Cuộc đua Tuyển chọn"
            }
        }],
        draft_text="Đây là các cuộc đua tuyển chọn.",
        source_text="This is the Selection Races.",
        source_language="English",
        target_language="Vietnamese"
    )
    assert errors == []


def test_pronoun_counting_narrator_hint_was_removed():
    import src.utils.translation_quality as quality

    assert not hasattr(quality, "build_narrative_voice_context")


@pytest.mark.asyncio
async def test_warnings_only_cannot_hide_narrator_violation_and_retry_converges():
    source = (
        "I crossed the station.\n"
        "I remembered the rain.\n"
        "I promised I would continue."
    )
    leaking = (
        "Tớ băng qua nhà ga.\n"
        "Tớ nhớ cơn mưa.\n"
        "Tớ tự hứa sẽ tiếp tục."
    )
    corrected = leaking.replace("Tớ", "Tôi")

    class Client:
        def __init__(self):
            self.responses = [
                '{"status":"no_issues","issues":[]}',
                f"<TRANSLATION>{leaking}</TRANSLATION>",
                f"<TRANSLATION>{corrected}</TRANSLATION>",
            ]
            self.calls = []

        async def generate_async(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content=self.responses[len(self.calls) - 1])

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk=source,
        draft_translation=leaking,
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={
            "source_language": "English",
            "file_type": "txt",
            "narrator_self_reference_override": "tôi",
        },
    )

    assert result == corrected
    assert len(client.calls) == 3
    assert client.calls[2]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_narrator_retry_exhaustion_preserves_review_required_draft():
    source = "I crossed the station.\nI remembered rain.\nI kept moving."
    leaking = "Tớ qua ga.\nTớ nhớ mưa.\nTớ tiếp tục."

    class Client:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content='{"status":"no_issues","issues":[]}')
            return SimpleNamespace(
                content=f"<TRANSLATION>{leaking}</TRANSLATION>"
            )

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk=source,
        draft_translation=leaking,
        target_language="Vietnamese",
        model_name="test",
        llm_client=client,
        prompt_options={
            "source_language": "English",
            "file_type": "txt",
            "narrator_self_reference_override": "tôi",
        },
    )

    assert result == leaking
    assert result.quality_status == "review_required"
    assert result.editor_validation["status"] == "blocked"
    assert client.calls == 3
