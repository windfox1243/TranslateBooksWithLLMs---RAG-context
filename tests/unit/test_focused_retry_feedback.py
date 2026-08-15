"""The focused locator retry may not ask the same question three times.

Every input to the retry prompt is fixed before the loop that sends it, so
without the rejection its previous answer earned, attempts two and three are
attempt one resent -- which is what nineteen measured runs did, spending 100k
tokens on retries that carried no new information.
"""

from types import SimpleNamespace

import pytest

from src.core.editor.prompting import _build_focused_locator_retry_prompt
from src.core.translator import run_chunk_reflection_pass

DRAFT = "Câu thứ nhất ở đây. Câu thứ hai ở đây. Câu thứ ba ở đây."
ISSUES = [{
    "issue_id": "ISSUE-001",
    "category": "mistranslation",
    "segment_id": "SEG-0001",
    "draft_quote": "Câu thứ hai ở đây.",
    "draft_replacement": {"draft": "thứ hai", "replacement": "thứ nhì"},
}]
IDS = {"ISSUE-001"}
ERRORS = ["locator_ambiguous:ISSUE-001"]


def test_the_first_attempt_carries_no_rejection():
    prompt = _build_focused_locator_retry_prompt(DRAFT, ISSUES, IDS, ERRORS)
    assert "WAS REJECTED FOR" not in prompt


def test_a_rejected_attempt_is_told_what_it_was_rejected_for():
    prompt = _build_focused_locator_retry_prompt(
        DRAFT, ISSUES, IDS, ERRORS,
        previous_attempt_errors=["local_patch_conflict:ISSUE-002:ISSUE-001"],
    )
    assert "WAS REJECTED FOR" in prompt
    assert "local_patch_conflict:ISSUE-002:ISSUE-001" in prompt


def test_the_second_attempt_is_not_the_first_one_resent():
    first = _build_focused_locator_retry_prompt(DRAFT, ISSUES, IDS, ERRORS)
    second = _build_focused_locator_retry_prompt(
        DRAFT, ISSUES, IDS, ERRORS,
        previous_attempt_errors=["locator_missing:ISSUE-001"],
    )
    third = _build_focused_locator_retry_prompt(
        DRAFT, ISSUES, IDS, ERRORS,
        previous_attempt_errors=["locator_ambiguous:ISSUE-001"],
    )
    assert first != second
    assert second != third


def test_an_empty_rejection_list_reads_as_no_rejection():
    assert _build_focused_locator_retry_prompt(
        DRAFT, ISSUES, IDS, ERRORS, previous_attempt_errors=[],
    ) == _build_focused_locator_retry_prompt(DRAFT, ISSUES, IDS, ERRORS)


INITIAL = (
    '{"status":"needs_repair","issues":['
    '{"issue_id":"a","segment_id":"D0001","category":"wording",'
    '"severity":"major","confidence":0.95,"repair_kind":"local_replace",'
    '"source_quote":"evidence","draft_quote":"alpha beta gamma",'
    '"instruction":"first","draft_replacement":{"draft":"alpha beta",'
    '"replacement":"X"},"glossary_update":null},'
    '{"issue_id":"b","segment_id":"D0001","category":"wording",'
    '"severity":"major","confidence":0.95,"repair_kind":"local_replace",'
    '"source_quote":"evidence","draft_quote":"alpha beta gamma",'
    '"instruction":"second","draft_replacement":{"draft":"beta gamma",'
    '"replacement":"Y"},"glossary_update":null}],"voice_observations":[]}'
)
CORRECTED = (
    '{"status":"needs_repair","issues":[{"issue_id":"a",'
    '"segment_id":"D0001","category":"wording","severity":"major",'
    '"confidence":0.99,"repair_kind":"local_replace",'
    '"source_quote":"evidence","draft_quote":"alpha beta gamma",'
    '"instruction":"combined","draft_replacement":'
    '{"draft":"alpha beta gamma","replacement":"X Y"},'
    '"glossary_update":null}],"voice_observations":[]}'
)


@pytest.mark.asyncio
async def test_the_three_live_retries_do_not_send_one_prompt_three_times():
    class Client:
        def __init__(self):
            self.responses = [
                INITIAL,
                '{"status":"no_issues","issues":[]}',
                '{"status":"needs_repair","issues":[]}',
                CORRECTED,
            ]
            self.requests = []

        async def generate_async(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(content=self.responses.pop(0))

    client = Client()
    result = await run_chunk_reflection_pass(
        source_chunk="evidence",
        draft_translation="alpha beta gamma",
        target_language="English",
        model_name="test",
        llm_client=client,
        prompt_options={"context_contract_version": 5, "source_language": "English"},
    )

    assert result == "X Y"
    retries = [
        item["prompt"] for item in client.requests
        if str(item["stage"]).startswith("local_patch_retry")
    ]
    assert len(retries) == 3
    assert len(set(retries)) == 3


def test_the_issues_and_candidates_still_come_through():
    prompt = _build_focused_locator_retry_prompt(
        DRAFT, ISSUES, IDS, ERRORS, previous_attempt_errors=["locator_missing"],
    )
    assert "ISSUE-001" in prompt
    assert "INVALID ISSUES AND CANDIDATE SEGMENTS" in prompt
    assert "Câu thứ hai ở đây." in prompt
