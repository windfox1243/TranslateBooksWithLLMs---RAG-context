"""analyze_unit_context: the pre-translation analysis every pipeline shares."""

import pytest

from src.core.context import analysis


class _Session:
    def __init__(self, fail=False):
        self.fail = fail
        self.content = "BEFORE"
        self.global_lore = ""
        self.dialogue_attribution = {"turns": []}
        self.addressing_candidates = ["a"]
        self.addressing_parse_status = "json"
        self.relationship_candidates = ["r"]
        self.relationship_parse_status = "json"
        self.saved = 0
        self.path = type("P", (), {"name": "ctx.md"})()

    async def analyze_source(self, **kwargs):
        if self.fail:
            raise RuntimeError("model unavailable")
        self.content = "AFTER"
        return ["changed"]

    def save(self):
        self.saved += 1


class _Db:
    def get_relationship_edges(self, translation_id, statuses):
        return [{"is_locked": True, "id": 1}, {"is_locked": False, "id": 2}]


@pytest.fixture
def calls(monkeypatch):
    seen = []

    async def judge(**kwargs):
        seen.append(("judge", [e["id"] for e in kwargs["locked_facts"]]))
        return ["judged"]

    def record(name):
        def fn(*args, **kwargs):
            seen.append((name, kwargs.get("fallback_context")))
        return fn

    monkeypatch.setattr("src.utils.relationship_sync.judge_ambiguous_relationship_candidates", judge)
    monkeypatch.setattr("src.utils.relationship_sync.apply_relationship_graph_to_session", record("apply_graph"))
    monkeypatch.setattr("src.utils.db_addressing.apply_db_addressing_to_session", record("apply_addressing"))
    monkeypatch.setattr("src.core.context.unit_pipeline.commit_source_social_evidence", record("social"))
    monkeypatch.setattr("src.core.context.unit_pipeline.relevant_character_names", lambda **kw: ["Alice"])
    return seen


async def _analyze(session, **overrides):
    kwargs = dict(
        context_session=session,
        db=_Db(),
        translation_id="job",
        llm_client=object(),
        model_name="m",
        source_text="SOURCE",
        source_language="English",
        target_language="Vietnamese",
        local_index=0,
        total_units=3,
        global_index=10,
        scene_key=None,
        contract_version=5,
        relationship_mode="project",
        prompt_options={},
    )
    kwargs.update(overrides)
    return await analysis.analyze_unit_context(**kwargs)


@pytest.mark.asyncio
async def test_project_mode_produces_pending_records_and_projects(calls):
    session = _Session()
    result = await _analyze(session)
    assert result.analyzed
    assert result.addressing["candidates"] == ["a"]
    assert result.relationships["candidates"] == ["judged"]
    assert result.relationships["active_character_names"] == ["Alice"]
    assert ("judge", [1]) in calls
    # Projection reads the context as it was before this unit's analysis.
    assert ("apply_addressing", "BEFORE") in calls
    assert ("apply_graph", "BEFORE") in calls
    assert session.saved == 1


@pytest.mark.asyncio
async def test_shadow_mode_never_projects_into_the_session(calls):
    session = _Session()
    result = await _analyze(session, relationship_mode="shadow")
    assert result.relationships is not None
    assert not [name for name, _ in calls if name.startswith("apply_")]
    assert session.saved == 0


@pytest.mark.asyncio
async def test_relationship_mode_off_skips_the_judge(calls):
    result = await _analyze(_Session(), relationship_mode="off")
    assert result.relationships is None
    assert not [name for name, _ in calls if name == "judge"]


@pytest.mark.asyncio
async def test_without_a_database_nothing_is_pending(calls):
    result = await _analyze(_Session(), db=None)
    assert result.analyzed
    assert result.addressing is None and result.relationships is None
    assert calls == []


@pytest.mark.asyncio
async def test_a_failing_analysis_is_logged_not_raised(calls):
    events = []
    result = await _analyze(
        _Session(fail=True), log_callback=lambda *a, **k: events.append(a[0])
    )
    assert not result.analyzed
    assert "novel_context_update_failed" in events
