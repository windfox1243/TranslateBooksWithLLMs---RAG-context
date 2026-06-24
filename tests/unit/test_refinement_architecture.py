from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from flask import Flask

from src.core.llm.base import LLMResponse
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.novel_context import (
    RefinementContextTracker,
    build_novel_context,
    compress_dynamic_state,
    map_context_snapshots_for_refinement,
    normalize_refinement_context,
)


def test_full_context_snapshots_map_without_becoming_nested_dynamic_state():
    full_context = build_novel_context("GLOBAL LORE", "HISTORICAL STATE")
    db_chunks = [
        {
            "chunk_index": 0,
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(full_context),
            },
        }
    ]

    mapped = map_context_snapshots_for_refinement(
        total_chunks=2,
        db_chunks=db_chunks,
        fallback_context=build_novel_context("FALLBACK", ""),
    )

    assert mapped == [full_context, full_context]
    assert mapped[0].count("---DYNAMIC_STATE_START---") == 1


def test_refinement_context_mapping_uses_translated_text_position():
    context_a = build_novel_context("CONTEXT A", "STATE A")
    context_b = build_novel_context("CONTEXT B", "STATE B")
    db_chunks = [
        {
            "chunk_index": 0,
            "translated_text": "a" * 100,
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(context_a),
            },
        },
        {
            "chunk_index": 1,
            "translated_text": "b" * 10,
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(context_b),
            },
        },
    ]

    mapped = map_context_snapshots_for_refinement(
        total_chunks=3,
        db_chunks=db_chunks,
        fallback_context="",
        refinement_units=["x" * 50, "y" * 50, "z" * 10],
    )

    assert mapped == [context_a, context_a, context_b]


def test_refinement_context_mapping_ignores_failed_or_partial_snapshots():
    context_a = build_novel_context("CONTEXT A", "STATE A")
    failed_context = build_novel_context("FAILED", "POISONED STATE")
    partial_context = build_novel_context("PARTIAL", "PARTIAL STATE")
    context_b = build_novel_context("CONTEXT B", "STATE B")
    db_chunks = [
        {
            "chunk_index": 0,
            "status": "completed",
            "translated_text": "translated a",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(context_a),
            },
        },
        {
            "chunk_index": 1,
            "status": "failed",
            "translated_text": "source fallback",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(failed_context),
            },
        },
        {
            "chunk_index": 2,
            "status": "partial",
            "translated_text": "source fallback",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(partial_context),
            },
        },
        {
            "chunk_index": 3,
            "status": "completed",
            "translated_text": "translated b",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(context_b),
            },
        },
    ]

    mapped = map_context_snapshots_for_refinement(
        total_chunks=2,
        db_chunks=db_chunks,
        fallback_context="",
    )

    assert mapped == [context_a, context_b]


def test_refinement_context_uses_final_lore_with_historical_dynamic_state():
    historical = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Kriha: Unspecified, Captain.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Kriha → Valentine: source form \"Major\" | formal\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Kriha → Valentine: Newly assigned subordinate."
        ),
    )
    final_context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Kriha: Female, Captain and loyal subordinate.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
            "- blood art: huyết thuật\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Kriha → Valentine: source form \"Valentine\" | intimate\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Kriha → Valentine: Deep mutual trust."
        ),
    )

    combined = normalize_refinement_context(historical, final_context)

    assert "- Kriha: Female, Captain and loyal subordinate." in combined
    assert "- blood art: huyết thuật" in combined
    assert 'source form "Major" | formal' in combined
    assert "Newly assigned subordinate" in combined
    assert 'source form "Valentine" | intimate' not in combined
    assert "Deep mutual trust" not in combined
    assert "Unspecified" not in combined


def test_context_editor_uses_latest_lore_with_selected_chunk_state(
    monkeypatch,
    tmp_path,
):
    from src.api.blueprints.translation_routes import (
        create_translation_blueprint,
    )
    from src.utils.novel_context import save_novel_context
    import src.config

    historical = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Kriha: Unspecified, Captain.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Kriha → Valentine: source form \"Major\" | formal"
        ),
    )
    latest = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Kriha: Female, Captain and loyal subordinate.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
            "- blood art: huyết thuật\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Kriha → Valentine: source form \"Valentine\" | intimate"
        ),
    )
    save_novel_context("novel.txt", tmp_path, latest)
    monkeypatch.setattr(src.config, "NOVEL_CONTEXTS_DIR", tmp_path)

    checkpoint_manager = MagicMock()
    checkpoint_manager.load_checkpoint.return_value = {
        "job": {
            "config": {
                "prompt_options": {
                    "novel_context_file": "novel.txt",
                }
            }
        },
        "chunks": [{
            "chunk_index": 0,
            "status": "completed",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(historical),
            },
        }],
    }
    state_manager = MagicMock()
    state_manager.checkpoint_manager = checkpoint_manager

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state_manager,
            lambda *_args, **_kwargs: None,
            str(tmp_path),
        )
    )

    with app.test_client() as client:
        response = client.get("/api/translation/job/context/0")

    assert response.status_code == 200
    content = response.get_json()["context_content"]
    assert "- Kriha: Female, Captain and loyal subordinate." in content
    assert "- blood art: huyết thuật" in content
    assert 'source form "Major" | formal' in content
    assert 'source form "Valentine" | intimate' not in content
    assert "Unspecified" not in content


def test_context_resync_route_accepts_numeric_context_revision(
    monkeypatch,
    tmp_path,
):
    from src.api.blueprints.translation_routes import (
        create_translation_blueprint,
    )

    edited = build_novel_context(
        "# GLOBAL LORE",
        "## RELATIONSHIP EVOLUTION\n- Alice ↔ Bob: Close friends.",
    )
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_checkpoint.return_value = {
        "job": {
            "config": {
                "prompt_options": {},
                "refine_after": False,
            },
            "progress": {"status": "completed"},
        },
        "chunks": [{
            "chunk_index": 0,
            "status": "completed",
            "original_text": "Source",
            "translated_text": "Translation",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(edited),
            },
        }],
    }
    checkpoint_manager.mark_refinement_stale.return_value = 7
    checkpoint_manager.db = MagicMock()

    state_manager = MagicMock()
    state_manager.checkpoint_manager = checkpoint_manager
    state_manager.get_translation.return_value = None

    thread = MagicMock()
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._claim_context_resync",
        lambda _translation_id: True,
    )
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._release_context_resync",
        lambda _translation_id: None,
    )
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes.threading.Thread",
        lambda **_kwargs: thread,
    )

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state_manager,
            lambda *_args, **_kwargs: None,
            str(tmp_path),
        )
    )

    with app.test_client() as client:
        response = client.post(
            "/api/translation/job/context/0/resync",
            json={"context_content": edited},
        )

    assert response.status_code == 200
    assert response.get_json()["context_revision"] == 7
    checkpoint_manager.mark_refinement_stale.assert_called_once_with("job")
    thread.start.assert_called_once()


def test_context_resync_route_accepts_failed_source_snapshot(
    monkeypatch,
    tmp_path,
):
    from src.api.blueprints.translation_routes import (
        create_translation_blueprint,
    )

    edited = build_novel_context(
        "# GLOBAL LORE",
        "## RELATIONSHIP EVOLUTION\n- Alice ↔ Bob: Close friends.",
    )
    checkpoint_manager = MagicMock()
    checkpoint_manager.load_checkpoint.return_value = {
        "job": {
            "config": {
                "prompt_options": {},
                "refine_after": False,
            },
            "progress": {"status": "partial"},
        },
        "chunks": [{
            "chunk_index": 2,
            "status": "failed",
            "original_text": "Source with useful context",
            "translated_text": "Source with useful context",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(edited),
            },
        }],
    }
    checkpoint_manager.mark_refinement_stale.return_value = 3
    checkpoint_manager.db = MagicMock()

    state_manager = MagicMock()
    state_manager.checkpoint_manager = checkpoint_manager
    state_manager.get_translation.return_value = None

    thread = MagicMock()
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._claim_context_resync",
        lambda _translation_id: True,
    )
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._release_context_resync",
        lambda _translation_id: None,
    )
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes.threading.Thread",
        lambda **_kwargs: thread,
    )

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state_manager,
            lambda *_args, **_kwargs: None,
            str(tmp_path),
        )
    )

    with app.test_client() as client:
        available = client.get("/api/translation/job/context/2")
        response = client.post(
            "/api/translation/job/context/2/resync",
            json={"context_content": edited},
        )

    assert available.status_code == 200
    assert available.get_json()["available_chunk_indices"] == [2]
    assert response.status_code == 200
    assert response.get_json()["context_revision"] == 3
    checkpoint_manager.db.save_chunk.assert_called_once()
    thread.start.assert_called_once()


def test_refinement_source_survives_completed_checkpoint_cleanup(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir()

    original = tmp_path / "novel.txt"
    original.write_text("Source novel", encoding="utf-8")
    translated = tmp_path / "translated.txt"
    translated.write_text("First-pass translation", encoding="utf-8")
    config = {
        "file_type": "txt",
        "file_path": str(original),
        "output_filename": translated.name,
        "refine_after": True,
    }

    assert manager.start_job(
        "job",
        "txt",
        config,
        str(original),
    )
    preserved_original = Path(config["preserved_input_path"])
    refinement_source = manager.preserve_refinement_source(
        "job",
        str(translated),
        config,
    )

    assert refinement_source is not None
    assert Path(refinement_source).read_text(
        encoding="utf-8"
    ) == "First-pass translation"
    assert manager.cleanup_completed_job("job")
    assert Path(refinement_source).is_file()
    assert not preserved_original.exists()

    persisted = manager.load_checkpoint("job")["job"]["config"]
    assert persisted["refinement_source_path"] == refinement_source
    assert persisted["refinement_stale"] is True

    assert manager.mark_refinement_current("job")
    persisted = manager.load_checkpoint("job")["job"]["config"]
    assert persisted["refinement_stale"] is False
    assert persisted["refinement_context_revision"] == 0

    assert manager.mark_refinement_stale("job") == 1
    persisted = manager.load_checkpoint("job")["job"]["config"]
    assert persisted["refinement_stale"] is True
    assert persisted["context_revision"] == 1


def test_context_resync_correction_replays_preserved_first_pass(tmp_path):
    from src.api.blueprints.translation_routes import (
        _build_corrective_refinement_config,
    )

    source = tmp_path / "first-pass.epub"
    source.write_bytes(b"draft")
    output = tmp_path / "final.epub"
    output.write_bytes(b"refined")
    config = {
        "refine_after": True,
        "refinement_source_path": str(source),
        "output_filepath": str(output),
        "output_filename": output.name,
        "refine_only": False,
    }

    correction = _build_corrective_refinement_config(config)

    assert correction is not None
    assert correction["file_path"] == str(source.resolve())
    assert correction["_force_output_filepath"] == str(output.resolve())
    assert correction["refine_only"] is True
    assert correction["refine_after"] is False
    assert correction["_context_resync_refinement"] is True
    assert config["refine_only"] is False


def test_context_resync_does_not_refine_an_already_refined_output(tmp_path):
    from src.api.blueprints.translation_routes import (
        _build_corrective_refinement_config,
    )

    output = tmp_path / "final.txt"
    output.write_text("Already refined", encoding="utf-8")

    assert _build_corrective_refinement_config({
        "refine_after": True,
        "output_filepath": str(output),
    }) is None


@pytest.mark.asyncio
async def test_early_refinement_unit_receives_final_lore_and_early_state():
    historical = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Kriha: Unspecified, Captain.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
        ),
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Kriha → Valentine: source form \"Major\" | formal\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Kriha → Valentine: Newly assigned subordinate."
        ),
    )
    final_context = build_novel_context(
        (
            "# GLOBAL LORE\n\n"
            "## CHARACTERS & GENDERS\n"
            "- Kriha: Female, Captain and loyal subordinate.\n\n"
            "## GLOSSARY & TERMINOLOGY\n"
            "- blood art: huyết thuật\n"
        ),
        "",
    )
    tracker = RefinementContextTracker(
        prompt_options={"novel_context": final_context},
        historical_contexts=[historical],
    )

    combined = await tracker.next_context(
        text="Draft chapter one.",
        llm_client=MagicMock(),
        model_name="model",
        target_language="English",
        display_index=1,
        total_chunks=1,
        scene_key=0,
    )

    assert "- Kriha: Female, Captain and loyal subordinate." in combined
    assert "- blood art: huyết thuật" in combined
    assert 'source form "Major" | formal' in combined
    assert "Newly assigned subordinate" in combined


@pytest.mark.asyncio
async def test_legacy_refinement_snapshots_recover_omitted_dormant_relationships():
    first_snapshot = build_novel_context(
        "# GLOBAL LORE",
        (
            "## CURRENT ADDRESSING FORMS\n"
            '- Alice → Bob: source form "Bob" | target-language form "anh" | intimate\n\n'
            "## RELATIONSHIP EVOLUTION\n"
            "- Alice ↔ Bob: Established romantic couple."
        ),
    )
    legacy_later_snapshot = build_novel_context(
        "# GLOBAL LORE",
        (
            "## CURRENT ADDRESSING FORMS\n"
            '- Guard → Captain: source form "Captain" | formal\n\n'
            "## RELATIONSHIP EVOLUTION\n"
            "- Guard → Captain: Temporary scene relationship."
        ),
    )
    tracker = RefinementContextTracker(
        prompt_options={
            "novel_context": build_novel_context(
                "# GLOBAL LORE",
                "## RELATIONSHIP EVOLUTION\n- Later → State: Must not leak backward.",
            )
        },
        historical_contexts=[
            first_snapshot,
            legacy_later_snapshot,
        ],
    )

    first = await tracker.next_context(
        text="First draft.",
        llm_client=MagicMock(),
        model_name="model",
        target_language="English",
        display_index=1,
        total_chunks=2,
    )
    second = await tracker.next_context(
        text="Second draft.",
        llm_client=MagicMock(),
        model_name="model",
        target_language="English",
        display_index=2,
        total_chunks=2,
    )

    assert "- Alice ↔ Bob: Established romantic couple." in first
    assert "Later → State" not in first
    assert "- Alice ↔ Bob: Established romantic couple." in second
    assert '- Alice → Bob: source form "Bob"' in second
    assert "- Guard → Captain: Temporary scene relationship." in second


@pytest.mark.asyncio
async def test_refinement_snapshot_explicit_delete_removes_durable_relationship():
    first_snapshot = build_novel_context(
        "# GLOBAL LORE",
        (
            "## CURRENT ADDRESSING FORMS\n"
            '- Alice → Bob: source form "Bob" | intimate\n\n'
            "## RELATIONSHIP EVOLUTION\n"
            "- Alice ↔ Bob: Established romantic couple."
        ),
    )
    deleted_snapshot = build_novel_context(
        "# GLOBAL LORE",
        (
            "## CURRENT ADDRESSING FORMS\n"
            "- Alice → Bob: DELETE\n\n"
            "## RELATIONSHIP EVOLUTION\n"
            "- Alice ↔ Bob: DELETE"
        ),
    )
    tracker = RefinementContextTracker(
        prompt_options={
            "novel_context": build_novel_context("# GLOBAL LORE", "")
        },
        historical_contexts=[first_snapshot, deleted_snapshot],
    )

    await tracker.next_context(
        text="First draft.",
        llm_client=MagicMock(),
        model_name="model",
        target_language="English",
        display_index=1,
        total_chunks=2,
    )
    second = await tracker.next_context(
        text="Second draft.",
        llm_client=MagicMock(),
        model_name="model",
        target_language="English",
        display_index=2,
        total_chunks=2,
    )

    assert "Alice → Bob" not in second
    assert "Alice ↔ Bob" not in second


@pytest.mark.asyncio
async def test_standalone_refinement_rebuilds_context_before_refining(monkeypatch):
    from src.core import translator

    events = []

    class FakeClient:
        async def generate(self, prompt, system_prompt=None):
            events.append("context")
            return LLMResponse(
                content=(
                    "[NEW_CHARACTERS]\n"
                    "- Mira: Female protagonist\n\n"
                    "[NEW_GLOSSARY]\n\n"
                    "[DYNAMIC_STATE]\n"
                    "Mira addresses Rowan formally."
                )
            )

        async def detect_thinking_model(self):
            return None

        async def close(self):
            return None

    async def fake_refine_request(**kwargs):
        events.append("refine")
        context = kwargs["context_content"]
        assert "Mira addresses Rowan formally." in context
        return "Polished text", LLMResponse(content="<TRANSLATION>Polished text</TRANSLATION>")

    monkeypatch.setattr(translator, "create_llm_client", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(translator, "_make_refinement_request", fake_refine_request)

    tracker = RefinementContextTracker(
        prompt_options={
            "auto_update_context": True,
            "novel_context": build_novel_context("# GLOBAL LORE", ""),
        },
        historical_contexts=[None],
    )
    result = await translator.refine_chunks(
        translated_chunks=["Mira greeted Rowan."],
        original_chunks=[{
            "context_before": "",
            "main_content": "Mira greeted Rowan.",
            "context_after": "",
        }],
        target_language="English",
        model_name="model",
        api_endpoint="http://localhost",
        prompt_options=tracker.prompt_options,
        context_tracker=tracker,
    )

    assert result == ["Polished text"]
    assert events == ["context", "refine"]


@pytest.mark.asyncio
async def test_txt_refine_after_reuses_exact_translation_units(monkeypatch, tmp_path):
    from src.core.refine import txt_refiner

    input_path = tmp_path / "translated.txt"
    output_path = tmp_path / "refined.txt"
    input_path.write_text("One\nTwo", encoding="utf-8")

    context_one = build_novel_context("LORE", "STATE ONE")
    context_two = build_novel_context("LORE", "STATE TWO")
    rows = [
        {
            "chunk_index": 0,
            "status": "completed",
            "translated_text": "One",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(context_one),
            },
        },
        {
            "chunk_index": 1,
            "status": "completed",
            "translated_text": "Two",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(context_two),
            },
        },
    ]
    checkpoint_manager = MagicMock()
    checkpoint_manager.db.get_chunks.return_value = rows

    monkeypatch.setattr(
        txt_refiner,
        "split_text_into_chunks",
        lambda *args, **kwargs: [{
            "context_before": "",
            "main_content": "One\nTwo",
            "context_after": "",
        }],
    )

    async def fake_refine_chunks(**kwargs):
        assert kwargs["translated_chunks"] == ["One", "Two"]
        assert len(kwargs["original_chunks"]) == 2
        assert kwargs["context_tracker"].historical_contexts == [
            context_one,
            context_two,
        ]
        return ["One refined", "Two refined"]

    monkeypatch.setattr(txt_refiner, "refine_chunks", fake_refine_chunks)

    success = await txt_refiner.refine_txt_file(
        input_filepath=str(input_path),
        output_filepath=str(output_path),
        target_language="English",
        model_name="model",
        checkpoint_manager=checkpoint_manager,
        translation_id="job",
        prompt_options={"novel_context": build_novel_context("LORE", "")},
    )

    assert success is True
    assert output_path.read_text(encoding="utf-8").startswith(
        "One refined\nTwo refined"
    )


@pytest.mark.asyncio
async def test_epub_refine_runs_without_stats_callback(monkeypatch, tmp_path):
    from src.core.refine import epub_refiner

    input_path = tmp_path / "input.epub"
    output_path = tmp_path / "output.epub"
    input_path.write_bytes(b"epub")

    client = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(
        epub_refiner,
        "build_refine_client",
        lambda **kwargs: (client, None),
    )
    monkeypatch.setattr(epub_refiner, "_extract_epub", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        epub_refiner,
        "_parse_epub_manifest",
        lambda *args, **kwargs: {"content_files": [], "opf_dir": str(tmp_path)},
    )

    def fake_repackage(*, output_filepath, **kwargs):
        Path(output_filepath).write_bytes(b"refined")

    monkeypatch.setattr(epub_refiner, "_repackage_epub", fake_repackage)

    success = await epub_refiner.refine_epub_file(
        input_filepath=str(input_path),
        output_filepath=str(output_path),
        target_language="English",
        model_name="model",
        stats_callback=None,
    )

    assert success is True
    assert output_path.read_bytes() == b"refined"


def test_web_refine_after_uses_one_backend_refinement_phase():
    project_root = Path(__file__).resolve().parents[2]
    batch_controller = (
        project_root / "src" / "web" / "static" / "js"
        / "translation" / "batch-controller.js"
    ).read_text(encoding="utf-8")
    handlers = (
        project_root / "src" / "api" / "handlers.py"
    ).read_text(encoding="utf-8")

    assert "refine: false" in batch_controller
    assert "translation_prompt_options['refine'] = False" in handlers
    assert "refine_success = await refine_file(" in handlers


def test_workflow_steps_and_resync_logs_use_the_canonical_ui_channel():
    project_root = Path(__file__).resolve().parents[2]
    tracker = (
        project_root / "src" / "web" / "static" / "js"
        / "translation" / "translation-tracker.js"
    ).read_text(encoding="utf-8")
    generic = (
        project_root / "src" / "core" / "adapters" / "generic_translator.py"
    ).read_text(encoding="utf-8")

    assert "MessageLogger.addStepLog" in tracker
    assert "ui_step === 'context_resync'" in tracker
    assert '"ui_step": "context_resync"' in generic
    assert "emit_update(" in generic


def test_context_resync_save_does_not_replace_latest_with_historical_snapshot():
    project_root = Path(__file__).resolve().parents[2]
    tracker = (
        project_root / "src" / "web" / "static" / "js"
        / "translation" / "translation-tracker.js"
    ).read_text(encoding="utf-8")

    save_block = tracker.split(
        "window.saveContextResync = async function()",
        1,
    )[1]
    assert "NovelContextUI.latestContent = newContent" not in save_block
    assert "NovelContextUI.renderContextTabs(newContent, true)" in save_block


def test_running_refinement_can_be_restored_after_browser_refresh():
    project_root = Path(__file__).resolve().parents[2]
    routes = (
        project_root / "src" / "api" / "blueprints" / "translation_routes.py"
    ).read_text(encoding="utf-8")
    tracker = (
        project_root / "src" / "web" / "static" / "js"
        / "translation" / "translation-tracker.js"
    ).read_text(encoding="utf-8")

    assert "'input_filename': (" in routes
    assert "job.input_filename || job.output_filename" in tracker
    assert "fileType: job.file_type || 'txt'" in tracker


def test_max_tokens_setting_is_runtime_reloadable(monkeypatch, tmp_path):
    import src.config as config

    env_path = tmp_path / ".env"
    env_path.write_text("MAX_TOKENS_PER_CHUNK=137\n", encoding="utf-8")
    old_value = config.MAX_TOKENS_PER_CHUNK

    monkeypatch.setattr(config, "_env_file", env_path)
    monkeypatch.setattr(config, "MAX_TOKENS_PER_CHUNK", old_value)
    monkeypatch.setenv("MAX_TOKENS_PER_CHUNK", str(old_value))

    config.reload_config()

    assert config.MAX_TOKENS_PER_CHUNK == 137


def test_web_jobs_use_and_clamp_live_chunk_budget(monkeypatch):
    import src.config as config
    from src.api.blueprints.translation_routes import _clamp_chunk_tokens

    monkeypatch.setattr(config, "MAX_TOKENS_PER_CHUNK", 137)

    assert _clamp_chunk_tokens(None) == 137
    assert _clamp_chunk_tokens(20) == 50
    assert _clamp_chunk_tokens(2000) == 2000
    assert _clamp_chunk_tokens(5000) == 5000
