"""A locked addressing rule has to reach the refine pass.

Refinement replays each chunk's stored context snapshot and reads nothing else.
The structured-context export used to rewrite only the newest snapshot, so a
rule the user locked in the UI applied to the last chunk and no other -- the
refine pass went on using the address forms the model had guessed.
"""
from types import SimpleNamespace

import pytest

from src.utils.db_addressing import overlay_locked_addressing_on_context
from src.utils.novel_context import (
    build_novel_context,
    compress_dynamic_state,
    extract_dynamic_state_from_text,
)

GLOBAL_LORE = """# GLOBAL LORE

## CHARACTERS & GENDERS
- Rin: Female, swordswoman.
- Sora: Male, apprentice.
"""

GUESSED = '- Sora → Rin: source form "senpai" | recommended target-language form "chị" | polite'
LOCKED = '- Sora → Rin: source form "senpai" | recommended target-language form "cô" | formal'
UNRELATED = '- Rin → Sora: source form "kun" | recommended target-language form "em" | casual'


def rule(speaker, addressee, pronoun, register, locked):
    return {
        "speaker_name": speaker,
        "addressee_name": addressee,
        "self_pronoun": "",
        "target_pronoun": pronoun,
        "vocative": "",
        "register": register,
        "source_forms": ["senpai"],
        "social_basis": [],
        "is_locked": 1 if locked else 0,
    }


class FakeDb:
    def __init__(self, rules):
        self._rules = rules

    def get_addressing_rules(self, _translation_id, validation_status="active"):
        return list(self._rules)


def snapshot_context(addressing_lines):
    dynamic = "### CURRENT ADDRESSING FORMS\n" + "\n".join(addressing_lines)
    return build_novel_context(GLOBAL_LORE, dynamic)


def test_a_locked_rule_replaces_the_guessed_line():
    db = FakeDb([rule("Sora", "Rin", "cô", "formal", locked=True)])

    updated = overlay_locked_addressing_on_context(
        snapshot_context([GUESSED]), "job-1", db
    )

    assert '"cô"' in updated
    assert '"chị"' not in updated


def test_an_unlocked_rule_leaves_the_snapshot_alone():
    db = FakeDb([rule("Sora", "Rin", "cô", "formal", locked=False)])
    original = snapshot_context([GUESSED])

    assert overlay_locked_addressing_on_context(original, "job-1", db) == original


def test_lines_for_other_pairs_survive_the_overlay():
    db = FakeDb([rule("Sora", "Rin", "cô", "formal", locked=True)])

    updated = overlay_locked_addressing_on_context(
        snapshot_context([GUESSED, UNRELATED]), "job-1", db
    )

    assert "Rin → Sora" in updated
    assert '"em"' in updated


def test_a_locked_rule_the_snapshot_never_had_is_added():
    db = FakeDb([rule("Sora", "Rin", "cô", "formal", locked=True)])

    updated = overlay_locked_addressing_on_context(
        snapshot_context([UNRELATED]), "job-1", db
    )

    dynamic = extract_dynamic_state_from_text(updated) or ""
    assert "Sora → Rin" in dynamic
    assert "Rin → Sora" in dynamic


def test_no_locked_rules_means_no_rewrite_at_all():
    original = snapshot_context([GUESSED])

    assert overlay_locked_addressing_on_context(original, "job-1", FakeDb([])) == original
    assert overlay_locked_addressing_on_context(original, "job-1", None) == original


class RecordingDb(FakeDb):
    """Enough of the Database surface for the structured-context export."""

    def __init__(self, rules, chunks):
        super().__init__(rules)
        self.chunks = chunks
        self.saved = {}

    def get_job(self, _translation_id):
        return {"config": {"prompt_options": {"novel_context_file": "book.txt"}}}

    def get_chunks(self, _translation_id):
        return [dict(chunk) for chunk in self.chunks]

    def get_relationship_edges(self, _translation_id, statuses=None):
        return []

    def get_relationship_edges_as_of(
        self, _translation_id, _chunk_index, statuses=None
    ):
        return []

    def save_chunk(self, translation_id, chunk_index, chunk_data, **_kwargs):
        self.saved[chunk_index] = chunk_data


@pytest.fixture
def contexts_dir(tmp_path, monkeypatch):
    directory = tmp_path / "Novel_Contexts"
    directory.mkdir()
    (directory / "book.txt").write_text(
        snapshot_context([GUESSED]), encoding="utf-8"
    )
    import src.config as config

    monkeypatch.setattr(config, "NOVEL_CONTEXTS_DIR", directory)
    return directory


def test_the_export_rewrites_every_snapshot_not_just_the_newest(contexts_dir):
    from src.api.blueprints.translation_routes.shared import build_shared

    chunks = [
        {
            "chunk_index": index,
            "original_text": "",
            "translated_text": "",
            "status": "completed",
            "chunk_data": {
                "context_snapshot": compress_dynamic_state(
                    snapshot_context([GUESSED])
                )
            },
        }
        for index in range(3)
    ]
    db = RecordingDb([rule("Sora", "Rin", "cô", "formal", locked=True)], chunks)
    deps = SimpleNamespace(
        state_manager=SimpleNamespace(
            checkpoint_manager=SimpleNamespace(db=db)
        ),
        start_translation_job=None,
        socketio=None,
    )

    assert build_shared(deps).export_structured_context("job-1") is True
    assert sorted(db.saved) == [0, 1, 2]
