"""Deleting a job and the lore file it leaves behind.

Deleting a job already wipes every structured context row it owns. The novel
context text file was the one piece that survived, so the next run of the same
book read stale lore with none of the structure that backed it. These pin that
the file can now go with the job, and -- more importantly -- the three cases
where it must not.
"""

import pytest

import src.config as config
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def contexts_dir(tmp_path, monkeypatch):
    directory = tmp_path / "Novel_Contexts"
    directory.mkdir()
    monkeypatch.setattr(config, "NOVEL_CONTEXTS_DIR", directory)
    return directory


@pytest.fixture
def manager(tmp_path):
    checkpoint_manager = CheckpointManager(
        db_path=str(tmp_path / "jobs.db"), server_session_id="session_1"
    )
    yield checkpoint_manager
    checkpoint_manager.close()


def _start(manager, translation_id, context_file):
    manager.start_job(
        translation_id,
        "txt",
        {"prompt_options": {"novel_context_file": context_file}},
    )


def test_the_lore_file_stays_unless_deletion_is_asked_for(manager, contexts_dir):
    lore = contexts_dir / "book_Vietnamese_context.txt"
    lore.write_text("Hanako is the trainer.", encoding="utf-8")
    _start(manager, "trans_1", lore.name)

    result = manager.delete_checkpoint("trans_1")

    assert result.deleted is True
    assert result.novel_context_removed is None
    assert lore.exists()


def test_the_lore_file_goes_with_the_job_when_asked(manager, contexts_dir):
    lore = contexts_dir / "book_Vietnamese_context.txt"
    lore.write_text("Hanako is the trainer.", encoding="utf-8")
    _start(manager, "trans_1", lore.name)

    result = manager.delete_checkpoint("trans_1", delete_novel_context=True)

    assert result.novel_context_removed == lore.name
    assert result.novel_context_kept_for is None
    assert not lore.exists()


def test_another_job_using_the_same_file_keeps_it_alive(manager, contexts_dir):
    # The name is built from book plus target language, not from the job id, so
    # a continuation or a re-run of the same book points at the same file.
    lore = contexts_dir / "book_Vietnamese_context.txt"
    lore.write_text("Hanako is the trainer.", encoding="utf-8")
    _start(manager, "trans_1", lore.name)
    _start(manager, "trans_2", lore.name)

    result = manager.delete_checkpoint("trans_1", delete_novel_context=True)

    assert result.novel_context_removed is None
    assert result.novel_context_kept_for == "trans_2"
    assert lore.exists()


def test_a_path_pointing_outside_the_directory_is_not_followed(
    manager, contexts_dir, tmp_path
):
    # Checkpoints carry absolute paths written by other installs, and the read
    # path deliberately redirects those back into the local directory. Deletion
    # must not inherit that leniency.
    outsider = tmp_path / "somewhere_else.txt"
    outsider.write_text("not ours to delete", encoding="utf-8")
    _start(manager, "trans_1", str(outsider))

    result = manager.delete_checkpoint("trans_1", delete_novel_context=True)

    assert result.deleted is True
    assert result.novel_context_removed is None
    assert outsider.exists()


def test_a_traversing_name_resolves_to_nothing(manager, contexts_dir, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("not ours to delete", encoding="utf-8")
    _start(manager, "trans_1", "../victim.txt")

    result = manager.delete_checkpoint("trans_1", delete_novel_context=True)

    assert result.novel_context_removed is None
    assert victim.exists()


def test_a_job_without_a_context_file_deletes_quietly(manager, contexts_dir):
    manager.start_job("trans_1", "txt", {"prompt_options": {}})

    result = manager.delete_checkpoint("trans_1", delete_novel_context=True)

    assert result.deleted is True
    assert result.novel_context_removed is None


def test_the_result_still_reads_as_the_boolean_it_replaced(manager, contexts_dir):
    # Deletion is idempotent: an id that is already gone still reports success,
    # which is why the endpoint's 404 branch never fires. Pinned as it stands so
    # the return type change is not blamed for it later.
    _start(manager, "trans_1", "")

    assert bool(manager.delete_checkpoint("trans_1")) is True
    assert bool(manager.delete_checkpoint("trans_missing")) is True
