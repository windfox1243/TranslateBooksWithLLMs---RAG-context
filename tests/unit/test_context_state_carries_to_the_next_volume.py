"""Structured context has to survive the move to the next job on the same book.

The markdown context file is per novel; every structured table is keyed by job.
Exporting those tables to markdown keeps the rendered line and nothing else, so
a rule the user locked in volume 1 used to arrive in volume 2 unlocked, at
default confidence, free for the model's first guess to overwrite.
"""
import pytest

from src.persistence.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "jobs.db"))
    try:
        yield database
    finally:
        database.close_all()


def make_job(db, translation_id, context_file):
    db.create_job(
        translation_id=translation_id,
        file_type="txt",
        config={"prompt_options": {"novel_context_file": context_file}},
    )


def seed_volume_one(db, translation_id="vol-1"):
    db.upsert_addressing_rule(
        translation_id=translation_id,
        speaker_name="Sora",
        addressee_name="Rin",
        self_pronoun="em",
        target_pronoun="cô",
        vocative="cô Rin",
        register="formal",
        social_basis=["senior"],
        confidence=0.95,
        is_locked=1,
        provenance="user",
    )
    source = db.upsert_relationship_node(
        translation_id=translation_id,
        canonical_name="Sora",
        normalized_name="sora",
        gender="male",
    )
    target = db.upsert_relationship_node(
        translation_id=translation_id,
        canonical_name="Rin",
        normalized_name="rin",
        gender="female",
    )
    db.upsert_relationship_edge(
        translation_id=translation_id,
        source_node_id=source,
        target_node_id=target,
        relationship_type="mentor",
        direction="directed",
        scope="durable",
        hierarchy="senior",
        intimacy="respectful",
        register="formal",
        confidence=0.9,
        status="accepted",
        is_locked=1,
        chunk_index=12,
        provenance="user",
    )
    return source, target


def test_the_previous_job_on_the_same_book_is_found(db):
    make_job(db, "vol-1", "saga.txt")
    make_job(db, "vol-2", "saga.txt")
    make_job(db, "other", "different-book.txt")

    assert db.find_previous_job_for_context_file(
        "saga.txt", exclude_translation_id="vol-2"
    ) == "vol-1"
    assert db.find_previous_job_for_context_file(
        "different-book.txt", exclude_translation_id="other"
    ) is None


def test_a_locked_rule_arrives_still_locked(db):
    make_job(db, "vol-1", "saga.txt")
    make_job(db, "vol-2", "saga.txt")
    seed_volume_one(db)

    counts = db.carry_over_structured_context("vol-2", "vol-1")

    assert counts["addressing_rules"] == 1
    carried = db.get_addressing_rules("vol-2")[0]
    assert carried["is_locked"] == 1
    assert carried["target_pronoun"] == "cô"
    assert carried["confidence"] == pytest.approx(0.95)
    assert carried["provenance"] == "user"


def test_relationship_edges_point_at_the_new_job_s_own_nodes(db):
    make_job(db, "vol-1", "saga.txt")
    make_job(db, "vol-2", "saga.txt")
    old_source, old_target = seed_volume_one(db)

    db.carry_over_structured_context("vol-2", "vol-1")

    nodes = {node["canonical_name"]: node["id"] for node in db.get_relationship_nodes("vol-2")}
    assert set(nodes) == {"Sora", "Rin"}
    assert set(nodes.values()).isdisjoint({old_source, old_target})

    edges = db.get_relationship_edges("vol-2")
    assert len(edges) == 1
    assert edges[0]["source_node_id"] == nodes["Sora"]
    assert edges[0]["target_node_id"] == nodes["Rin"]
    assert edges[0]["is_locked"] == 1
    assert db.get_relationship_edges("vol-1")[0]["source_node_id"] == old_source


def test_a_job_that_already_has_state_is_left_alone(db):
    make_job(db, "vol-1", "saga.txt")
    make_job(db, "vol-2", "saga.txt")
    seed_volume_one(db)
    db.upsert_addressing_rule(
        translation_id="vol-2",
        speaker_name="Sora",
        addressee_name="Rin",
        self_pronoun="tôi",
        target_pronoun="bạn",
        register="neutral",
    )

    assert db.carry_over_structured_context("vol-2", "vol-1") == {}
    assert db.get_addressing_rules("vol-2")[0]["target_pronoun"] == "bạn"


def test_carrying_over_to_the_same_job_does_nothing(db):
    make_job(db, "vol-1", "saga.txt")
    seed_volume_one(db)

    assert db.carry_over_structured_context("vol-1", "vol-1") == {}
    assert len(db.get_addressing_rules("vol-1")) == 1
