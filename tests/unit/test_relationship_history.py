"""A relationship edge holds one state; a book needs the ones before it too.

Two things used to leak the end of the book into its middle. An edge's mutable
side -- details, register, status -- was overwritten in place, so chapter 5 was
re-exported with wording only chapter 40 had earned. And an edge born at chapter
40 (a new relationship_type is a new row, not an update) was exported into every
snapshot, including the chapters where those two were still strangers.
"""
import pytest

from src.persistence.database import Database
from src.utils.relationship_sync import export_relationship_graph_to_markdown


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "jobs.db"))
    try:
        yield database
    finally:
        database.close_all()


@pytest.fixture
def pair(db):
    db.create_job(
        translation_id="job-1",
        file_type="txt",
        config={"prompt_options": {"novel_context_file": "saga.txt"}},
    )
    source = db.upsert_relationship_node(
        translation_id="job-1",
        canonical_name="Sora",
        normalized_name="sora",
        gender="male",
    )
    target = db.upsert_relationship_node(
        translation_id="job-1",
        canonical_name="Rin",
        normalized_name="rin",
        gender="female",
    )
    return source, target


def record(db, pair, relationship_type, chunk_index, details=""):
    source, target = pair
    return db.upsert_relationship_edge(
        translation_id="job-1",
        source_node_id=source,
        target_node_id=target,
        relationship_type=relationship_type,
        direction="symmetric",
        scope="durable",
        hierarchy="peer",
        intimacy="unknown",
        register="neutral",
        confidence=0.9,
        status="accepted",
        is_locked=0,
        chunk_index=chunk_index,
        provenance="model",
        details=details,
    )


def test_only_an_actual_change_is_recorded(db, pair):
    edge_id = record(db, pair, "enemy", 0, details="sworn rivals")
    record(db, pair, "enemy", 10, details="sworn rivals")
    record(db, pair, "enemy", 20, details="sworn rivals")
    record(db, pair, "enemy", 40, details="an uneasy truce")

    history = db.get_relationship_edge_history("job-1", edge_id)

    assert [(row["from_chunk_index"], row["details"]) for row in history] == [
        (0, "sworn rivals"),
        (40, "an uneasy truce"),
    ]


def test_the_state_at_a_chunk_is_the_one_in_force_then(db, pair):
    record(db, pair, "enemy", 0, details="sworn rivals")
    record(db, pair, "enemy", 40, details="an uneasy truce")

    def wording(chunk):
        return [
            edge["details"] for edge in db.get_relationship_edges_as_of("job-1", chunk)
        ]

    assert wording(0) == ["sworn rivals"]
    assert wording(39) == ["sworn rivals"]
    assert wording(40) == ["an uneasy truce"]
    assert wording(500) == ["an uneasy truce"]


def test_an_edge_that_did_not_exist_yet_is_not_back_dated(db, pair):
    record(db, pair, "ally", 40)

    assert db.get_relationship_edges_as_of("job-1", 5) == []
    assert len(db.get_relationship_edges_as_of("job-1", 40)) == 1


def test_a_relationship_that_replaces_another_leaves_the_first_in_place(db, pair):
    # A different relationship_type is a new row rather than an update, so both
    # edges exist afterwards -- but only the older one existed at chapter 5.
    record(db, pair, "enemy", 0, details="sworn rivals")
    record(db, pair, "ally", 40, details="fought side by side")

    assert [
        edge["relationship_type"]
        for edge in db.get_relationship_edges_as_of("job-1", 10)
    ] == ["enemy"]
    assert sorted(
        edge["relationship_type"]
        for edge in db.get_relationship_edges_as_of("job-1", 60)
    ) == ["ally", "enemy"]


def test_names_and_details_travel_with_the_recorded_state(db, pair):
    record(db, pair, "enemy", 0, details="sworn rivals")
    record(db, pair, "enemy", 40, details="an uneasy truce")

    early = db.get_relationship_edges_as_of("job-1", 10)[0]

    assert early["details"] == "sworn rivals"
    assert early["source_name"] == "Sora"
    assert early["target_name"] == "Rin"


def test_the_markdown_export_can_ask_for_an_earlier_era(db, pair):
    record(db, pair, "enemy", 0, details="sworn rivals")
    record(db, pair, "enemy", 40, details="an uneasy truce")

    early = export_relationship_graph_to_markdown("job-1", db, as_of_chunk=10)
    late = export_relationship_graph_to_markdown("job-1", db, as_of_chunk=60)

    assert "sworn rivals" in early and "uneasy truce" not in early
    assert "an uneasy truce" in late
    # No as_of_chunk still means "as it stands now".
    assert "an uneasy truce" in export_relationship_graph_to_markdown("job-1", db)


def test_a_status_filter_still_applies_to_historical_state(db, pair):
    source, target = pair
    record(db, pair, "enemy", 0)
    db.upsert_relationship_edge(
        translation_id="job-1",
        source_node_id=source,
        target_node_id=target,
        relationship_type="enemy",
        direction="symmetric",
        scope="durable",
        hierarchy="peer",
        intimacy="unknown",
        register="neutral",
        confidence=0.4,
        status="pending",
        is_locked=0,
        chunk_index=40,
        provenance="model",
    )

    assert db.get_relationship_edges_as_of("job-1", 60, statuses=["accepted"]) == []
    assert len(db.get_relationship_edges_as_of("job-1", 60)) == 1


def test_history_survives_the_move_to_the_next_volume(db, pair):
    db.create_job(
        translation_id="job-2",
        file_type="txt",
        config={"prompt_options": {"novel_context_file": "saga.txt"}},
    )
    record(db, pair, "enemy", 0)
    record(db, pair, "ally", 40)

    counts = db.carry_over_structured_context("job-2", "job-1")

    assert counts["relationship_history"] == 2
    carried = db.get_relationship_edges_as_of("job-2", 10)
    assert [edge["relationship_type"] for edge in carried] == ["enemy"]
