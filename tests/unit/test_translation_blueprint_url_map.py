"""Pin the URL map produced by create_translation_blueprint.

The blueprint factory was a single 2,879-line function registering 60 routes.
It is now assembled from per-domain registration modules. Splitting it must not
move, rename or drop a single rule: this test snapshots every (rule, endpoint,
methods) triple and fails on any drift, including changes to the dual
"/api/translation/<id>/..." and "/<id>/..." registrations.

When a route is deliberately added, add its triple to EXPECTED_RULES in the
same commit.
"""

import json
import pathlib

import pytest
from flask import Flask

from src.api.blueprints.translation_routes import create_translation_blueprint

SNAPSHOT = pathlib.Path(__file__).with_name("translation_blueprint_url_map.json")


class _FakeDatabase:
    def reconcile_interrupted_operations(self):
        return None


class _FakeCheckpointManager:
    def __init__(self):
        self.db = _FakeDatabase()


class _FakeStateManager:
    def __init__(self):
        self.checkpoint_manager = _FakeCheckpointManager()


def _build_url_map(tmp_path):
    app = Flask(__name__)
    blueprint = create_translation_blueprint(
        state_manager=_FakeStateManager(),
        start_translation_job=lambda *a, **kw: None,
        output_dir=str(tmp_path),
        socketio=None,
    )
    app.register_blueprint(blueprint)
    return sorted(
        [rule.rule, rule.endpoint, sorted(rule.methods)]
        for rule in app.url_map.iter_rules()
        if rule.endpoint != "static"
    )


def test_url_map_matches_snapshot(tmp_path):
    actual = _build_url_map(tmp_path)

    if not SNAPSHOT.exists():  # pragma: no cover - first run only
        SNAPSHOT.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        pytest.skip("wrote initial url_map snapshot")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    actual_rules = {(r[0], r[1]) for r in actual}
    expected_rules = {(r[0], r[1]) for r in expected}

    missing = sorted(expected_rules - actual_rules)
    added = sorted(actual_rules - expected_rules)

    assert not missing, f"routes disappeared: {missing}"
    assert not added, f"routes appeared without updating the snapshot: {added}"
    assert actual == expected, "route methods changed"
