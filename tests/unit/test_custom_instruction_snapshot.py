"""The custom-instruction preset is read once per job (ported from upstream f76b64e1)."""

import src.config as config_module
from src.api.handlers import (
    CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG,
    snapshot_custom_instructions,
)


def test_preset_text_is_frozen_when_the_job_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "CUSTOM_INSTRUCTIONS_DIR", tmp_path)
    preset = tmp_path / "style.txt"
    preset.write_text("Keep honorifics.", encoding="utf-8")
    config = {"prompt_options": {"custom_instruction_file": "style.txt"}}
    logs = []

    snapshot_custom_instructions(config, lambda *args: logs.append(args))
    options = config["prompt_options"]
    assert options["custom_instructions"] == "Keep honorifics."
    assert options[CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG] is True

    # Editing (or deleting) the preset mid-book must not change the job.
    preset.write_text("Drop honorifics.", encoding="utf-8")
    snapshot_custom_instructions(config, lambda *args: logs.append(args))
    assert options["custom_instructions"] == "Keep honorifics."


def test_job_without_a_preset_is_not_flagged():
    config = {"prompt_options": {}}
    snapshot_custom_instructions(config, lambda *args: None)
    assert CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG not in config["prompt_options"]
