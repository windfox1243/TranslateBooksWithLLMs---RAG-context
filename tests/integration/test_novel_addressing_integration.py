"""
Integration test for directed addressing context engine on sample novel text from drive F.
"""

import os
import tempfile
import pytest
from src.persistence.database import Database
from src.utils.context_schema import extract_addressing_deltas_from_text, AddressingUpdateDelta
from src.utils.context_merge_engine import ContextMergeEngine
from src.utils.context_projection import render_addressing_projection, render_addressing_markdown


def test_novel_addressing_integration_with_drive_f():
    novel_path = r"F:\I Want to Be a Fluffy, Airheaded Gray Umamusume and Trick My Trainer!.txt"
    if not os.path.exists(novel_path):
        pytest.skip("Drive F sample novel file not found.")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "novel_test.db")
        db = Database(db_path=db_path)
        engine = ContextMergeEngine(db=db, confidence_threshold=0.80)
        tx_id = "tx_drive_f_test"

        try:
            # Simulate processing 2 dialogue chunks from novel
            simulated_llm_output_chunk1 = """
            Translation text chunk 1...

            ```json
            {
              "addressing_updates": [
                {
                  "speaker": "Trainer",
                  "addressee": "Oguri",
                  "self_pronoun": "tôi",
                  "second_pronoun": "em",
                  "vocative": "Oguri",
                  "register": "polite",
                  "confidence": 0.92,
                  "evidence_quote": "Nào Oguri, ăn bớt đi chứ em."
                },
                {
                  "speaker": "Oguri",
                  "addressee": "Trainer",
                  "self_pronoun": "em",
                  "second_pronoun": "thầy",
                  "vocative": "thầy",
                  "register": "polite",
                  "confidence": 0.90,
                  "evidence_quote": "Thầy ơi em vẫn đói."
                }
              ]
            }
            ```
            """

            deltas1 = extract_addressing_deltas_from_text(simulated_llm_output_chunk1)
            assert len(deltas1) == 2

            applied_count1 = engine.apply_batch_deltas(tx_id, chunk_index=0, deltas=deltas1)
            assert applied_count1 == 2

            # Check DB directed rules
            rules = db.get_addressing_rules(tx_id)
            assert len(rules) == 2

            rule_trainer_oguri = next(r for r in rules if r["speaker_name"] == "Trainer")
            assert rule_trainer_oguri["self_pronoun"] == "tôi"
            assert rule_trainer_oguri["target_pronoun"] == "em"

            rule_oguri_trainer = next(r for r in rules if r["speaker_name"] == "Oguri")
            assert rule_oguri_trainer["self_pronoun"] == "em"
            assert rule_oguri_trainer["target_pronoun"] == "thầy"

            # Check rendered projection for next chunk system prompt
            projection = render_addressing_projection(tx_id, db=db)
            assert "**Trainer** khi nói với **Oguri**: Tự xưng là 'tôi', gọi đối phương là 'em'" in projection
            assert "**Oguri** khi nói với **Trainer**: Tự xưng là 'em', gọi đối phương là 'thầy'" in projection

            # Check audit logs
            logs = db.get_context_audit_logs(tx_id)
            assert len(logs) == 2

        finally:
            db.close()
