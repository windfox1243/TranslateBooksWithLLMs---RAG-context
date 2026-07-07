"""
Simulation test for UniversalAddressingEngine on real novel text snippets.
Reads sample dialogue lines from F:\\I Want to Be a Fluffy, Airheaded Gray Umamusume and Trick My Trainer!.txt
and verifies addressing pair repairs.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.universal_addressing_engine import UniversalAddressingEngine


def test_novel_snippets():
    engine = UniversalAddressingEngine(language="vi")

    # Sample extracted addressing pairs from novel context
    test_pairs = [
        # Trainer - Trainee pair from Umamusume novel
        {
            "speaker": "Apollo Rainbow",
            "addressee": "Tomio Momozawa",
            "self": "tôi",
            "target": "cậu",
            "vocative": "Trainer",
            "context": "professional mentor/trainee relationship; trainer/student hierarchy",
        },
        {
            "speaker": "Tomio Momozawa",
            "addressee": "Apollo Rainbow",
            "self": "tôi",
            "target": "em",
            "vocative": "Apollo",
            "context": "professional mentor/trainee relationship",
        },
        # Peer classmate pair
        {
            "speaker": "Oguri Cap",
            "addressee": "Tamamo Cross",
            "self": "tớ",
            "target": "mày",
            "vocative": "Tama",
            "context": "friendly classmates",
        },
        # Hostile opponent pair
        {
            "speaker": "Demon King",
            "addressee": "Hero",
            "self": "tao",
            "target": "ngài",
            "vocative": "Hero",
            "context": "hostile enemies",
        },
    ]

    print("=== NOVEL ADDRESSING ENGINE SIMULATION RESULTS ===")
    for idx, pair in enumerate(test_pairs, 1):
        s_rep, t_rep, v_rep = engine.validate_and_repair_pair(
            self_pronoun=pair["self"],
            target_pronoun=pair["target"],
            speaker=pair["speaker"],
            addressee=pair["addressee"],
            vocative=pair["vocative"],
            details_context=pair["context"],
        )
        print(f"Test {idx}: {pair['speaker']} → {pair['addressee']}")
        print(f"  Input : Self='{pair['self']}', Target='{pair['target']}', Vocative='{pair['vocative']}'")
        print(f"  Result: Self='{s_rep}', Target='{t_rep}', Vocative='{v_rep}'")
        print("-" * 50)


if __name__ == "__main__":
    test_novel_snippets()
