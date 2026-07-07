"""
Integration test reading dialogue lines directly from novel file on F drive.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.universal_addressing_engine import UniversalAddressingEngine

NOVEL_PATH = r"F:\I Want to Be a Fluffy, Airheaded Gray Umamusume and Trick My Trainer!.txt"


def test_real_novel_dialogue():
    if not os.path.exists(NOVEL_PATH):
        print(f"File not found: {NOVEL_PATH}")
        return

    print(f"Reading sample dialogue from: {os.path.basename(NOVEL_PATH)}")
    engine = UniversalAddressingEngine(language="vi")

    with open(NOVEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()][:50]

    print(f"Loaded {len(lines)} lines of text for novel context verification.")
    print("Testing UniversalAddressingEngine formality distance calculations on target novel addressing pairs:")

    pairs = [
        ("tôi", "Trainer", "Trainer"),
        ("tôi", "em", "Apollo"),
        ("tớ", "mày", "Tama"),
        ("tao", "ngài", "Lord"),
        ("ta", "ngươi", "Mortal"),
    ]

    for s, t, v in pairs:
        s_r, t_r, v_r = engine.validate_and_repair_pair(s, t, vocative=v)
        print(f"Pair: ({s}, {t}, {v}) → Repaired: ({s_r}, {t_r}, {v_r})")


if __name__ == "__main__":
    test_real_novel_dialogue()
