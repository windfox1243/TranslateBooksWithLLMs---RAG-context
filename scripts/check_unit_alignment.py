"""Check the unit aligner against real finished translations.

Usage:

    python scripts/check_unit_alignment.py --db path/to/jobs.db
    python scripts/check_unit_alignment.py --db path/to/jobs.db --job trans_123

The unit-rewrite editor rests on one claim: source and draft can be split into
addressable units deterministically, and a rewritten unit can be turned back
into exact draft spans without anything being quoted or searched for. That
claim is cheap to test and expensive to be wrong about, so this replays it over
whatever a real job already produced -- no provider, no key, no cost.

Reported per job: how units were paired, how many source paragraphs the draft
never produced, and two integrity checks that must both come out at zero
failures -- that a unit's recorded offsets reproduce its text, and that a
rewrite applied through the computed spans lands exactly where it should.

Exit status is 1 when an integrity check fails or a chunk cannot be aligned.
The pairing rates are reported, never asserted: how often a translator merges
two paragraphs is a property of the book, not a regression.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.editor.units import align_units, unit_rewrite_patches  # noqa: E402

# Applied to every unit to exercise the span arithmetic. The text is irrelevant
# -- what is being checked is that the spans land where the rewrite says.
_PROBE_SUFFIX = " [probe]"


def _chunks(connection: sqlite3.Connection, job: str | None) -> list:
    query = (
        "SELECT translation_id, chunk_index, original_text, translated_text "
        "FROM checkpoint_chunks WHERE translated_text IS NOT NULL "
        "AND original_text IS NOT NULL"
    )
    parameters: tuple = ()
    if job:
        query += " AND translation_id = ?"
        parameters = (job,)
    return [dict(row) for row in connection.execute(
        query + " ORDER BY translation_id, chunk_index", parameters,
    )]


def _apply(draft: str, patches) -> str:
    for start, end, replacement, _ in sorted(patches, reverse=True):
        draft = draft[:start] + replacement + draft[end:]
    return draft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to a jobs.db")
    parser.add_argument("--job", help="one translation_id (default: all)")
    arguments = parser.parse_args()

    path = Path(arguments.db)
    if not path.exists():
        print(f"no such database: {path}")
        return 1

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = _chunks(connection, arguments.job)
    finally:
        connection.close()

    if not rows:
        print("no finished chunks found")
        return 1

    by_job: dict = collections.defaultdict(
        lambda: {
            "chunks": 0, "units": 0, "moves": collections.Counter(),
            "unaligned": 0, "offset_failures": 0, "span_failures": 0,
            "rewrites": 0,
        }
    )
    for row in rows:
        report = by_job[row["translation_id"]]
        report["chunks"] += 1
        draft = row["translated_text"]
        units = align_units(row["original_text"], draft)
        if not units:
            report["unaligned"] += 1
            continue
        for unit in units:
            report["units"] += 1
            report["moves"][f"{unit.source_blocks}-{unit.draft_blocks}"] += 1
            if unit.draft_blocks == 1 and draft[
                unit.draft_start:unit.draft_end
            ] != unit.draft:
                report["offset_failures"] += 1
            if unit.is_omission or unit.is_addition:
                continue
            rewritten = unit.draft + _PROBE_SUFFIX
            patched = _apply(draft, unit_rewrite_patches(unit, rewritten, "probe"))
            expected = draft[:unit.draft_start] + rewritten + draft[unit.draft_end:]
            report["rewrites"] += 1
            if patched != expected:
                report["span_failures"] += 1

    failed = False
    for job, report in by_job.items():
        units = max(1, report["units"])
        paired = report["moves"].get("1-1", 0)
        omitted = sum(
            count for move, count in report["moves"].items()
            if move.endswith("-0")
        )
        print(f"{job}: {report['chunks']} chunks, {report['units']} units")
        print(f"  one to one     {paired} ({100 * paired / units:.1f}%)")
        print(f"  merges/splits  {units - paired - omitted}")
        print(f"  unpaired       {omitted}")
        print(f"  rewrites       {report['rewrites']}")
        print(
            f"  failures       {report['unaligned']} unaligned, "
            f"{report['offset_failures']} offset, {report['span_failures']} span"
        )
        if (
            report["unaligned"]
            or report["offset_failures"]
            or report["span_failures"]
        ):
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
