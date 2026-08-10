"""Score a set of produced translations against a regression corpus.

Usage:

    python scripts/run_quality_regression.py \
        --corpus tests/fixtures/quality_regression/en_vi_baseline.jsonl \
        --outputs my_run.jsonl \
        --baseline quality_baseline.json

`--outputs` is a JSONL file of `{"case_id": ..., "translation": ...}` lines,
produced however you like: a live run of the pipeline, a replay of a finished
job, or a hand-edited file. Keeping production out of this script is deliberate
-- scoring must stay runnable offline, with no provider, no key, and no cost,
so that a baseline can be re-checked long after the run that produced it.

Exit status is 1 when anything regressed, so this can gate a release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.quality.regression import (  # noqa: E402
    DEFAULT_TOLERANCE,
    compare_to_baseline,
    load_cases,
    run_cases,
)


def _load_outputs(path: Path) -> dict:
    translations = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        payload = json.loads(stripped)
        case_id = str(payload.get("case_id") or "")
        if not case_id:
            raise ValueError(f"{path}:{number}: output is missing 'case_id'")
        translations[case_id] = str(payload.get("translation") or "")
    return translations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--save-baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    cases = load_cases(args.corpus)
    translations = _load_outputs(args.outputs)
    missing = [case.case_id for case in cases if case.case_id not in translations]
    if missing:
        # A silently absent case would score as an empty translation and look
        # like a quality collapse, which is a worse lie than refusing to run.
        print(f"No translation for {len(missing)} case(s): {', '.join(missing)}")
        return 2

    scorecard = run_cases(cases, lambda case: translations[case.case_id])

    for result in sorted(scorecard.results, key=lambda item: item.chrf):
        mark = "ok  " if result.passed else "FAIL"
        print(f"{mark} {result.chrf:.4f}  {result.case_id}")
        for failure in result.failures:
            print(f"       {failure}")
    print(f"\nmean chrF {scorecard.mean_chrf:.4f} over {len(cases)} case(s)")

    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    regressions = compare_to_baseline(
        scorecard, baseline, tolerance=args.tolerance
    )
    if regressions:
        print("\nRegressions:")
        for line in regressions:
            print(f"  - {line}")

    if args.save_baseline:
        args.save_baseline.write_text(
            json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nBaseline written to {args.save_baseline}")

    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
