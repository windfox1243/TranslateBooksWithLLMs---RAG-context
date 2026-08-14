"""Turn unit-rewrite findings into draft patches, or refuse them.

The span contract fails at locating a repair; this one cannot, because the
location is an id we issued. What it can fail at is scope. Asked to return a
whole unit, a model will tidy the sentence next to the one it reported, and the
old contract had no way to notice -- a span edit only ever showed the span it
admitted to. Here the whole unit comes back, so the size of the edit is
measurable, and an edit far larger than the defect reported is refused rather
than applied and regretted.

A refused rewrite is not a lost finding. It leaves as an unresolved issue with
a reason attached, which is what the existing repair pass already consumes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.editor.units import (
    Unit,
    align_units,
    rewrite_change_ratio,
    unit_rewrite_patches,
)

# How much of a unit one repair may touch. A real local repair changes a term,
# a pronoun or a clause -- well under half the unit -- while a model that has
# decided to retranslate the paragraph changes nearly all of it. Set where those
# two populations separate rather than where either one ends, because the cost
# of the two mistakes is not symmetric: a refused repair is retried by the
# existing pass, an accepted rewrite silently replaces wording nobody reviewed.
MAX_REWRITE_CHANGE_RATIO = 0.5

UNIT_REPAIR_KIND = "unit_rewrite"


def unit_mode_enabled(options: Optional[Dict[str, Any]] = None) -> bool:
    """Report whether this run addresses repairs by unit id.

    A per-run option wins over the process-wide default so one job can be put
    on the new path without restarting anything; the config value is read at
    call time so the default is never frozen at import.
    """

    from src import config

    requested = (options or {}).get("editor_unit_mode")
    if requested is None:
        return bool(getattr(config, "EDITOR_UNIT_MODE", False))
    return bool(requested)


def _unit_index(units: Iterable[Unit]) -> Dict[str, Unit]:
    return {unit.unit_id: unit for unit in units}


def collect_unit_rewrite_patches(
    draft_text: str,
    source_text: str,
    issues: Iterable[Dict[str, Any]],
    *,
    units: Optional[Iterable[Unit]] = None,
) -> Tuple[List[Tuple[int, int, str, str]], List[Dict[str, Any]]]:
    """Return `(patches, unresolved)` for unit-rewrite issues.

    Units are re-derived from the same two texts the editor was shown, which is
    the alignment it answered against: the aligner reads nothing but those two
    strings, so recomputing it is cheaper than carrying it through the pass.
    """

    draft = str(draft_text or "")
    known = _unit_index(
        units if units is not None else align_units(source_text, draft)
    )
    patches: List[Tuple[int, int, str, str]] = []
    unresolved: List[Dict[str, Any]] = []
    claimed: set[str] = set()
    for issue in issues or []:
        unit_id = str(issue.get("unit_id") or "").strip().upper()
        rewritten = str(issue.get("rewritten_unit") or "")
        issue_id = str(issue.get("issue_id") or "issue")
        unit = known.get(unit_id)
        reason = ""
        if unit is None:
            reason = "unit_unknown"
        elif unit.is_omission or unit.draft_start < 0:
            # There is no draft text to replace. The source is missing from the
            # translation entirely, which is a real finding and a job for the
            # rewrite pass, not for a span edit.
            reason = "unit_has_no_draft"
        elif not rewritten.strip():
            reason = "unit_rewrite_empty"
        elif unit_id in claimed:
            # Two rewrites of one unit are two answers to the same question,
            # and nothing here can tell which was meant.
            reason = "unit_rewrite_conflict"
        elif rewrite_change_ratio(unit.draft, rewritten) > MAX_REWRITE_CHANGE_RATIO:
            reason = "unit_rewrite_drift"
        if reason:
            issue["unresolved_reason"] = reason
            unresolved.append(issue)
            continue
        computed = unit_rewrite_patches(unit, rewritten, issue_id)
        if not computed:
            # The rewrite is the draft. The editor described a defect and then
            # returned the text unchanged, so there is nothing to apply and
            # nothing to hand the repair pass either.
            issue["unresolved_reason"] = "unit_rewrite_no_op"
            unresolved.append(issue)
            continue
        claimed.add(unit_id)
        patches.extend(computed)
    return patches, unresolved
