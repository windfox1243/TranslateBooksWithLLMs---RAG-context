"""The deterministic half of the unit-rewrite editor.

Everything here runs without a model. That is the point of the design: the
work the old contract asked a model to do -- quote a span character for
character -- is arithmetic, and arithmetic can be pinned by a test.
"""

from src.core.editor.units import (
    align_units,
    format_bitext,
    rewrite_change_ratio,
    split_blocks,
    unit_rewrite_patches,
)


def _apply(draft, patches):
    """Apply computed spans to a draft, rightmost first."""

    for start, end, replacement, _ in sorted(patches, reverse=True):
        draft = draft[:start] + replacement + draft[end:]
    return draft


def test_blocks_keep_the_offsets_they_came_from():
    text = "First one.\n\n  Second one.  \n\nThird one."
    blocks = split_blocks(text)
    assert [block.text for block in blocks] == [
        "First one.", "Second one.", "Third one.",
    ]
    for block in blocks:
        assert text[block.start:block.end] == block.text


def test_text_without_blank_lines_splits_on_lines_instead():
    # Subtitles, verse and stripped exports have no blank line anywhere, and
    # one unit spanning the whole chunk would make the unit id meaningless.
    text = "Line one\nLine two\nLine three"
    assert [block.text for block in split_blocks(text)] == [
        "Line one", "Line two", "Line three",
    ]


def test_paragraphs_pair_one_to_one():
    source = "Alpha one.\n\nBeta two.\n\nGamma three."
    draft = "Alpha un.\n\nBeta deux.\n\nGamma trois."
    units = align_units(source, draft)
    assert [unit.unit_id for unit in units] == ["U-0001", "U-0002", "U-0003"]
    assert [unit.draft for unit in units] == [
        "Alpha un.", "Beta deux.", "Gamma trois.",
    ]
    for unit in units:
        assert draft[unit.draft_start:unit.draft_end] == unit.draft


def test_two_source_paragraphs_joined_in_the_draft_are_one_unit():
    # A translator joining two short paragraphs is not an error and must not
    # push every later unit off by one, which a naive zip would do.
    source = "Alpha one.\n\nBeta two.\n\nGamma three is a much longer paragraph."
    draft = "Alpha un. Beta deux.\n\nGamma trois is a much longer paragraph."
    units = align_units(source, draft)
    assert len(units) == 2
    assert (units[0].source_blocks, units[0].draft_blocks) == (2, 1)
    assert units[1].draft.startswith("Gamma trois")


def test_a_paragraph_the_draft_never_produced_is_reported_as_an_omission():
    # Found by counting, not by asking: the aligner has to leave the source
    # unit unpaired, and an unpaired source unit is an omission. It only gets
    # there by re-measuring the length ratio after a first pass -- taken from
    # the two texts whole, the ratio is depressed by the very omission being
    # looked for, which makes the gap read as ordinary compression.
    source = (
        "Alpha one is here.\n\nBeta two is entirely missing.\n\n"
        "Gamma three is here."
    )
    draft = "Alpha one is here.\n\nGamma three is here."
    units = align_units(source, draft)
    omitted = [unit for unit in units if unit.is_omission]
    assert len(omitted) == 1
    assert omitted[0].source == "Beta two is entirely missing."
    assert omitted[0].draft == ""


def test_a_dropped_short_paragraph_still_reaches_the_editor():
    """Length alone cannot tell a merge from a short omission.

    Both leave one draft block of about the right length, so the aligner reads
    this real case -- a line of dialogue the translator dropped -- as a merge.
    What matters is that it does not thereby disappear: the untranslated
    sentence stays on the source side of the unit the editor is shown, which is
    the whole point of putting the two languages side by side.
    """

    source = '"That is it for the meeting. Rest up for tomorrow!"\n\n"Got it!"'
    draft = '"Cuoc hop ket thuc tai day. Nghi ngoi cho ngay mai di!"'
    units = align_units(source, draft)
    assert len(units) == 1
    assert "Got it!" in units[0].source
    assert "Got it!" in format_bitext(units)


def test_the_bitext_puts_each_pair_under_one_id():
    source = "Alpha one.\n\nBeta two."
    draft = "Alpha un.\n\nBeta deux."
    rendered = format_bitext(align_units(source, draft))
    assert rendered.splitlines() == [
        "[U-0001]",
        "  SRC   Alpha one.",
        "  DRAFT Alpha un.",
        "[U-0002]",
        "  SRC   Beta two.",
        "  DRAFT Beta deux.",
    ]


def test_a_draft_only_view_omits_the_source_side():
    rendered = format_bitext(
        align_units("", "Alpha un."), source_available=False,
    )
    assert "SRC" not in rendered
    assert "DRAFT Alpha un." in rendered


def test_a_rewritten_unit_becomes_exact_draft_spans():
    source = "Alpha one.\n\nBeta two."
    draft = "Alpha un.\n\nBeta deux."
    unit = align_units(source, draft)[1]
    patches = unit_rewrite_patches(unit, "Beta trois.", "issue-1")
    # The span is computed from the rewrite, so it lands in the draft's own
    # coordinates and no search can fail to find it.
    assert _apply(draft, patches) == "Alpha un.\n\nBeta trois."
    for start, end, _, issue_id in patches:
        assert draft[start:end] in unit.draft
        assert issue_id == "issue-1"


def test_edits_a_few_characters_apart_are_one_patch():
    # Diffing a rewritten sentence otherwise yields a patch per changed word,
    # which gives the overlap rules a dozen adjacent spans to argue about.
    unit = align_units("Alpha.", "the cat sat on the mat")[0]
    patches = unit_rewrite_patches(unit, "the dog sat on the rug")
    assert len(patches) == 1
    assert _apply("the cat sat on the mat", patches) == "the dog sat on the rug"


def test_distant_edits_stay_separate_patches():
    draft = "Alpha here. " + "filler " * 12 + "omega there."
    unit = align_units("source", draft)[0]
    patches = unit_rewrite_patches(
        unit, draft.replace("Alpha", "Beta").replace("omega", "sigma"),
    )
    assert len(patches) == 2


def test_an_unchanged_rewrite_produces_nothing():
    unit = align_units("Alpha one.", "Alpha un.")[0]
    assert unit_rewrite_patches(unit, "Alpha un.") == []


def test_the_change_ratio_separates_a_repair_from_a_rewrite():
    # Handed a whole unit a model will improve what nobody asked about, and
    # this is the number that lets that be refused rather than merely regretted.
    assert rewrite_change_ratio(
        "The gate order was announced.", "The gate order was posted.",
    ) < 0.3
    assert rewrite_change_ratio(
        "The gate order was announced.", "Nothing at all resembling the first.",
    ) > 0.8
    assert rewrite_change_ratio("Same text.", "Same text.") == 0.0


def test_alignment_survives_a_draft_that_drifts_far_off_the_diagonal():
    # Wider than the band, so this only passes if the unbanded retry runs.
    # A draft missing seven paragraphs in eight is ambiguous to a length-only
    # aligner -- pairing twenty and dropping seventy reads no better than
    # merging pairs and dropping fifty -- so what is asserted is the invariant
    # that holds either way: every block is accounted for exactly once.
    source = "\n\n".join(f"Source paragraph number {index}." for index in range(90))
    draft = "\n\n".join(f"Draft paragraph number {index}." for index in range(20))
    units = align_units(source, draft)
    assert sum(unit.source_blocks for unit in units) == 90
    assert sum(unit.draft_blocks for unit in units) == 20
    assert sum(1 for unit in units if unit.is_omission) >= 50


def test_empty_texts_align_to_nothing():
    assert align_units("", "") == []
    assert [unit.draft for unit in align_units("", "Only a draft.")] == [
        "Only a draft.",
    ]
    assert align_units("Only a source.", "")[0].is_omission is True
