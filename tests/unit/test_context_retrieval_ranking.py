"""What the prompt keeps when not everything relevant fits.

Selection alone is enough while the budget is generous. On a long book it stops
being enough: a chapter can name thirty characters, and the pinned gender roster
names every character in the book on every chunk. Left to itself the renderer
gave way in file order and in section order -- so the cast discovered first won
regardless of who the chapter was about, and addressing, rendered last, was
starved by a roster of characters who were not in the scene.
"""
from src.utils.novel_context import render_novel_context_for_prompt
from src.utils.novel_context.constants import (
    ADDRESSING_SECTION,
    ALIASES_SECTION,
    CHARACTERS_SECTION,
    DYNAMIC_STATE_END,
    DYNAMIC_STATE_START,
    GLOSSARY_SECTION,
)
from src.utils.novel_context.retrieval import (
    cap_roster,
    rank,
    score_entry,
    split_budget,
)


def build_context(characters, addressing=(), aliases=(), glossary=()):
    lines = ["# GLOBAL LORE", "", CHARACTERS_SECTION]
    lines += [f"- {name}: {value}" for name, value in characters]
    if aliases:
        lines += ["", ALIASES_SECTION]
        lines += [f"- {alias}: {target}" for alias, target in aliases]
    if glossary:
        lines += ["", GLOSSARY_SECTION]
        lines += [f"- {term}: {value}" for term, value in glossary]
    lines += ["", DYNAMIC_STATE_START, "# DYNAMIC RELATIONSHIP STATE"]
    if addressing:
        lines += ["", ADDRESSING_SECTION]
        lines += list(addressing)
    lines.append(DYNAMIC_STATE_END)
    return "\n".join(lines)


LARGE_CAST = [
    (f"Filler{index}", "Male, a minor retainer of the northern house")
    for index in range(60)
]


class TestScoring:
    def test_a_named_character_outranks_a_merely_described_one(self):
        named = score_entry("Sora", "Male, a swordsman", "Sora drew his blade.")
        described = score_entry(
            "Rin", "Female, a swordsman of the north", "Sora drew his blade."
        )
        assert named > described

    def test_a_character_on_stage_outranks_one_the_chunk_ignores(self):
        on_stage = score_entry(
            "Rin", "Female", "She said nothing.", on_stage_keys={"rin"}
        )
        ignored = score_entry("Kaede", "Female", "She said nothing.")
        assert on_stage > ignored
        assert ignored == 0

    def test_the_viewpoint_character_scores_above_nothing(self):
        assert score_entry("Sora", "Male", "I said nothing.", is_pov=True) > 0


class TestRanking:
    def test_ranking_keeps_file_order_within_one_score(self):
        items = [("a", ""), ("b", ""), ("c", "")]
        assert rank(items, lambda _entry: 5) == items

    def test_ranking_puts_the_strongest_match_first(self):
        items = [("ignored", ""), ("wanted", "")]
        ranked = rank(items, lambda entry: 100 if entry[0] == "wanted" else 0)
        assert ranked[0][0] == "wanted"


class TestBudgetSplit:
    def test_no_budget_reserves_nothing(self):
        assert split_budget(0) == (0, 0)

    def test_the_dynamic_state_keeps_a_share_back_from_the_lore(self):
        lore, reserve = split_budget(1000)
        assert reserve > 0
        assert lore + reserve == 1000


class TestRosterCap:
    def test_a_short_roster_is_untouched(self):
        lines = ["- A: Male", "- B: Female"]
        assert cap_roster(lines) == (lines, 0)

    def test_a_long_roster_is_cut_and_the_cut_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            "src.utils.novel_context.retrieval._gender_roster_max", lambda: 3
        )
        kept, cut = cap_roster([f"- N{index}: Male" for index in range(10)])
        assert len(kept) == 3
        assert cut == 7

    def test_the_cap_can_be_turned_off(self, monkeypatch):
        monkeypatch.setattr(
            "src.utils.novel_context.retrieval._gender_roster_max", lambda: 0
        )
        lines = [f"- N{index}: Male" for index in range(200)]
        assert cap_roster(lines) == (lines, 0)


class TestRenderedPrompt:
    def test_the_roster_no_longer_grows_with_the_whole_book(self, monkeypatch):
        monkeypatch.setattr(
            "src.utils.novel_context.retrieval._gender_roster_max", lambda: 10
        )
        context = build_context(
            [("Sora", "Male, the swordsman")] + LARGE_CAST,
        )

        rendered = render_novel_context_for_prompt(
            context, reference_text="Sora drew his blade."
        )

        assert "Sora" in rendered
        assert rendered.count("Filler") == 10

    def test_the_character_the_chunk_is_about_survives_a_tight_budget(self):
        # Zoya sorts last in the file; the chunk is entirely about her.
        context = build_context(
            LARGE_CAST + [("Zoya", "Female, the exiled queen")],
        )

        rendered = render_novel_context_for_prompt(
            context,
            reference_text="Zoya stepped into the throne room alone.",
            max_tokens=260,
        )

        assert "Zoya" in rendered

    def test_addressing_is_not_starved_by_a_long_cast(self):
        context = build_context(
            LARGE_CAST + [("Sora", "Male, the swordsman"), ("Rin", "Female, his mentor")],
            addressing=['- Sora → Rin: self "em", target "cô" (formal)'],
        )

        rendered = render_novel_context_for_prompt(
            context,
            reference_text="Sora bowed to Rin before he spoke.",
            max_tokens=300,
        )

        assert "Sora → Rin" in rendered
        assert '"cô"' in rendered

    def test_what_the_budget_cut_is_reported(self, caplog):
        context = build_context(LARGE_CAST + [("Sora", "Male, the swordsman")])

        with caplog.at_level("INFO"):
            render_novel_context_for_prompt(
                context, reference_text="Sora drew his blade.", max_tokens=180
            )

        assert any("left out" in record.message for record in caplog.records)

    def test_a_generous_budget_still_renders_everything_relevant(self):
        context = build_context(
            [("Sora", "Male, the swordsman"), ("Rin", "Female, his mentor")],
            addressing=['- Sora → Rin: self "em", target "cô" (formal)'],
        )

        rendered = render_novel_context_for_prompt(
            context, reference_text="Sora bowed to Rin before he spoke."
        )

        assert "Sora" in rendered and "Rin" in rendered
        assert "Sora → Rin" in rendered
