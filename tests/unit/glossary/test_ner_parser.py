"""
Unit tests for parse_ner_response.

Verifies the permissive NER JSON parser: tag extraction, fence/array/object
fallbacks, repair of trailing commas, deduplication, alias keys, and warning
emission.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.ner import (
    apply_related_glossary_to_candidates,
    parse_ner_response,
    related_existing_glossary_terms,
    suggest_terms,
)


class _StaticProvider:
    """Minimal LLM provider returning a fixed NER payload."""

    def __init__(self, content):
        self.content = content

    async def generate(self, user_prompt, system_prompt=None):
        return self.content


class TestRelatedExistingGlossaryTerms:
    """Relevant old glossary entries are selected as hints for NER."""

    def test_exact_term_match_is_selected_for_compound_context(self):
        related = related_existing_glossary_terms(
            "The Zone Gate opened above the city.",
            {
                "Zone": "Zone",
                "Unrelated Kingdom": "Vuong quoc khong lien quan",
            },
        )

        assert related == {"Zone": "Zone"}

    def test_shared_meaningful_keyword_can_select_longer_old_entry(self):
        related = related_existing_glossary_terms(
            "A new Academy Zone appeared.",
            {
                "Summoner Academy": "Hoc vien Trieu hoi",
                "of": "cua",
            },
        )

        assert related == {"Summoner Academy": "Hoc vien Trieu hoi"}

    def test_caps_related_terms(self):
        glossary = {f"Zone {index}": f"Zone {index}" for index in range(5)}
        related = related_existing_glossary_terms(
            "Zone 0 Zone 1 Zone 2 Zone 3 Zone 4",
            glossary,
            max_entries=2,
        )

        assert len(related) == 2


class TestApplyRelatedGlossaryToCandidates:
    """Related glossary rows repair obvious untranslated fragments only."""

    def test_repairs_untranslated_component_in_compound_target(self):
        candidates = [
            {
                "source": "Domain Zone",
                "target": "Domain zone",
                "category": "location",
            }
        ]

        repaired = apply_related_glossary_to_candidates(
            candidates,
            {"Domain": "lanh dia", "zone": "zone"},
        )

        assert repaired[0]["target"] == "lanh dia zone"

    def test_leaves_already_translated_compound_target_unchanged(self):
        candidates = [
            {
                "source": "Domain Zone",
                "target": "lanh dia zone",
                "category": "location",
            }
        ]

        repaired = apply_related_glossary_to_candidates(
            candidates,
            {"Domain": "lanh dia", "zone": "zone"},
        )

        assert repaired[0]["target"] == "lanh dia zone"

    def test_does_not_rewrite_idiomatic_target_without_raw_source_fragment(self):
        candidates = [
            {
                "source": "Domain Zone",
                "target": "territorial field",
                "category": "location",
            }
        ]

        repaired = apply_related_glossary_to_candidates(
            candidates,
            {"Domain": "lanh dia", "zone": "zone"},
        )

        assert repaired[0]["target"] == "territorial field"

    @pytest.mark.asyncio
    async def test_suggest_terms_repairs_ignored_related_glossary_component(self):
        provider = _StaticProvider(
            '<NER_JSON>[{"source":"Domain Zone","target":"Domain zone",'
            '"category":"location"}]</NER_JSON>'
        )

        candidates, warnings = await suggest_terms(
            "The Domain Zone opened.",
            "English",
            "Vietnamese",
            provider,
            existing_glossary_terms={"Domain": "lanh dia", "zone": "zone"},
        )

        assert warnings == []
        assert candidates == [
            {
                "source": "Domain Zone",
                "target": "lanh dia zone",
                "category": "location",
            }
        ]


class TestParseNerResponseBasic:
    """Basic parsing paths for parse_ner_response."""

    def test_empty_input_returns_warning(self):
        """Empty input returns ([], ['empty LLM response'])."""
        candidates, warnings = parse_ner_response("")
        assert candidates == []
        assert warnings == ["empty LLM response"]

    def test_clean_tagged_response(self):
        """Clean <NER_JSON>...</NER_JSON> response yields one candidate, no warnings."""
        raw = '<NER_JSON>[{"source":"X","target":"x","category":"character"}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "X"
        assert candidates[0]["target"] == "x"
        assert candidates[0]["category"] == "character"
        assert warnings == []

    def test_markdown_code_fence_fallback(self):
        """Markdown ```json fence (no NER tags) is extracted with a warning."""
        raw = 'Here you go:\n```json\n[{"source":"A","target":"a","category":"item"}]\n```'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "A"
        assert any("code fence" in w for w in warnings)

    def test_bare_balanced_array_fallback(self):
        """Bare [...] array (no tags, no fence) is extracted with a warning."""
        raw = '[{"source":"A","target":"a","category":"item"}]'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "A"
        assert any("balanced JSON array" in w for w in warnings)


class TestParseNerResponseObjectUnwrap:
    """Tests for the object-unwrap fallback paths."""

    def test_object_with_entities_key_unwraps(self):
        """A bare object with an 'entities' key unwraps the inner list."""
        raw = '{"entities":[{"source":"A","target":"a","category":"item"}]}'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "A"
        assert any("entities" in w for w in warnings)

    def test_object_with_terms_key_unwraps(self):
        """A bare object with a 'terms' key unwraps the inner list."""
        raw = '{"terms":[{"source":"A","target":"a","category":"item"}]}'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "A"
        assert any("terms" in w for w in warnings)

    def test_single_object_with_source_key_coerced_to_list(self):
        """A single object containing a 'source' key is coerced to a one-element list."""
        raw = '<NER_JSON>{"source":"X","target":"x","category":"character"}</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "X"
        assert candidates[0]["target"] == "x"
        assert candidates[0]["category"] == "character"


class TestParseNerResponseRepair:
    """JSON repair and broken-input handling."""

    def test_trailing_comma_is_repaired(self):
        """Trailing comma is repaired and a 'repaired' warning is emitted."""
        raw = '<NER_JSON>[{"source":"A","target":"a","category":"item",}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "A"
        assert any("repaired" in w for w in warnings)

    def test_truly_broken_json_returns_parse_error(self):
        """Unrepairable JSON yields no candidates and a 'JSON parse error' warning."""
        raw = '<NER_JSON>[{"source":</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert candidates == []
        assert any(w.startswith("JSON parse error") for w in warnings)

    def test_garbage_text_with_no_json(self):
        """Pure garbage with no JSON yields a 'could not locate' warning."""
        raw = "lorem ipsum dolor sit amet, no json here at all"
        candidates, warnings = parse_ner_response(raw)
        assert candidates == []
        assert any("could not locate" in w for w in warnings)


class TestParseNerResponseThinkingAndDedup:
    """Thinking-block stripping and silent deduplication."""

    def test_thinking_block_is_stripped(self):
        """A <think>...</think> block before the NER tags is stripped silently."""
        raw = (
            "<think>let me reason about this passage carefully</think>"
            '<NER_JSON>[{"source":"X","target":"x","category":"character"}]</NER_JSON>'
        )
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "X"
        assert warnings == []

    def test_duplicate_sources_are_silently_deduped(self):
        """Two entries with the same 'source' keep only the first; no dup warning."""
        raw = (
            '<NER_JSON>['
            '{"source":"X","target":"x","category":"character"},'
            '{"source":"X","target":"y","category":"item"}'
            ']</NER_JSON>'
        )
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["target"] == "x"
        assert candidates[0]["category"] == "character"
        assert warnings == []


class TestParseNerResponseEntries:
    """Per-entry handling: categories, missing fields, defaults, aliases."""

    def test_unknown_category_is_kept_with_warning(self):
        """An unknown category is kept on the candidate but a warning is emitted."""
        raw = '<NER_JSON>[{"source":"X","target":"x","category":"weird"}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["category"] == "weird"
        assert any("unknown category" in w for w in warnings)

    def test_missing_target_field_yields_empty_string(self):
        """An entry without a 'target' field is kept with target=''."""
        raw = '<NER_JSON>[{"source":"X","category":"character"}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "X"
        assert candidates[0]["target"] == ""

    def test_empty_source_field_is_skipped(self):
        """An entry with an empty source string is skipped with a warning."""
        raw = '<NER_JSON>[{"source":"","target":"x","category":"character"}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert candidates == []
        assert any("skipped entry without 'source'" in w for w in warnings)

    def test_default_category_is_other(self):
        """An entry without a 'category' field defaults to 'other'."""
        raw = '<NER_JSON>[{"source":"X","target":"x"}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["category"] == "other"

    def test_cjk_source_preserved_exactly(self):
        """A CJK source string is preserved exactly as input."""
        raw = '<NER_JSON>[{"source":"李凡","target":"Li Fan","category":"character"}]</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "李凡"
        assert candidates[0]["target"] == "Li Fan"
        assert candidates[0]["category"] == "character"

    def test_aliased_keys_are_recognized(self):
        """source_term, translation, and type are accepted as aliases."""
        raw = (
            '<NER_JSON>[{"source_term":"X","translation":"x","type":"character"}]'
            '</NER_JSON>'
        )
        candidates, warnings = parse_ner_response(raw)
        assert len(candidates) == 1
        assert candidates[0]["source"] == "X"
        assert candidates[0]["target"] == "x"
        assert candidates[0]["category"] == "character"


class TestParseNerResponseRootType:
    """Tests for unexpected JSON root types."""

    def test_invalid_root_type_number(self):
        """A bare number as JSON root yields a warning about the root type."""
        raw = '<NER_JSON>42</NER_JSON>'
        candidates, warnings = parse_ner_response(raw)
        assert candidates == []
        assert any("unexpected JSON root type" in w for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
