"""Pin the import surface of src.utils.novel_context.

novel_context was split from a single 9,800-line module into a package. Every
caller reaches it through `from src.utils.novel_context import ...` (an AST scan
of src/, tests/, scripts/, benchmark/ and tools/ found no module-alias attribute
access), so re-exporting this exact set of names from the package __init__ makes
the split invisible.

The list below is the surface as it existed immediately before the split. Names
may be added, but removing or renaming one breaks a caller and must fail here.
Several entries are underscore-prefixed: they are private by naming convention
only and are imported by other modules in src/, so they are de-facto public.
"""

import importlib

# Snapshot taken from the pre-split module; do not prune.
EXPECTED_SURFACE = frozenset({
    "ADDRESSING_SECTION",
    "ALIASES_SECTION",
    "CHARACTERS_SECTION",
    "DYNAMIC_STATE_END",
    "DYNAMIC_STATE_START",
    "GLOSSARY_SECTION",
    "NovelContextSession",
    "RefinementContextTracker",
    "SOURCE_ANALYSIS_SYSTEM_PROMPT",
    "UPDATE_SYSTEM_PROMPT",
    "_bounded_source_memory",
    "_character_profile_map",
    "_find_lore_section",
    "_format_dynamic_sections",
    "_has_vietnamese_addressing_mismatch",
    "_is_disposable_unnamed_character",
    "_is_inverted_target_to_source_glossary_pair",
    "_is_non_character_group_entry",
    "_is_non_character_work_entry",
    "_json_dynamic_state",
    "_missing_addressing_requirements",
    "_normalize_glossary_entries",
    "_parse_alias_entries",
    "_parse_bullet_entries",
    "_repair_vietnamese_addressing_block",
    "_repair_vietnamese_addressing_details",
    "_repair_vietnamese_addressing_line",
    "_retry_missing_addressing_candidates",
    "_section_body",
    "_select_relevant_character_profiles",
    "_source_identity_link_proof_status",
    "_split_dynamic_sections",
    "_vietnamese_addressing_field",
    "build_novel_context",
    "character_alias_map",
    "compress_dynamic_state",
    "consolidate_context_lore",
    "decode_context_snapshot",
    "decompress_dynamic_state",
    "extract_dynamic_state_from_text",
    "extract_global_lore",
    "infer_dynamic_address_identity_links",
    "infer_source_gender_updates",
    "infer_source_identity_links",
    "is_safe_filename",
    "list_novel_contexts",
    "load_novel_context",
    "make_novel_context_filename",
    "map_context_snapshots_for_refinement",
    "map_dialogue_attributions_for_refinement",
    "merge_dynamic_state",
    "merge_new_lore",
    "normalize_global_lore",
    "normalize_novel_context_content",
    "normalize_novel_context_filename",
    "normalize_refinement_context",
    "open_novel_context_session",
    "render_novel_context_for_prompt",
    "render_novel_context_update_view",
    "resolve_novel_context_path",
    "save_novel_context",
    "should_update_novel_context_for_index",
    "update_novel_context_chunk",
})


def test_every_imported_name_is_still_reachable():
    module = importlib.import_module("src.utils.novel_context")
    missing = sorted(name for name in EXPECTED_SURFACE if not hasattr(module, name))
    assert missing == [], f"novel_context no longer exports: {missing}"


def test_callables_and_constants_are_not_none():
    module = importlib.import_module("src.utils.novel_context")
    nulls = sorted(
        name for name in EXPECTED_SURFACE if getattr(module, name, None) is None
    )
    assert nulls == [], f"novel_context exports None for: {nulls}"
