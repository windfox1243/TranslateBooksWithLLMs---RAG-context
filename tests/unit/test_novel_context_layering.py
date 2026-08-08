"""Keep the two layers lifted out of characters.py from sinking back into it.

characters.py was one 2,300-line module because character classification,
gender inference and value merging really are mutually recursive: they all go
through _normalize_character_value, which reads and rewrites the same string.

Two layers were not part of that cycle and came out:

  name_keys        naming, key derivation and name comparison -- a leaf
  character_facts  the fact list a character value carries -- name_keys only

The value of the split is exactly that those two do not reach back. A single
new import in the wrong direction restores the cycle and silently un-does it,
which is what these tests exist to catch. If a genuinely name-shaped helper
needs something from characters.py, the helper is in the wrong module.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "utils" / "novel_context"


def _imported_siblings(module_name: str) -> set[str]:
    """Names of same-package modules that `module_name` imports from."""
    tree = ast.parse((PACKAGE / f"{module_name}.py").read_text(encoding="utf-8"))
    siblings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            siblings.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                prefix = "src.utils.novel_context."
                if alias.name.startswith(prefix):
                    siblings.add(alias.name[len(prefix):])
    return siblings


def test_name_keys_only_reads_constants():
    assert _imported_siblings("name_keys") <= {"constants"}


def test_character_facts_only_reads_constants_and_name_keys():
    assert _imported_siblings("character_facts") <= {"constants", "name_keys"}


def test_neither_layer_imports_characters():
    for module in ("name_keys", "character_facts"):
        assert "characters" not in _imported_siblings(module), (
            f"{module} imports characters, which puts the cycle back"
        )


def test_characters_still_re_exports_everything_it_moved_out():
    """Sibling modules import these through .characters and must keep working."""
    from src.utils.novel_context import character_facts, characters, name_keys

    moved_constants = {
        "_LATIN_BOUNDARY_CHARS",
        "_LATIN_NAME_PART_STOPWORDS",
        "_UNSTABLE_PHYSICAL_WORDS",
        "_DETAIL_STOP_WORDS",
    }
    for source in (name_keys, character_facts):
        moved = {
            name
            for name, value in vars(source).items()
            if callable(value) and getattr(value, "__module__", "") == source.__name__
        } | (moved_constants & set(vars(source)))
        assert moved, f"{source.__name__} defines nothing"
        missing = sorted(name for name in moved if not hasattr(characters, name))
        assert not missing, f"characters stopped re-exporting {missing}"
