"""Guard against names that only fail at run time.

The per-unit novel-context reload in GenericTranslator called helpers that
were never imported into the module. The NameError was caught by the
surrounding `except`, so every reload silently fell back to stale context
and only a log line hinted at it. pyflakes finds this class of bug
statically; string annotations are excluded because they are never
evaluated at run time.
"""

from pathlib import Path

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
pyflakes_reporter = pytest.importorskip("pyflakes.reporter")

ROOT = Path(__file__).resolve().parents[2]

GUARDED_MODULES = [
    "src/core/adapters/generic_translator.py",
    "src/core/common/plain_text_pipeline.py",
    "src/core/epub/xhtml_translator.py",
    "src/core/epub/translator.py",
    "src/core/llm/base.py",
    "src/core/editor/requests.py",
    "src/core/context/analysis.py",
    "src/core/context/commit.py",
    "src/core/context/unit_pipeline.py",
]


class _Collector:
    def __init__(self):
        self.messages = []

    def unexpectedError(self, filename, msg):
        self.messages.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.messages.append(f"{filename}:{lineno}: {msg}")

    def flake(self, message):
        self.messages.append(str(message))


@pytest.mark.parametrize("relative_path", GUARDED_MODULES)
def test_module_has_no_undefined_names(relative_path):
    path = ROOT / relative_path
    collector = _Collector()
    pyflakes_api.check(path.read_text(encoding="utf-8"), str(path), collector)
    undefined = [m for m in collector.messages if "undefined name" in m]
    assert undefined == [], "\n".join(undefined)
