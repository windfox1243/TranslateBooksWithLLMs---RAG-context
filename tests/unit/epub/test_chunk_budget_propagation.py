from src.core.docx.docx_translation_adapter import DocxTranslationAdapter
from src.core.epub.container import TranslationContainer
from src.core.epub.epub_translation_adapter import EpubTranslationAdapter


class _RecordingChunker:
    created_with = []

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.created_with.append(max_tokens)

    def chunk_html_with_placeholders(self, text, tag_map):
        return [{"text": text, "max_tokens": self.max_tokens}]


def test_docx_adapter_honors_explicit_job_chunk_budget(monkeypatch):
    import src.core.docx.docx_translation_adapter as module

    _RecordingChunker.created_with = []
    adapter = DocxTranslationAdapter()
    monkeypatch.setattr(module, "HtmlChunker", _RecordingChunker)

    chunks = adapter.create_chunks("document", {}, 137, None)

    assert _RecordingChunker.created_with == [137]
    assert chunks[0]["max_tokens"] == 137


def test_epub_adapter_honors_explicit_job_chunk_budget(monkeypatch):
    import src.core.epub.html_chunker as chunker_module

    _RecordingChunker.created_with = []
    adapter = EpubTranslationAdapter()
    monkeypatch.setattr(chunker_module, "HtmlChunker", _RecordingChunker)

    chunks = adapter.create_chunks("document", {}, 163, None)

    assert _RecordingChunker.created_with == [163]
    assert chunks[0]["max_tokens"] == 163


def test_shared_xhtml_chunker_honors_explicit_budget(monkeypatch):
    import src.core.epub.xhtml_translator as module

    _RecordingChunker.created_with = []
    monkeypatch.setattr(module, "HtmlChunker", _RecordingChunker)

    chunks = module._create_chunks(
        "document",
        {},
        211,
        container=TranslationContainer(),
    )

    assert _RecordingChunker.created_with == [211]
    assert chunks[0]["max_tokens"] == 211
