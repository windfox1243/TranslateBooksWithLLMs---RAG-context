"""A job in which no unit got a translation ends 'error', not 'partial'.

Before, a provider failing every request produced a 'partial' job whose
output was 100% source text. The verdict now calls it a failure (web status
'error' with error_code 'nothing_translated', failure notification), while a
job with at least one translated unit and at least one failed unit stays
'partial'. Text-free EPUB chunks, passed through without an LLM call, are
reported as ``skipped_chunks`` so they do not count as translated work.
(Ported from upstream bec5fe93.)
"""
import os
import tempfile

import pytest

from src.api.completion_status import classify_completion
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.translator import _global_stats_payload, _precount_chunks


@pytest.fixture
def output(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("untouched source text", encoding="utf-8")
    return str(path)


def test_every_chunk_failed_is_an_error(output):
    stats = {'total_chunks': 6, 'completed_chunks': 6, 'failed_chunks': 6}
    verdict = classify_completion(stats, output)
    assert verdict.status == 'error'
    assert verdict.nothing_translated
    assert verdict.error


def test_every_chunk_left_as_source_is_an_error_even_with_text_free_chunks(output):
    stats = {'total_chunks': 65, 'completed_chunks': 65, 'failed_chunks': 64,
             'fallback_used': 64, 'skipped_chunks': 1}
    verdict = classify_completion(stats, output)
    assert verdict.status == 'error' and verdict.nothing_translated


def test_fallback_counter_alone_is_enough(output):
    stats = {'total_chunks': 22, 'completed_chunks': 22, 'fallback_used': 22}
    assert classify_completion(stats, output).status == 'error'


@pytest.mark.parametrize("stats", [
    {'total_chunks': 6, 'completed_chunks': 6, 'failed_chunks': 5},
    {'total_chunks': 65, 'completed_chunks': 65, 'fallback_used': 63, 'skipped_chunks': 1},
])
def test_one_translated_unit_keeps_the_job_partial(output, stats):
    verdict = classify_completion(stats, output)
    assert verdict.status == 'partial'
    assert not verdict.nothing_translated


def test_a_book_with_nothing_to_translate_is_still_completed(output):
    stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 0,
             'skipped_chunks': 3}
    assert classify_completion(stats, output).status == 'completed'


def test_missing_output_is_not_reported_as_nothing_translated(tmp_path):
    stats = {'total_chunks': 2, 'failed_chunks': 2}
    verdict = classify_completion(stats, str(tmp_path / "missing.txt"))
    assert verdict.status == 'error'
    assert not verdict.nothing_translated


# -- EPUB: text-free chunks are counted by the precount --------------------

_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>t</title></head>
  <body>
    {body}
  </body>
</html>
"""


@pytest.mark.asyncio
async def test_precount_reports_text_free_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "cover.xhtml"), "w", encoding="utf-8") as f:
            f.write(_XHTML.format(body='<div><img src="cover.jpg" alt=""/></div>'))
        with open(os.path.join(tmp, "ch1.xhtml"), "w", encoding="utf-8") as f:
            f.write(_XHTML.format(body="<p>Hello there, this is a sentence.</p>"))

        precount = await _precount_chunks(
            content_files=["cover.xhtml", "ch1.xhtml"],
            opf_dir=tmp,
            max_tokens_per_chunk=500,
        )

    # The cover page is one chunk of pure markup; the prose chapter has letters.
    assert precount.chunks_per_file[0] == 1
    assert precount.chunks_per_file[1] >= 1
    assert precount.text_free_chunks == 1


@pytest.mark.asyncio
async def test_plain_text_mode_reports_no_text_free_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "ch1.xhtml"), "w", encoding="utf-8") as f:
            f.write(_XHTML.format(body="<p>Hello there.</p>"))
        precount = await _precount_chunks(
            content_files=["ch1.xhtml"], opf_dir=tmp, max_tokens_per_chunk=500,
            plain_text_mode=True,
        )
    assert precount.text_free_chunks == 0


def test_stats_payload_carries_skipped_chunks():
    payload = _global_stats_payload(10, 4, TranslationMetrics(), skipped_chunks=2)
    assert payload['skipped_chunks'] == 2
    assert _global_stats_payload(10, 4, TranslationMetrics())['skipped_chunks'] == 0
