"""The editor quotes segments, so a segment must not stop mid-sentence."""

from src.utils.translation_quality import build_editor_segments


def _covers(text, segments):
    """Every offset is exact and the segments read back as the original."""

    return (
        all(text[item["start"]:item["end"]] == item["text"] for item in segments)
        and "".join(item["text"] for item in segments) == text
    )


def test_a_name_carrying_an_abbreviation_stays_in_one_segment():
    # The measured chunk: the draft was offered as two segments split at "St.",
    # so the editor's quote stopped there and no repair in the batch applied.
    text = "Anh thắng Kikuka-shō (Japanese St. Leger) năm ngoái. Rồi anh nghỉ."

    segments = build_editor_segments(text)

    assert [item["text"] for item in segments] == [
        "Anh thắng Kikuka-shō (Japanese St. Leger) năm ngoái. ",
        "Rồi anh nghỉ.",
    ]
    assert _covers(text, segments)


def test_ordinary_sentences_still_split():
    text = "Câu một. Câu hai! Câu ba?"

    segments = build_editor_segments(text)

    assert [item["text"] for item in segments] == [
        "Câu một. ", "Câu hai! ", "Câu ba?",
    ]
    assert _covers(text, segments)


def test_an_initial_does_not_end_a_segment():
    text = "Ông J. Smith đến. Rồi đi."

    segments = build_editor_segments(text)

    assert [item["text"] for item in segments] == ["Ông J. Smith đến. ", "Rồi đi."]


def test_a_line_break_ends_a_segment_whatever_the_bracket_says():
    # Merging across a paragraph break would hand the editor one locator
    # spanning two paragraphs, which is worse than the split it avoids.
    text = "Mở ngoặc (chưa đóng.\n\nĐoạn hai."

    segments = build_editor_segments(text)

    assert [item["text"] for item in segments] == [
        "Mở ngoặc (chưa đóng.\n\n", "Đoạn hai.",
    ]
    assert _covers(text, segments)


def test_a_dangling_open_bracket_at_the_end_still_yields_a_segment():
    text = "Câu một. Câu hai (chưa đóng."

    segments = build_editor_segments(text)

    assert [item["text"] for item in segments] == [
        "Câu một. ", "Câu hai (chưa đóng.",
    ]
    assert _covers(text, segments)


def test_segment_ids_stay_dense_after_a_merge():
    text = "Xem vol. 3 nhé. Hết."

    segments = build_editor_segments(text)

    assert [item["segment_id"] for item in segments] == ["SEG-0001", "SEG-0002"]
