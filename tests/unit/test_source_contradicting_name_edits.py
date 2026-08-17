"""An edit may not rewrite how the source names someone at that very place.

Every case here is a real editor finding from one book: edits that renamed a
vocative the source had written plainly, one that wrote a name over the title
the source speaks, and two that respelled a name the way the source spelled it
there. Only the rewrites are wrong, and the source quote is what tells them
apart.
"""

from src.utils.translation_quality import (
    filter_source_contradicting_name_edits,
    source_anchored_proper_names,
)

SOURCE = (
    "Tomio calls McQueen and Rice by their last names, but I get the full "
    "“Apollo Rainbow” treatment?\n"
    "“Hey, Toshio—mind just calling me ‘Apollo’?”\n"
    "“Tomio, which one do you think Meek will go for?”\n"
    "Also, it’s not like I’m going on a date with Tomio.\n"
    "Date or not, I just wanna spend Christmas with my trainer!\n"
)


def issue(source_quote, draft, replacement, issue_id="ISSUE-01"):
    return {
        "issue_id": issue_id,
        "category": "pronoun bleed",
        "source_quote": source_quote,
        "draft_replacement": {"draft": draft, "replacement": replacement},
    }


def test_vocative_the_source_speaks_is_not_replaced_by_a_role_word():
    retained, rejected = filter_source_contradicting_name_edits(
        SOURCE,
        [issue(
            "“Tomio, which one do you think Meek will go for?”",
            "Tomio, anh nghĩ Meek sẽ chọn cách nào?",
            "Trainer, anh nghĩ Meek sẽ chọn cách nào?",
        )],
    )
    assert retained == []
    assert rejected == ["ISSUE-01"]


def test_a_rewritten_source_quote_does_not_license_dropping_the_name():
    # The real one: the source says `a date with Tomio`, the edit quoted it as
    # `a date with my trainer` and replaced the name with that role word.
    retained, rejected = filter_source_contradicting_name_edits(
        SOURCE,
        [issue(
            "Also, it’s not like I’m going on a date with my trainer.",
            "tớ cũng đâu có hẹn hò với Tomio.",
            "tớ cũng đâu có hẹn hò với huấn luyện viên của tớ.",
        )],
    )
    assert retained == []
    assert rejected == ["ISSUE-01"]


def test_a_verbatim_quote_without_the_name_keeps_the_edit():
    # Nothing here is about forbidding role words: where the source itself says
    # `my trainer`, an edit quoting that line may write it.
    edit = issue(
        "Date or not, I just wanna spend Christmas with my trainer!",
        "Tomio ơi, tớ chỉ muốn đón Giáng sinh cùng cậu ấy!",
        "Tớ chỉ muốn đón Giáng sinh với huấn luyện viên của mình!",
    )
    retained, rejected = filter_source_contradicting_name_edits(SOURCE, [edit])
    assert retained == [edit]
    assert rejected == []


def test_respelling_a_name_the_source_writes_there_is_kept():
    # The source really does say Toshio once -- the slip of the tongue is the
    # joke -- so restoring it is the repair, not the defect.
    edit = issue(
        "“Hey, Toshio—mind just calling me ‘Apollo’?”",
        "Này, Tomio—gọi tớ là ‘Apollo’ được không?",
        "Này, Toshio—gọi tớ là ‘Apollo’ được không?",
    )
    retained, rejected = filter_source_contradicting_name_edits(SOURCE, [edit])
    assert retained == [edit]
    assert rejected == []


def test_a_word_that_only_ever_opens_a_sentence_is_not_a_name():
    names = source_anchored_proper_names(SOURCE)
    assert "Tomio" in names
    assert "Toshio" in names
    assert "Also" not in names
    assert "Date" not in names


def test_a_stutter_is_a_word_to_translate_not_a_name():
    source = "“…W-well?” Tomio said. Well, that settles it."
    names = source_anchored_proper_names(source)
    assert "W-well" not in names
    edit = issue("…W-well?", "…W-well?", "…Ừm, thế nào?")
    retained, rejected = filter_source_contradicting_name_edits(source, [edit])
    assert retained == [edit]
    assert rejected == []


def test_an_opener_that_is_not_a_name_does_not_block_an_edit():
    edit = issue(
        "Date or not, I just wanna spend Christmas with my trainer!",
        "Date hay không, tớ chỉ muốn đón Giáng sinh với huấn luyện viên!",
        "Hẹn hò hay không, tớ chỉ muốn đón Giáng sinh với huấn luyện viên!",
    )
    retained, rejected = filter_source_contradicting_name_edits(SOURCE, [edit])
    assert retained == [edit]
    assert rejected == []


TITLE_SOURCE = (
    "I grinned and stopped outside Tomio’s door. I was the only one gunning "
    "for Tomio.\n"
    "“But Apollo, why are you so worked up? Who gives me chocolate shouldn’t "
    "matter to you, right?”\n"
    "I peeked up at him through my bangs. Tomio, ever oblivious, just tilted "
    "his head.\n"
    "“...Trainer. This is, um… handmade ganache cake...”\n"
    "So much for pretending my trainer was anyone else’s.\n"
)


def test_the_title_the_source_speaks_is_not_replaced_by_the_stored_vocative():
    # The real one: an addressing rule says Apollo calls him `Tomio`, and the
    # editor read it as licence to rewrite the line where she calls him by his
    # title -- a line whose source it quoted correctly as `...Trainer.`
    retained, rejected = filter_source_contradicting_name_edits(
        TITLE_SOURCE,
        [issue("...Trainer.", "...Huấn luyện viên.", "...Tomio.")],
    )
    assert retained == []
    assert rejected == ["ISSUE-01"]


def test_the_pronoun_repairs_in_the_same_response_are_kept():
    # Three of the four findings in that response were right, and none of them
    # is about naming: they swap the second-person pronoun the pair uses.
    edits = [
        issue(
            "But Apollo, why are you so worked up?",
            "Nhưng Apollo, tại sao cậu lại kích động thế?",
            "Nhưng Apollo, tại sao em lại kích động thế?",
            issue_id="ISSUE-02",
        ),
        issue(
            "Who gives me chocolate shouldn’t matter to you, right?",
            "Ai tặng sô-cô-la cho tôi đâu có quan trọng với cậu, đúng không?",
            "Ai tặng sô-cô-la cho anh đâu có quan trọng với em, đúng không?",
            issue_id="ISSUE-03",
        ),
    ]
    retained, rejected = filter_source_contradicting_name_edits(TITLE_SOURCE, edits)
    assert retained == edits
    assert rejected == []


def test_a_name_the_quoted_line_does_speak_may_be_written_in():
    # Vietnamese says `you` with a name often enough; the source quoting the
    # name in that line is what makes it the source's own naming.
    edit = issue(
        "But Apollo, why are you so worked up?",
        "Nhưng mà, tại sao cậu lại kích động thế?",
        "Nhưng Apollo à, tại sao em lại kích động thế?",
    )
    retained, rejected = filter_source_contradicting_name_edits(TITLE_SOURCE, [edit])
    assert retained == [edit]
    assert rejected == []


def test_a_capitalized_word_inside_a_sentence_is_not_an_address():
    # `Trainer` counts because it stands where a vocative stands. A capitalized
    # word in the middle of a sentence is just a word, and blocks nothing.
    source = "Tomio met Trainer Amami at the gate. My trainer waved back.\n"
    edit = issue(
        "Tomio met Trainer Amami at the gate.",
        "Anh ấy gặp cô Amami ở cổng.",
        "Tomio gặp cô Amami ở cổng.",
    )
    retained, rejected = filter_source_contradicting_name_edits(source, [edit])
    assert retained == [edit]
    assert rejected == []


def test_translating_source_left_in_the_draft_is_not_removing_a_name():
    # Replayed over the runs, this is what the first version of the filter got
    # wrong: a draft that had copied the source keeps every capitalized word in
    # it, and translating the line has to lose them all.
    source = (
        "Tomio just sighed at me.\n"
        "“Stop getting blocked by mobs.” “Don’t lose by a nose.” “More fans.”\n"
    )
    edit = issue(
        "“Stop getting blocked by mobs.” “Don’t lose by a nose.” “More fans.”",
        "“Đừng để bị chặn bởi đám đông.” “Don’t lose by a nose.” “More fans.”",
        "“Đừng để bị chặn bởi đám đông.” “Đừng thua sát nút.” “Nhiều fan hơn.”",
    )
    retained, rejected = filter_source_contradicting_name_edits(source, [edit])
    assert retained == [edit]
    assert rejected == []


def test_an_edit_elsewhere_in_the_same_chunk_is_untouched():
    edits = [
        issue(
            "“Tomio, which one do you think Meek will go for?”",
            "Tomio, anh nghĩ Meek sẽ chọn cách nào?",
            "Trainer, anh nghĩ Meek sẽ chọn cách nào?",
            issue_id="ISSUE-01",
        ),
        issue(
            "Date or not, I just wanna spend Christmas with my trainer!",
            "tớ chỉ muốn đón Giáng sinh với huấn luyện viên!",
            "tớ chỉ muốn đón lễ Giáng sinh với huấn luyện viên!",
            issue_id="ISSUE-02",
        ),
    ]
    retained, rejected = filter_source_contradicting_name_edits(SOURCE, edits)
    assert rejected == ["ISSUE-01"]
    assert retained == [edits[1]]
