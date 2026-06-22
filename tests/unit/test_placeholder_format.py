from src.common.placeholder_format import PlaceholderFormat


def test_parse_and_matches_require_the_entire_placeholder():
    placeholder_format = PlaceholderFormat.from_config()

    assert placeholder_format.parse("[id42]") == 42
    assert placeholder_format.matches("[id42]")
    assert placeholder_format.parse("[id42]trailing") is None
    assert not placeholder_format.matches("[id42]trailing")


def test_renumber_does_not_rewrite_generated_placeholders():
    placeholder_format = PlaceholderFormat.from_config()

    text, mapping = placeholder_format.renumber("[id5]first[id0]second")

    assert text == "[id0]first[id1]second"
    assert mapping == {
        "[id5]": "[id0]",
        "[id0]": "[id1]",
    }
