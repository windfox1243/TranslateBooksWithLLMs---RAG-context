"""Fold HTML <ruby> annotations into plain text before translation (issue #242).

Ruby markup carries one word in two halves: a base and a reading. In Japanese
and Korean books the reading is often not a pronunciation at all but a second
layer of meaning the author wants read alongside the base (a contextual reading,
a loanword, a character's inner thought, an invented term).

Both extraction paths used to hand those halves to the model as separate units:

    structured mode   <p>彼は<ruby>宇宙<rt>そら</rt></ruby>を見上げた。</p>
                      → "[id0]彼は[id1]宇宙[id2]そら[id3]を見上げた。[id4]"
                        two one-word fragments the model must translate blind,
                        with placeholders pinning them in place

    plain-text mode   → "彼は宇宙そらを見上げた。"
                        base and reading glued into a word that does not exist

Folding the subtree into 宇宙（そら） keeps both halves and makes them one
contiguous string, so the model reads them together and can render the intended
meaning. This is the fallback the reporter asked for, and the form <rp> already
provides for legacy browsers.

Full-width parentheses are used deliberately: the text is CJK at fold time, and
the distinctive form reads as an annotation rather than an ordinary aside. When
the source supplies its own <rp> delimiters they are kept verbatim instead.

Documents without ruby markup are left untouched, so this is a no-op for the
vast majority of books.
"""
from typing import Callable, List, Optional

from lxml import etree

RUBY_TAG = "ruby"
# <rt> holds the reading. <rtc> is the double-sided-ruby container; its own <rt>
# descendants come along with its text, which merges both annotation layers into
# one reading — a graceful degradation for a form that is rare in EPUBs.
READING_TAGS = ("rt", "rtc")
# <rp> is the source's own parenthesis fallback. Its text IS the delimiter.
DELIMITER_TAG = "rp"
# Everything else inside <ruby> is base text, including the explicit <rb>
# segments that group ruby uses several of in a row.

OPEN_PAREN = "（"
CLOSE_PAREN = "）"


def _local_name(element: etree._Element) -> str:
    """Return the lowercase local tag name, stripping any namespace.

    Returns "" for comments and processing instructions, whose .tag is not a str.
    """
    tag = element.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return tag.lower()


def _inner_text(element: etree._Element) -> str:
    """Flatten an element's textual content, including nested markup."""
    return "".join(element.itertext())


def _fold_text(ruby: etree._Element) -> str:
    """Return the plain-text replacement for one <ruby> subtree.

    Text is collected in document order, so an <rp>(</rp><rt>x</rt><rp>)</rp>
    run yields the reading "(x)" already delimited by the source.
    """
    base: List[str] = []
    reading: List[str] = []
    has_delimiters = False
    has_reading = False

    if ruby.text:
        base.append(ruby.text)

    for child in ruby:
        name = _local_name(child)
        if not name:
            # Comment or processing instruction: no text of its own, and
            # itertext() rejects it outright.
            pass
        elif name == DELIMITER_TAG:
            has_delimiters = True
            reading.append(_inner_text(child))
        elif name in READING_TAGS:
            text = _inner_text(child)
            if text.strip():
                has_reading = True
            reading.append(text)
        else:
            # <rb>, or inline markup wrapping the base (<em>, <span>, a nested
            # <ruby>): its text belongs to the base.
            base.append(_inner_text(child))
        # A tail sits at the ruby's own level, alongside the base.
        if child.tail:
            base.append(child.tail)

    folded_base = "".join(base)

    # No actual reading means nothing to annotate with — drop any stray <rp>
    # delimiters rather than emitting an empty "base()".
    if not has_reading:
        return folded_base
    if has_delimiters:
        # Keep the source's own delimiters rather than imposing ours.
        return folded_base + "".join(reading)
    return f"{folded_base}{OPEN_PAREN}{''.join(reading)}{CLOSE_PAREN}"


def _outermost_rubies(body: etree._Element) -> List[etree._Element]:
    """Collect the <ruby> elements that no other <ruby> contains.

    A nested annotation's text is already folded into its ancestor's base, and
    the ancestor takes the whole subtree with it when detached. The list is
    computed before any mutation, so the ancestor walks are done on the intact
    tree.
    """
    rubies = [e for e in body.iter() if _local_name(e) == RUBY_TAG]
    if len(rubies) < 2:
        return rubies

    inner = set()
    for ruby in rubies:
        ancestor = ruby.getparent()
        while ancestor is not None and ancestor is not body:
            if _local_name(ancestor) == RUBY_TAG:
                inner.add(id(ruby))
                break
            ancestor = ancestor.getparent()

    return [r for r in rubies if id(r) not in inner]


def _replace_with_text(element: etree._Element, text: str) -> None:
    """Detach an element, merging `text` plus its tail into the surrounding text."""
    parent = element.getparent()
    if parent is None:
        return

    merged = text + (element.tail or "")
    previous = element.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or "") + merged
    else:
        parent.text = (parent.text or "") + merged

    parent.remove(element)


def fold_ruby_annotations(
    body: Optional[etree._Element],
    log_callback: Optional[Callable] = None,
) -> int:
    """Replace every <ruby> subtree with its folded text, in place.

    Returns the number of annotations folded. Idempotent: no <ruby> element
    survives the pass, so a second call is a no-op.
    """
    if body is None:
        return 0

    rubies = _outermost_rubies(body)
    if not rubies:
        return 0

    for ruby in rubies:
        _replace_with_text(ruby, _fold_text(ruby))
    folded = len(rubies)

    if folded and log_callback:
        log_callback(
            "ruby_annotations_folded",
            f"🈁 Folded {folded} ruby annotation(s) into base（reading） form",
        )

    return folded
