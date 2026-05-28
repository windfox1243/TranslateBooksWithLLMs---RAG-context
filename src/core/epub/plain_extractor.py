"""
Plain-text extraction and rebuild for Plain Text Mode (EPUB).

Walks an XHTML <body> in DOM order, collecting block-level paragraphs as plain
strings and anchoring any <img> they contain to their parent paragraph index.
The LLM never sees inline markup or images — only the textual content of each
block.

At rebuild time, the body is wiped and reconstructed as a flat sequence of
block elements (<p>, <h1..h6>, <li>, <blockquote>, <pre>) plus, after each
block that originally contained images, an extra <p class="plain-text-images"> wrapper
with the original <img> elements unchanged.
"""
from typing import Dict, List, Tuple

from lxml import etree


# Block-level tags we preserve at rebuild time (li flattens to p later — see replace_body_with_paragraphs).
BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre")
# Containers we descend into looking for blocks
CONTAINER_TAGS = ("div", "section", "article", "main", "header", "footer", "aside", "nav")
# Subtrees never sent to the LLM in Plain Text Mode
DROP_TAGS = ("table", "svg", "figure", "picture", "video", "audio", "iframe", "form", "script", "style")
# List wrappers we descend into (the inner <li> items become individual blocks)
LIST_WRAPPER_TAGS = ("ul", "ol")


def _local_name(elem: etree._Element) -> str:
    """Return the lowercase local tag name, stripping XHTML namespace."""
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return tag.lower() if isinstance(tag, str) else ""


def _extract_text_keep_inline(elem: etree._Element, image_sink: List[etree._Element]) -> str:
    """
    Flatten an element's textual content, ignoring inline tags.

    Adds any <img> encountered to image_sink (preserves DOM order).
    Returns whitespace-normalized text.
    """
    out: List[str] = []

    def walk(node: etree._Element, include_tail: bool):
        name = _local_name(node)
        if name in DROP_TAGS:
            # Skip subtree entirely. Still pick up its tail since it sits at
            # the parent's level.
            if include_tail and node.tail:
                out.append(node.tail)
            return
        if name == "img":
            image_sink.append(_clone_img(node))
            if include_tail and node.tail:
                out.append(node.tail)
            return
        if name == "br":
            out.append(" ")
            if include_tail and node.tail:
                out.append(node.tail)
            return
        if node.text:
            out.append(node.text)
        for child in node:
            walk(child, include_tail=True)
        if include_tail and node.tail:
            out.append(node.tail)

    if elem.text:
        out.append(elem.text)
    for child in elem:
        walk(child, include_tail=True)

    text = "".join(out)
    return " ".join(text.split())


def _clone_img(img: etree._Element) -> etree._Element:
    """Create a standalone copy of an <img> with its attributes, no namespace."""
    new = etree.Element("img")
    for k, v in img.attrib.items():
        if isinstance(k, str) and k.startswith("{"):
            k = k.split("}", 1)[1]
        new.set(k, v)
    return new


def _collect_blocks(
    root: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
) -> None:
    """
    DOM-walk a container, emitting one entry per block-level element found.

    For lists, we descend into <li> items individually (each is its own block).
    For containers (div, section, ...), we recurse.
    """
    for child in root:
        name = _local_name(child)

        if name in DROP_TAGS:
            continue

        if name in LIST_WRAPPER_TAGS:
            _collect_blocks(child, paragraphs_text, paragraphs_tag, images_by_paragraph)
            continue

        if name in CONTAINER_TAGS:
            _collect_blocks(child, paragraphs_text, paragraphs_tag, images_by_paragraph)
            continue

        if name in BLOCK_TAGS:
            images: List[etree._Element] = []
            if name == "pre":
                # Preserve code/pre verbatim — but skip <img> inside (rare)
                text = "".join(child.itertext())
            else:
                text = _extract_text_keep_inline(child, images)

            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append(name)
            if images:
                images_by_paragraph[idx] = images
            continue

        if name == "img":
            # Standalone <img> at body level — anchor to the previous block,
            # or create a synthetic anchor if it's first.
            img_copy = _clone_img(child)
            if paragraphs_text:
                anchor = len(paragraphs_text) - 1
                images_by_paragraph.setdefault(anchor, []).append(img_copy)
            else:
                paragraphs_text.append("")
                paragraphs_tag.append("p")
                images_by_paragraph[0] = [img_copy]
            continue

        # Anything else: try to extract textual content as a generic paragraph
        images: List[etree._Element] = []
        text = _extract_text_keep_inline(child, images)
        if text.strip() or images:
            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append("p")
            if images:
                images_by_paragraph[idx] = images


def extract_plain_paragraphs(
    body_element: etree._Element,
) -> Tuple[List[str], List[str], Dict[int, List[etree._Element]]]:
    """
    Extract the body as a flat list of (text, tag) pairs plus an image anchor map.

    Args:
        body_element: <body> element from a parsed XHTML doc.

    Returns:
        paragraphs_text:        list of plain-text strings, one per block
        paragraphs_tag:         parallel list of tag names ("p", "h1", "li", ...)
        images_by_paragraph:    {paragraph_index: [<img> elements]}
    """
    paragraphs_text: List[str] = []
    paragraphs_tag: List[str] = []
    images_by_paragraph: Dict[int, List[etree._Element]] = {}

    if body_element is None:
        return paragraphs_text, paragraphs_tag, images_by_paragraph

    _collect_blocks(body_element, paragraphs_text, paragraphs_tag, images_by_paragraph)
    return paragraphs_text, paragraphs_tag, images_by_paragraph


def replace_body_with_paragraphs(
    body_element: etree._Element,
    translated_paragraphs: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
    bilingual: bool = False,
    source_paragraphs: List[str] = None,
) -> None:
    """
    Wipe body_element and refill it from the translated paragraphs.

    Args:
        body_element: target <body> to overwrite
        translated_paragraphs: same length as paragraphs_tag
        paragraphs_tag: tag name per paragraph ("p", "h1", "li", ...)
        images_by_paragraph: anchored images per paragraph index
        bilingual: when True, emit a <p class="src"> with the source text
                   right before each translated block.
        source_paragraphs: required when bilingual is True
    """
    # Clear body
    body_element.text = None
    for child in list(body_element):
        body_element.remove(child)

    count = len(translated_paragraphs)
    for i in range(count):
        text = (translated_paragraphs[i] or "").strip()
        raw_tag = paragraphs_tag[i] if i < len(paragraphs_tag) else "p"
        # <li> outside <ul>/<ol> is not valid XHTML — flatten to <p> in Plain Text Mode.
        tag = "p" if raw_tag == "li" else raw_tag

        # Bilingual: emit source first when we have it
        if bilingual and source_paragraphs and i < len(source_paragraphs):
            source_text = (source_paragraphs[i] or "").strip()
            if source_text:
                src_block = etree.SubElement(body_element, tag)
                src_block.set("class", "plain-text-source")
                src_block.text = source_text

        # Emit translated block when there is text
        if text:
            block = etree.SubElement(body_element, tag)
            if bilingual:
                block.set("class", "plain-text-target")
            block.text = text

        # Emit anchored images right after
        if i in images_by_paragraph and images_by_paragraph[i]:
            img_wrapper = etree.SubElement(body_element, "p")
            img_wrapper.set("class", "plain-text-images")
            for img in images_by_paragraph[i]:
                img_wrapper.append(img)
