"""Locate the package document (OPF) of an extracted EPUB.

The OCF spec names it in META-INF/container.xml; every pass that edits the
OPF must find the same file. (Ported from upstream 835130df and 4c29abbb.)
"""
import os
from typing import Optional
from urllib.parse import unquote

from lxml import etree

OPF_MEDIA_TYPE = 'application/oebps-package+xml'
_CONTAINER_NS = 'urn:oasis:names:tc:opendocument:xmlns:container'


def find_opf_file(temp_dir: str) -> Optional[str]:
    """The package document of an extracted EPUB.

    The one ``META-INF/container.xml`` names (``rootfile/@full-path`` of its
    first ``application/oebps-package+xml`` rootfile), as the OCF spec
    requires. Falls back to the first ``*.opf`` in directory walk order when
    the container is missing, unreadable, names no package or names a file
    that does not exist inside the book.
    """
    return opf_from_container(temp_dir) or _first_opf_in_tree(temp_dir)


def opf_from_container(temp_dir: str) -> Optional[str]:
    """The package ``META-INF/container.xml`` names, if it exists in the book."""
    container_path = os.path.join(temp_dir, 'META-INF', 'container.xml')
    if not os.path.isfile(container_path):
        return None
    try:
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
        root = etree.parse(container_path, parser).getroot()
    except (etree.XMLSyntaxError, OSError):
        return None
    if root is None:
        return None
    for rootfile in root.iter(f'{{{_CONTAINER_NS}}}rootfile', 'rootfile'):
        if rootfile.get('media-type') != OPF_MEDIA_TYPE:
            continue
        full_path = rootfile.get('full-path')
        if not full_path:
            return None
        candidate = os.path.normpath(os.path.join(temp_dir, unquote(full_path)))
        base = os.path.abspath(temp_dir)
        try:
            inside = os.path.commonpath([base, os.path.abspath(candidate)]) == base
        except ValueError:  # different drives
            inside = False
        return candidate if inside and os.path.isfile(candidate) else None
    return None


def _first_opf_in_tree(temp_dir: str) -> Optional[str]:
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.opf'):
                return os.path.join(root_dir, file)
    return None
