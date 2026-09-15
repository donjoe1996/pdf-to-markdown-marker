"""Chunk assembly.

Only the pure filesystem parts are covered. The OCR itself needs models and
hours, so it is deliberately outside the suite -- ``bt.verify`` checks that
against real output instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bt.transcribe import (
    build_config,
    chunk_path,
    concatenate,
    rewrite_image_links,
    save_images,
)


def test_chunk_path_is_zero_padded(tmp_path):
    """Padding is what makes the filenames sort in page order."""
    assert chunk_path(tmp_path, 0, 9).name == "0000-0009.md"
    assert chunk_path(tmp_path, 100, 109).name == "0100-0109.md"


def test_concatenate_joins_in_page_order(tmp_path, make_chunks):
    make_chunks(tmp_path, [(10, 19), (0, 9)])
    out = tmp_path / "raw.md"
    concatenate(tmp_path / "chunks", out, [(0, 9), (10, 19)])

    text = out.read_text()
    assert text.index("chunk 0-9") < text.index("chunk 10-19")


def test_concatenate_excludes_stale_chunks(tmp_path, make_chunks):
    """REGRESSION: concatenate() used to glob the whole chunk directory.

    Re-running with a different --chunk-size leaves chunks with different
    boundaries behind, so globbing spliced overlapping page ranges together and
    produced a book with duplicated passages.
    """
    make_chunks(tmp_path, [(0, 19), (0, 9), (10, 19)])  # 0-19 is from an older run
    out = tmp_path / "raw.md"
    concatenate(tmp_path / "chunks", out, [(0, 9), (10, 19)])

    text = out.read_text()
    assert "chunk 0-19" not in text
    assert "chunk 0-9" in text and "chunk 10-19" in text


def test_concatenate_refuses_when_a_chunk_is_missing(tmp_path, make_chunks):
    make_chunks(tmp_path, [(0, 9)])
    with pytest.raises(RuntimeError, match="missing chunk output"):
        concatenate(tmp_path / "chunks", tmp_path / "raw.md", [(0, 9), (10, 19)])


def test_concatenate_fallback_ignores_unrelated_markdown(tmp_path, make_chunks):
    """Without bounds it must still not swallow a stray .md file."""
    make_chunks(tmp_path, [(0, 9), (10, 19)])
    (tmp_path / "chunks" / "notes.md").write_text("personal notes", encoding="utf-8")

    out = tmp_path / "raw.md"
    concatenate(tmp_path / "chunks", out, None)
    assert "personal notes" not in out.read_text()


# --------------------------------------------------------------------------
# the OCR / extraction switch
# --------------------------------------------------------------------------
def test_config_forces_ocr_by_default():
    """The embedded layer of a scan is someone else's OCR; never trust it."""
    cfg = build_config("0-9")
    assert cfg["force_ocr"] is True
    assert cfg["strip_existing_ocr"] is True
    assert "disable_ocr" not in cfg


def test_config_extraction_mode_disables_ocr():
    cfg = build_config("0-9", ocr=False)
    assert cfg["disable_ocr"] is True
    assert "force_ocr" not in cfg


def test_config_always_sets_output_format():
    """ConfigParser raises a bare KeyError without it."""
    assert build_config()["output_format"] == "markdown"


def test_config_paginates_so_pages_stay_addressable():
    """Downstream code splits on the {N} separators this produces."""
    assert build_config()["paginate_output"] is True


# --------------------------------------------------------------------------
# figures and images
# --------------------------------------------------------------------------
class FakeImage:
    """Stands in for the PIL image marker hands back."""

    def __init__(self) -> None:
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        Path(path).write_bytes(b"\x89PNG fake")


def test_images_are_namespaced_by_chunk(tmp_path):
    """marker restarts page numbering inside each converted range.

    Two chunks therefore both offer a `_page_3_Figure_2.jpeg`. Written under
    that name, the second silently overwrites the first and both chunks'
    Markdown ends up pointing at the same picture -- a wrong figure on a page
    that still reads perfectly. The chunk bounds make the name unique.
    """
    images = tmp_path / "images"
    first = save_images({"_page_3_Figure_2.jpeg": FakeImage()}, images, "0000-0009")
    second = save_images({"_page_3_Figure_2.jpeg": FakeImage()}, images, "0010-0019")

    assert first != second
    assert sorted(p.name for p in images.iterdir()) == [
        "0000-0009_page_3_Figure_2.jpeg",
        "0010-0019_page_3_Figure_2.jpeg",
    ]


def test_image_links_are_rewritten_to_the_saved_name(tmp_path):
    md = "Text.\n\n![](_page_3_Figure_2.jpeg)\n\nFigura 2. Patrón anual.\n"
    mapping = save_images(
        {"_page_3_Figure_2.jpeg": FakeImage()}, tmp_path / "images", "0000-0009"
    )

    out = rewrite_image_links(md, mapping, "images")
    assert "![](images/0000-0009_page_3_Figure_2.jpeg)" in out
    assert "_page_3_Figure_2.jpeg)" not in out.replace(
        "images/0000-0009_page_3_Figure_2.jpeg", ""
    )


def test_unknown_image_link_is_left_alone(tmp_path):
    """A link marker did not produce is not ours to rewrite."""
    md = "![alt](https://example.com/a.png)\n"
    assert rewrite_image_links(md, {}, "images") == md


def test_config_extracts_images_only_when_asked():
    assert build_config()["disable_image_extraction"] is True
    assert build_config(extract_images=True)["disable_image_extraction"] is False
