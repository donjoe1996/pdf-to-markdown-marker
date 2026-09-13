"""Extracted figures: saving them, and keeping the links pointing at them.

The failure this module exists to prevent is silent. marker hands back its
images in a dict keyed by names it invents per conversion, and a chunked run
converts the same document a dozen times. If two chunks ever produce the same
key, the second write overwrites the first and the finished book shows the
wrong plate under the right caption -- readable, plausible, and wrong.

So nothing here trusts marker's numbering: every chunk's images are namespaced
by the chunk that produced them, which is collision-proof by construction
rather than by assumption.
"""

from __future__ import annotations

from PIL import Image

from bt.images import save_chunk_images, stage_chunk


def _img(colour: str = "red", mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (4, 4), color=colour)


def test_saves_the_image_and_repoints_the_link(tmp_path):
    md = "Some text.\n\n![](_page_3_Picture_0.jpeg)\n"
    out = stage_chunk(md, {"_page_3_Picture_0.jpeg": _img()}, tmp_path / "images", "0000-0019")

    saved = tmp_path / "images" / "0000-0019_page_3_Picture_0.jpeg"
    assert saved.exists()
    assert "![](images/0000-0019_page_3_Picture_0.jpeg)" in out
    assert "_page_3_Picture_0.jpeg)" not in out.replace(saved.name, "")


def test_two_chunks_with_the_same_marker_name_keep_both_images(tmp_path):
    """The collision this module is for.

    marker may number pictures from the start of each *conversion*, so chunk
    two can hand back "_page_0_Picture_0.jpeg" for a completely different
    figure. Namespacing by chunk keeps the two apart whether it does or not.
    """
    image_dir = tmp_path / "images"
    name = "_page_0_Picture_0.jpeg"
    a = stage_chunk(f"![]({name})", {name: _img("red")}, image_dir, "0000-0019")
    b = stage_chunk(f"![]({name})", {name: _img("blue")}, image_dir, "0020-0039")

    files = sorted(p.name for p in image_dir.iterdir())
    assert files == ["0000-0019_page_0_Picture_0.jpeg", "0020-0039_page_0_Picture_0.jpeg"]
    assert files[0] in a and files[1] in b
    assert Image.open(image_dir / files[0]).getpixel((0, 0)) != Image.open(
        image_dir / files[1]
    ).getpixel((0, 0)), "the second chunk overwrote the first chunk's figure"


def test_alt_text_and_captions_survive_the_rewrite(tmp_path):
    md = "![Map of the fire belt](_page_9_Figure_2.png)"
    out = stage_chunk(md, {"_page_9_Figure_2.png": _img()}, tmp_path / "images", "0000-0009")
    assert out == "![Map of the fire belt](images/0000-0009_page_9_Figure_2.png)"


def test_a_link_marker_did_not_hand_back_is_left_alone(tmp_path):
    """Rewriting only what was actually saved keeps links honest.

    A reference with no image behind it is a broken link either way, but
    inventing a path for it would hide that from verify.check_images.
    """
    md = "![](missing.jpeg)"
    assert stage_chunk(md, {}, tmp_path / "images", "0000-0009") == md


def test_a_name_that_would_escape_the_directory_is_neutralised(tmp_path):
    """Nothing marker names should ever be written outside the image dir."""
    image_dir = tmp_path / "images"
    saved = save_chunk_images({"../../evil.jpeg": _img()}, image_dir, "0000-0009")

    written = list(image_dir.iterdir())
    assert len(written) == 1
    assert written[0].parent == image_dir
    assert ".." not in written[0].name
    assert saved["../../evil.jpeg"] == f"images/{written[0].name}"


def test_transparency_does_not_break_a_jpeg_write(tmp_path):
    """PIL refuses to write RGBA as JPEG; a figure with alpha must not kill a chunk."""
    stage_chunk("![](_page_1_Picture_0.jpeg)", {"_page_1_Picture_0.jpeg": _img(mode="RGBA")},
                tmp_path / "images", "0000-0009")
    assert (tmp_path / "images" / "0000-0009_page_1_Picture_0.jpeg").exists()


def test_no_images_writes_no_directory(tmp_path):
    """A text-only book should not sprout an empty images/ folder."""
    assert stage_chunk("plain text", {}, tmp_path / "images", "0000-0009") == "plain text"
    assert not (tmp_path / "images").exists()


def test_writes_are_atomic(tmp_path):
    """Chunk files are written temp-then-rename; their images must be too.

    A chunk .md is the record that its pages are done. If a half-written image
    could survive a kill, resume would skip the chunk and the book would carry
    a truncated figure for good.
    """
    image_dir = tmp_path / "images"
    save_chunk_images({"_page_0_Picture_0.jpeg": _img()}, image_dir, "0000-0009")
    assert not list(image_dir.glob("*partial*"))
