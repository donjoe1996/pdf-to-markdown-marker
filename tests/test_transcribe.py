"""Chunk assembly.

Only the pure filesystem parts are covered. The OCR itself needs models and
hours, so it is deliberately outside the suite -- ``bt.verify`` checks that
against real output instead.
"""

from __future__ import annotations

import pytest

from bt.transcribe import _chunk_path, build_config, concatenate


def test_chunk_path_is_zero_padded(tmp_path):
    """Padding is what makes the filenames sort in page order."""
    assert _chunk_path(tmp_path, 0, 9).name == "0000-0009.md"
    assert _chunk_path(tmp_path, 100, 109).name == "0100-0109.md"


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


def test_config_extracts_figures_by_default():
    """A map or a plate is part of the document, not decoration."""
    assert build_config()["disable_image_extraction"] is False


def test_config_can_turn_figure_extraction_off():
    """Escape hatch: a noisy scan can have whole pages read as pictures."""
    assert build_config(images=False)["disable_image_extraction"] is True


# --------------------------------------------------------------------------
# the chunk loop's half of figure handling
# --------------------------------------------------------------------------
def _stub_marker(monkeypatch):
    """Stand in for marker.models so the chunk loop can be exercised.

    marker is never imported by the fast suite -- it needs models and hours.
    The loop around it is still ours, and the ordering it guarantees is worth
    pinning, so only the import surface is faked.
    """
    import sys
    import types

    models = types.ModuleType("marker.models")
    models.create_model_dict = lambda: {}
    models.shutdown_models = lambda artifacts: None
    pkg = sys.modules.get("marker") or types.ModuleType("marker")
    monkeypatch.setitem(sys.modules, "marker", pkg)
    monkeypatch.setitem(sys.modules, "marker.models", models)


def test_figures_are_saved_before_the_chunk_that_references_them(monkeypatch, tmp_path):
    """Order matters, and only in one direction.

    A chunk's ``.md`` is the record that its pages are done -- resume skips a
    chunk that exists. If the Markdown landed first and the run died before the
    figures were written, the book would be permanently missing them and no
    re-run would notice.
    """
    from PIL import Image

    from bt import transcribe as t

    _stub_marker(monkeypatch)
    out_md = tmp_path / "raw.md"
    chunk_dir = tmp_path / "chunks"
    chunk = chunk_dir / "0000-0001.md"

    class WatchfulImage:
        """A figure that refuses to be written after its chunk is complete."""

        mode = "RGB"

        def save(self, path):
            assert not chunk.exists(), "the chunk was marked done before its figure"
            Image.new("RGB", (2, 2)).save(path)

    monkeypatch.setattr(t, "make_converter", lambda config, artifacts: object())
    monkeypatch.setattr(
        t,
        "convert_range",
        lambda converter, pdf: (
            "![](_page_0_Picture_0.jpeg)",
            {"_page_0_Picture_0.jpeg": WatchfulImage()},
        ),
    )

    t.transcribe(tmp_path / "book.pdf", out_md, chunk_dir, total_pages=2, chunk_size=2)

    assert (tmp_path / "images" / "0000-0001_page_0_Picture_0.jpeg").exists()
    assert "![](images/0000-0001_page_0_Picture_0.jpeg)" in out_md.read_text()
