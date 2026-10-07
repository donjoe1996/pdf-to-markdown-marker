"""Shared fixtures, and the isolation that makes the suite safe to run.

Several functions in this project have **global** side effects: ``stop_pids()``
runs ``pkill -f llama-server``, ``clear_stale_sentinels()`` deletes from the
real ``~/.cache/datalab/surya``, and the worker lock lives in the repo's
``output/``. A transcription worker may be processing books while these tests
run, so a test that touched any of that would destroy hours of real work.

The ``isolate`` fixture below is autouse: every test is redirected at a
temporary directory whether it asks or not. Tests that need to exercise process
handling spawn their own short-lived children and never signal a real pid.

Fixtures are also **synthetic**. Real pipeline output is book text, so it can be
neither committed nor used as a golden file; instead the builders here reproduce
marker's *structure* -- ``{N}----`` page separators, ``<sup>`` footnote markers,
running heads, hyphenation -- with placeholder prose. Test PDFs are drawn
programmatically, so the suite needs no binary fixtures at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from bt import jobs
from bt import queue as bt_queue
from bt import translate as bt_translate

PAGE_RULE = "-" * 48


def pytest_addoption(parser):
    parser.addoption(
        "--golden-update",
        action="store_true",
        default=False,
        help="rewrite golden files from current output (review the diff!)",
    )


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Point every piece of global state at a temp dir. Applied to all tests."""
    out_root = tmp_path / "output"
    out_root.mkdir()

    # surya sentinels -- clear_stale_sentinels() deletes files here
    sentinels = tmp_path / "surya"
    sentinels.mkdir()
    monkeypatch.setattr(jobs, "SURYA_SENTINEL_DIR", sentinels)

    # the worker lock -- claim/release would otherwise clobber a live worker's
    monkeypatch.setattr(jobs, "worker_lock_path", lambda root=None: out_root / ".worker.lock")

    # queue state: discovery root, output root, and the overlay file
    monkeypatch.setattr(bt_queue, "ROOT", tmp_path)
    monkeypatch.setattr(bt_queue, "OUTPUT_ROOT", out_root)
    monkeypatch.setattr(bt_queue, "QUEUE_FILE", out_root / "queue.json")

    # Outbound HTTP. The GUI lists a provider's models as soon as it has a key,
    # so a test that types one would otherwise reach api.groq.com for real --
    # slow, flaky offline, and spending someone's free tier from a test run.
    # Refusing here exercises the fallback path by default; a test that wants a
    # listing injects one through the transport argument.
    def _no_network(url, headers, timeout):
        raise bt_translate.TranslationError(f"no network in tests: {url}")

    monkeypatch.setattr(bt_translate, "_urllib_get", _no_network)

    return tmp_path


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "output" / "book"
    (d / "chunks").mkdir(parents=True)
    return d


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def marker_page(index: int, body: str) -> str:
    """One page as marker emits it: a ``{N}`` separator then the body.

    The separator shape matters -- a bare rule was assumed once and matched
    nothing, collapsing a whole book into a single page.
    """
    return f"{{{index}}}{PAGE_RULE}\n\n{body.strip()}\n"


def marker_markdown(pages: list[str], start: int = 0) -> str:
    return "\n".join(marker_page(i, body) for i, body in enumerate(pages, start=start))


@pytest.fixture
def book_markdown():
    """A short synthetic book: running heads, footnotes, a hyphen break."""

    def build(n_pages: int = 8) -> str:
        pages = []
        for i in range(1, n_pages + 1):
            pages.append(
                f"{i}  A Placeholder Title  II. {i % 3}\n\n"
                f"Body text unique to page {i}, long enough to read as a real "
                f"sentence rather than a label.<sup>1</sup>\n\n"
                f"A word split across the hyphen-\nbreak on page {i}.\n\n"
                f"<sup>1</sup> A note attached to page {i}.\n"
            )
        return marker_markdown(pages)

    return build


def _draw(page, rect, text):
    page.insert_textbox(rect, text, fontsize=9)


@pytest.fixture
def make_pdf(tmp_path):
    """Build small PDFs on the fly -- no binary fixtures in the repo."""

    def build(
        name: str,
        pages: int = 3,
        spread: bool = False,
        blank_left_on: tuple[int, ...] = (),
    ) -> Path:
        doc = pymupdf.open()
        for i in range(pages):
            if spread:
                # Landscape, with text in the outer thirds so the middle stays
                # blank -- the gutter the splitter must find.
                page = doc.new_page(width=800, height=500)
                if i not in blank_left_on:
                    _draw(page, pymupdf.Rect(40, 40, 330, 460), f"Left page {i}. " * 40)
                _draw(page, pymupdf.Rect(470, 40, 760, 460), f"Right page {i}. " * 40)
            else:
                page = doc.new_page(width=400, height=600)
                _draw(page, pymupdf.Rect(40, 40, 360, 560), f"Page {i}. " * 60)
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return build


@pytest.fixture
def make_scan_pdf(tmp_path):
    """Build a PDF that looks like a *scan*: every page one full-page image.

    This is the distinction ``analyze`` turns on. A scan's pages are photographs
    of paper, so any text on them came from somebody else's OCR however clean it
    reads -- which is why coverage, not the producer string, decides.

    ``text_layer=True`` adds extractable text over the image, reproducing a scan
    that has been OCRed by someone else.
    """

    def build(
        name: str,
        pages: int = 3,
        text_layer: bool = False,
        producer: str | None = None,
    ) -> Path:
        # Render some prose to a bitmap once, then stamp it on every page.
        scratch = pymupdf.open()
        drawn = scratch.new_page(width=400, height=600)
        _draw(drawn, pymupdf.Rect(30, 30, 370, 570), "Scanned line of text. " * 60)
        pixmap = drawn.get_pixmap(dpi=72)
        scratch.close()

        doc = pymupdf.open()
        for i in range(pages):
            page = doc.new_page(width=400, height=600)
            page.insert_image(page.rect, pixmap=pixmap)
            if text_layer:
                _draw(
                    page,
                    pymupdf.Rect(30, 30, 370, 570),
                    f"Recognised text for page {i}, as an OCR layer would carry. " * 8,
                )
        if producer:
            doc.set_metadata({"producer": producer})
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return build


@pytest.fixture
def make_bitonal_spread(tmp_path):
    """Build a spread the way a 300 DPI bitonal scanner stores one.

    Text is rendered at 300 DPI, thresholded to pure black and white, and
    placed as a single full-page image -- with a tight gutter and the dark
    binding shadow a flatbed leaves at the fold. Vector-text fixtures cannot
    reproduce the splitter's real failure: their strokes stay solid when the
    probe renders at low DPI, while a bitonal scan's thin strokes average out
    to light grey and vanish.

    Returns ``(path, (left_text_end, right_text_start))`` as fractions of width:
    a correct cut lies between the two.
    """

    def build(
        name: str = "bitonal.pdf", blank_left: bool = False
    ) -> tuple[Path, tuple[float, float]]:
        width, height = 800, 500
        left_end, fold, right_start = 372, 382, 392
        prose = "Through his thoughts and the gaze of his eyes, the offering. " * 30

        scratch = pymupdf.open()
        drawn = scratch.new_page(width=width, height=height)
        if not blank_left:
            drawn.insert_textbox(
                pymupdf.Rect(40, 40, left_end, 460), prose, fontsize=7
            )
        drawn.insert_textbox(
            pymupdf.Rect(right_start, 40, 760, 460), prose, fontsize=7
        )
        drawn.draw_line((fold, 0), (fold, height), width=2)
        grey = drawn.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
        scratch.close()

        samples = np.frombuffer(grey.samples, dtype=np.uint8)
        pixels = samples.reshape(grey.height, grey.stride)[:, : grey.width]
        bitonal = np.where(pixels < 128, 0, 255).astype(np.uint8)
        image = pymupdf.Pixmap(
            pymupdf.csGRAY, grey.width, grey.height, bitonal.tobytes(), False
        )

        doc = pymupdf.open()
        page = doc.new_page(width=width, height=height)
        page.insert_image(page.rect, pixmap=image)
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path, (left_end / width, right_start / width)

    return build


@pytest.fixture
def write_lock():
    """Write a run lock naming a source PDF, as jobs.start() does."""

    def build(folder: Path, pdf: Path, pid: int = 1, total_pages: int = 20) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / jobs.LOCK_NAME
        lock.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "started": 0.0,
                    "spec": {"pdf": str(pdf), "out_dir": str(folder),
                             "total_pages": total_pages, "chunk_size": 10},
                }
            ),
            encoding="utf-8",
        )
        return lock

    return build


@pytest.fixture
def make_chunks():
    """Create chunk files named the way transcribe.py names them."""

    def build(folder: Path, bounds: list[tuple[int, int]], text: str = "chunk") -> None:
        chunks = folder / "chunks"
        chunks.mkdir(parents=True, exist_ok=True)
        for first, last in bounds:
            (chunks / f"{first:04d}-{last:04d}.md").write_text(
                f"{text} {first}-{last}", encoding="utf-8"
            )

    return build
