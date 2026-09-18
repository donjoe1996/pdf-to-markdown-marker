"""The Streamlit app executes without raising.

Worth its own test because the obvious check is misleading: the server answers
HTTP 200 with the page shell before the script runs, so a script that raises
still looks healthy from outside. ``AppTest`` executes it in-process and
surfaces the exception.

This is a smoke test, not a UI test. It asserts the page renders and the pieces
that matter are present -- not how they look.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# Absolute: AppTest resolves a relative path against the caller's directory,
# which is tests/, not the project root.
APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120

# Each run boots Streamlit and analyses whatever PDFs are on disk, so this file
# takes tens of seconds against a real library. Excluded from the default run
# and exercised in CI; run locally with `pytest -m slow`.
pytestmark = pytest.mark.slow


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def has_documents(at) -> bool:
    """Whether the picker offers any document.

    Addressed through the sidebar rather than by global index: the result page
    adds selectboxes of its own, so `at.selectbox[0]` is not stable.
    """
    return any(o.endswith(".pdf") for o in at.sidebar.selectbox[0].options)


def test_queue_panel_is_present(make_pdf):
    """The queue is how the unattended worker is understood from the GUI.

    Only rendered once a document is selected: with an empty library the script
    stops early, which is the correct behaviour and is covered separately below.
    """
    make_pdf("queued.pdf", pages=1)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "Queue" in [s.value for s in at.subheader]


def test_empty_library_degrades_gracefully(app):
    """A library with no PDFs at all -- the state every new install starts in.

    The app must explain itself rather than render a broken half-page or raise.
    """
    assert not has_documents(app)
    assert not app.exception
    assert any("Choose a PDF" in i.value for i in app.info)
    assert any("Upload" in o for o in app.sidebar.selectbox[0].options)


def test_document_picker_offers_upload(app):
    """A document can always be added, even with none on disk."""
    options = app.sidebar.selectbox[0].options
    assert any("Upload" in o for o in options)


def test_worker_button_starts_and_echoes_the_command(monkeypatch, make_pdf):
    """The start button shows what it ran, so a click is never a mystery.

    ``start_worker`` is stubbed: a real one would launch ``bt.worker`` against
    the repo and start transcribing.
    """
    from bt import jobs

    calls = []
    monkeypatch.setattr(jobs, "start_worker", lambda: calls.append(1) or jobs.worker_command())
    make_pdf("worker.pdf", pages=1)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()

    start = next(b for b in at.button if b.label == "Start worker")
    start.click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert calls == [1]
    assert any("-m bt.worker" in c.value for c in at.code)


def test_switching_document_does_not_raise(make_pdf):
    """Changing the selection re-runs the whole script, analysis included."""
    make_pdf("first.pdf", pages=1)
    make_pdf("second.pdf", pages=1)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.selectbox[0].set_value("second.pdf").run()
    assert not at.exception, [str(e.value) for e in at.exception]


# --------------------------------------------------------------------------
# the result page: figures, and the translate panel
# --------------------------------------------------------------------------
def _write_png(path: Path) -> None:
    """A real 1x1 PNG -- Streamlit decodes what it renders."""
    import pymupdf

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1), False)
    path.write_bytes(pixmap.tobytes("png"))


@pytest.fixture
def finished_book(tmp_path, make_pdf):
    """A document with output already on disk, as after a completed run.

    Discovery goes through ``bt.queue``, which conftest's autouse ``isolate``
    fixture points at tmp_path -- so this builds a whole library without
    touching the real one, where a running worker would otherwise pick the
    fixture up and start transcribing it.
    """
    pdf = make_pdf("book.pdf", pages=2)
    out = tmp_path / "output" / "book"
    (out / "chunks").mkdir(parents=True)
    (out / "book.md").write_text(
        "<!-- page 0 -->\n\nUna página en español.\n\n"
        "![](images/0000-0009_page_0_Figure_1.jpeg)\n\n"
        "<!-- page 1 -->\n\nOtra página.\n",
        encoding="utf-8",
    )
    images = out / "images"
    images.mkdir()
    _write_png(images / "0000-0009_page_0_Figure_1.jpeg")
    return pdf, out


def _open(at, pdf):
    at.run()
    at.sidebar.selectbox[0].set_value(pdf.name).run()
    return at


def test_result_and_translate_sections_render(finished_book):
    """Everything below "Result" only exists once a run has produced output.

    With no PDFs in the repo the other smoke tests skip before reaching it, so
    this half of the page was never executed -- and AppTest is the only thing
    that catches a NameError in a branch the happy path never takes.
    """
    pdf, _ = finished_book
    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)

    assert not at.exception, [str(e.value) for e in at.exception]
    headings = [s.value for s in at.subheader]
    assert "Result" in headings and "Translate" in headings
    assert any("Figures" in e.label for e in at.expander)


def test_translate_button_launches_a_detached_job(finished_book, monkeypatch):
    """The click must hand a spec to jobs, not translate inside the app.

    ``start_translate`` is stubbed: a real one would spend hours and money.
    """
    from bt import jobs

    pdf, out = finished_book
    launched = []
    monkeypatch.setattr(
        jobs,
        "start_translate",
        lambda spec, folder, api_key="": launched.append((spec, folder))
        or jobs.JobStatus(),
    )

    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)
    button = next(b for b in at.button if b.label.startswith("Translate to"))
    button.click().run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(launched) == 1
    spec, folder = launched[0]
    assert spec.src.endswith("book.md")
    assert spec.target == "English"
    assert Path(folder) == out
    # Its chunks must not land in the OCR chunk directory: queue.survey() counts
    # files there to decide a book is finished.
    assert spec.chunk_dir != out / "chunks"


def _key_field(at):
    """The API key box, addressed by label rather than by index.

    The Translate panel holds several text inputs and the page above it holds
    more; a positional index here would break on the next layout change.
    """
    return next(t for t in at.text_input if "API key" in t.label)


def test_a_pasted_key_is_handed_to_the_job_and_not_kept_on_disk(
    finished_book, monkeypatch
):
    """The point of the field: a key typed in the browser must reach the run.

    The GUI holds no durable state by design, so the key lives only in this
    session and is handed to ``start_translate`` at launch. ``jobs`` then puts
    it in the child's environment -- never the command line, never the lock
    file (see test_jobs.py).
    """
    from bt import jobs

    pdf, _out = finished_book
    launched = []
    monkeypatch.setattr(
        jobs,
        "start_translate",
        lambda spec, folder, api_key="": launched.append((spec, api_key))
        or jobs.JobStatus(),
    )

    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)
    provider = next(s for s in at.selectbox if s.label == "Provider")
    provider.set_value("groq").run()
    _key_field(at).set_value("gsk-pasted-in-the-browser").run()
    next(b for b in at.button if b.label.startswith("Translate to")).click().run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert launched and launched[0][1] == "gsk-pasted-in-the-browser"
    # Nothing on the spec: it is serialised into .translate.lock verbatim.
    assert "gsk-pasted-in-the-browser" not in str(launched[0][0])


def test_the_key_field_is_per_provider(finished_book):
    """REGRESSION RISK: one shared field would send groq's key to gemini.

    Switching providers must not carry the previous key across -- the request
    would be rejected, and a secret would have been sent to a service it was
    not issued for.
    """
    pdf, _out = finished_book
    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)
    provider = next(s for s in at.selectbox if s.label == "Provider")
    provider.set_value("groq").run()
    _key_field(at).set_value("gsk-for-groq").run()
    assert _key_field(at).value == "gsk-for-groq"

    next(s for s in at.selectbox if s.label == "Provider").set_value("gemini").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert _key_field(at).value == ""


def test_a_keyless_provider_offers_no_key_field(finished_book):
    """`local` and `ollama` need no account; a key box there is only confusing."""
    pdf, _out = finished_book
    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)
    next(s for s in at.selectbox if s.label == "Provider").set_value("local").run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert not [t for t in at.text_input if "API key" in t.label]


def test_an_output_folder_outside_the_project_does_not_crash(finished_book, tmp_path):
    """REGRESSION: the Result caption used a strict relative_to().

    Pointing "Output folder" at an absolute path elsewhere raised ValueError
    and took the whole page down.
    """
    pdf, out = finished_book
    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)
    at.sidebar.text_input[0].set_value(str(out)).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_a_corrupt_figure_does_not_take_the_page_down(finished_book):
    """An image that cannot be decoded is a damaged file, not a page error.

    Streamlit raises UnidentifiedImageError from st.image, which would abort
    the whole script -- losing the Result section, the download button and the
    translate panel over one bad figure.
    """
    pdf, out = finished_book
    (out / "images" / "0000-0009_page_1_Figure_9.jpeg").write_bytes(b"not an image")

    at = _open(AppTest.from_file(APP, default_timeout=TIMEOUT), pdf)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("could not be read" in w.value for w in at.warning)

