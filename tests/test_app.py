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


def test_queue_panel_is_present(app):
    """The queue is how the unattended worker is understood from the GUI."""
    assert "Queue" in [s.value for s in app.subheader]


def test_document_picker_offers_upload(app):
    """A document can always be added, even with none on disk."""
    options = app.selectbox[0].options
    assert any("Upload" in o for o in options)


def test_switching_document_does_not_raise(app):
    """Changing the selection re-runs the whole script, analysis included."""
    pdfs = [o for o in app.selectbox[0].options if o.endswith(".pdf")]
    if not pdfs:
        pytest.skip("no PDFs available to select")
    app.selectbox[0].set_value(pdfs[0]).run()
    assert not app.exception, [str(e.value) for e in app.exception]
