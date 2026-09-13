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
    return any(o.endswith(".pdf") for o in at.selectbox[0].options)


def test_queue_panel_is_present(app):
    """The queue is how the unattended worker is understood from the GUI.

    Only rendered once a document is selected: with an empty library the script
    stops early, which is the correct behaviour and is covered separately below.
    """
    if not has_documents(app):
        pytest.skip("no documents; the empty-library path is tested separately")
    assert "Queue" in [s.value for s in app.subheader]


def test_empty_library_degrades_gracefully(app):
    """CI runs with no PDFs at all, since they are gitignored.

    That is a path a local run never takes, so it is worth pinning: the app must
    explain itself rather than render a broken half-page or raise.
    """
    if has_documents(app):
        pytest.skip("library is not empty here; CI covers this path")
    assert not app.exception
    assert any("Choose a PDF" in i.value for i in app.info)
    assert any("Upload" in o for o in app.selectbox[0].options)


def test_document_picker_offers_upload(app):
    """A document can always be added, even with none on disk."""
    options = app.selectbox[0].options
    assert any("Upload" in o for o in options)


def test_worker_button_starts_and_echoes_the_command(monkeypatch):
    """The start button shows what it ran, so a click is never a mystery.

    ``start_worker`` is stubbed: a real one would launch ``bt.worker`` against
    the repo and start transcribing.
    """
    from bt import jobs

    calls = []
    monkeypatch.setattr(jobs, "start_worker", lambda: calls.append(1) or jobs.worker_command())
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if not has_documents(at):
        pytest.skip("the queue panel needs a document")

    start = next(b for b in at.button if b.label == "Start worker")
    start.click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert calls == [1]
    assert any("-m bt.worker" in c.value for c in at.code)


def test_switching_document_does_not_raise(app):
    """Changing the selection re-runs the whole script, analysis included."""
    pdfs = [o for o in app.selectbox[0].options if o.endswith(".pdf")]
    if not pdfs:
        pytest.skip("no PDFs available to select")
    app.selectbox[0].set_value(pdfs[0]).run()
    assert not app.exception, [str(e.value) for e in app.exception]


# --------------------------------------------------------------------------
# PUBLIC_MODE: the login gate the Hugging Face Spaces deployment sets
# BT_PUBLIC_MODE=1 for. Off by default, so every test above already covers
# that this feature changes nothing for local, personal use.
# --------------------------------------------------------------------------
def _public_app(monkeypatch) -> AppTest:
    monkeypatch.setenv("BT_PUBLIC_MODE", "1")
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


def test_public_mode_shows_login_gate_when_logged_out(monkeypatch):
    at = _public_app(monkeypatch)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # The ordinary document picker and queue must not be reachable pre-login.
    assert not at.selectbox
    assert "Queue" not in [s.value for s in at.subheader]
    assert any("New here" in t.label for t in at.tabs)


def test_public_mode_signup_shows_a_one_time_code(monkeypatch):
    at = _public_app(monkeypatch)
    at.run()
    signup = at.tabs[0]
    signup.text_input[0].set_value("newuser")
    signup.text_input[1].set_value("newuser@example.com")
    signup.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("one-time login code" in s.value for s in at.success)


def test_public_mode_login_rejects_wrong_code(monkeypatch):
    from bt import auth

    auth.signup("someone", "someone@example.com")
    at = _public_app(monkeypatch)
    at.run()
    login = at.tabs[1]
    login.text_input[0].set_value("someone")
    login.text_input[1].set_value("000000")
    login.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Wrong username or code" in e.value for e in at.error)
    assert "user" not in at.session_state


def test_public_mode_logged_in_hides_queue(monkeypatch):
    """A signed-in user with nothing uploaded yet sees the upload prompt, not

    the Queue panel -- that panel lists and can launch unattended processing
    across every account's uploads, which has no way to respect the
    per-account page cap.
    """
    at = _public_app(monkeypatch)
    at.session_state["user"] = "someone"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "Queue" not in [s.value for s in at.subheader]
    assert any("Signed in as" in c.value for c in at.caption)
    assert any("Choose a PDF" in i.value for i in at.info)
