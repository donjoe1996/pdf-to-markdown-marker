"""Queue state.

State is derived from disk rather than tracked, so these tests mostly build a
directory layout and assert what the queue concludes from it.
"""

from __future__ import annotations

from bt import queue as bt_queue
from bt.queue import MAX_ATTEMPTS, DocState


def state(**kw) -> DocState:
    base = {"pdf": bt_queue.ROOT / "a.pdf", "out_dir": bt_queue.OUTPUT_ROOT / "a"}
    return DocState(**{**base, **kw})


# --------------------------------------------------------------------------
# status derivation
# --------------------------------------------------------------------------
def test_status_transitions():
    assert state(chunks_done=0, chunks_total=5).status == "pending"
    assert state(chunks_done=2, chunks_total=5).status == "in progress"
    assert state(chunks_done=5, chunks_total=5).status == "done"
    assert state(chunks_done=6, chunks_total=5).status == "done"
    assert state(chunks_done=1, chunks_total=5, skip=True).status == "skipped"
    assert state(chunks_done=1, chunks_total=5, attempts=MAX_ATTEMPTS).status == "stalled"


def test_unknown_total_is_never_done():
    """A document whose page count is unknown must not look finished."""
    s = state(chunks_done=0, chunks_total=0)
    assert s.done is False
    assert s.fraction == 0.0


def test_fraction_is_capped():
    assert state(chunks_done=9, chunks_total=5).fraction == 1.0


# --------------------------------------------------------------------------
# which folder belongs to which document
# --------------------------------------------------------------------------
def test_folder_belongs_only_to_its_own_pdf(tmp_path, write_lock, make_chunks):
    """REGRESSION: a folder was adopted merely because it had chunks.

    Chunk files are named by page number alone and say nothing about their
    source, so selecting a second book once claimed the first book's output --
    which would have resumed one book onto another's pages.
    """
    folder = bt_queue.OUTPUT_ROOT
    mine = tmp_path / "mine.pdf"
    theirs = tmp_path / "theirs.pdf"
    mine.write_bytes(b"%PDF-1.4\n")
    theirs.write_bytes(b"%PDF-1.4\n")

    make_chunks(folder, [(0, 9)])
    write_lock(folder, mine)

    assert bt_queue.folder_belongs_to(folder, mine) is True
    assert bt_queue.folder_belongs_to(folder, theirs) is False


def test_resolve_out_dir_prefers_the_per_document_folder(tmp_path, make_chunks):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    per_doc = bt_queue.OUTPUT_ROOT / "book"
    make_chunks(per_doc, [(0, 9)])

    assert bt_queue.resolve_out_dir(pdf) == per_doc


def test_resolve_out_dir_ignores_another_documents_legacy_folder(
    tmp_path, write_lock, make_chunks
):
    pdf = tmp_path / "book.pdf"
    other = tmp_path / "other.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    other.write_bytes(b"%PDF-1.4\n")

    make_chunks(bt_queue.OUTPUT_ROOT, [(0, 9)])
    write_lock(bt_queue.OUTPUT_ROOT, other)

    assert bt_queue.resolve_out_dir(pdf) == bt_queue.OUTPUT_ROOT / "book"


# --------------------------------------------------------------------------
# attempts
# --------------------------------------------------------------------------
def test_progress_resets_the_attempt_counter():
    """A run stopped early by the disk guard still did useful work.

    Attempts count runs that advanced nothing, not runs that ended early.
    """
    bt_queue.record_attempt("a.pdf", progressed=False, error="nothing")
    bt_queue.record_attempt("a.pdf", progressed=False, error="nothing")
    assert bt_queue.load_overlay()["a.pdf"]["attempts"] == 2

    bt_queue.record_attempt("a.pdf", progressed=True)
    assert bt_queue.load_overlay()["a.pdf"]["attempts"] == 0
    assert bt_queue.load_overlay()["a.pdf"]["last_error"] == ""


def test_skip_is_persisted():
    bt_queue.set_skip("a.pdf", True)
    assert bt_queue.load_overlay()["a.pdf"]["skip"] is True


def test_overlay_survives_a_corrupt_file():
    bt_queue.QUEUE_FILE.write_text("{ not json", encoding="utf-8")
    assert bt_queue.load_overlay() == {}


# --------------------------------------------------------------------------
# discovery and ordering
# --------------------------------------------------------------------------
def test_discover_finds_root_and_uploads(tmp_path, make_pdf):
    make_pdf("root.pdf", pages=1)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "dropped.pdf").write_bytes((tmp_path / "root.pdf").read_bytes())

    names = {p.name for p in bt_queue.discover()}
    assert names == {"root.pdf", "dropped.pdf"}


def test_next_pending_prefers_a_started_document(tmp_path, make_pdf, make_chunks):
    """Finishing a book releases its disk and yields a usable result sooner
    than spreading effort across every book at once."""
    make_pdf("aaa_untouched.pdf", pages=25)
    make_pdf("zzz_started.pdf", pages=25)
    # 25 pages is three chunks, so one finished chunk leaves it in progress
    # rather than complete -- a document that is already done is not "next".
    make_chunks(bt_queue.OUTPUT_ROOT / "zzz_started", [(0, 9)])

    started = [s for s in bt_queue.survey() if s.pdf.name == "zzz_started.pdf"][0]
    assert started.status == "in progress"

    nxt = bt_queue.next_pending()
    assert nxt is not None
    assert nxt.pdf.name == "zzz_started.pdf", "alphabetically last, but already underway"


def test_next_pending_skips_skipped_and_stalled(tmp_path, make_pdf):
    make_pdf("only.pdf", pages=2)
    bt_queue.set_skip("only.pdf", True)
    assert bt_queue.next_pending() is None


def test_survey_reports_every_document(tmp_path, make_pdf):
    make_pdf("one.pdf", pages=2)
    make_pdf("two.pdf", pages=2)
    states = bt_queue.survey()
    assert {s.pdf.name for s in states} == {"one.pdf", "two.pdf"}
    assert all(s.chunks_total > 0 for s in states), "page counts should be known"
