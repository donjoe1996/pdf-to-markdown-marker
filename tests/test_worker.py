"""The worker's judgement: did that run accomplish anything?

This is the one decision the worker makes unattended, and both ways of getting
it wrong are silent. If a normal early stop were counted as failure, every book
would be abandoned after three cycles while transcribing perfectly well. If a
genuinely stuck book never counted as failure, it would be retried forever and
block every other book. Neither raises, neither logs a crash.

The expensive part is faked out. ``jobs.start()`` loads a model and runs OCR for
hours; the stand-ins here just write chunk files, so what gets tested is the
counting and the decision rather than marker's OCR.

**Safety note.** ``supervise()`` calls ``jobs.stop_pids()``, whose default runs a
global ``pkill -f llama-server`` — which would kill a live transcription on this
machine. Every test that reaches that path replaces ``stop_pids`` with a stub.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from bt import jobs, worker
from bt import queue as bt_queue
from bt.queue import MAX_ATTEMPTS, DocState


@pytest.fixture
def doc():
    """A document with five chunks of work outstanding."""
    out = bt_queue.OUTPUT_ROOT / "book"
    (out / "chunks").mkdir(parents=True)
    return DocState(
        pdf=bt_queue.ROOT / "book.pdf",
        out_dir=out,
        chunks_total=5,
        pages=50,
    )


def start_that_writes(*chunks: tuple[int, int]):
    """Stand in for jobs.start(): write chunk files instead of running OCR.

    Returns a status with ``pid=None`` so ``run_one`` skips supervision — the
    point here is the judgement afterwards, not the waiting.
    """

    def _start(spec):
        folder = Path(spec.out_dir) / "chunks"
        folder.mkdir(parents=True, exist_ok=True)
        for first, last in chunks:
            (folder / f"{first:04d}-{last:04d}.md").write_text("x", encoding="utf-8")
        return jobs.JobStatus(pid=None)

    return _start


# --------------------------------------------------------------------------
# the judgement
# --------------------------------------------------------------------------
def test_a_run_that_wrote_chunks_counts_as_progress(doc, monkeypatch):
    monkeypatch.setattr(jobs, "start", start_that_writes((0, 9), (10, 19)))

    assert worker.run_one(doc, chunk_size=10) is True
    assert bt_queue.load_overlay()[doc.key]["attempts"] == 0


def test_an_early_stop_that_still_did_work_is_not_a_failure(doc, monkeypatch):
    """The case that matters most on this machine.

    Runs stop early here constantly — the disk guard halts them once space runs
    low. That is normal, not exceptional. If it were counted as failure, every
    book would hit the attempt limit and be abandoned while transcribing fine.
    """
    # one chunk out of five, then the run ends: exactly what a disk-guard stop
    # looks like from the outside
    monkeypatch.setattr(jobs, "start", start_that_writes((0, 9)))

    assert worker.run_one(doc, chunk_size=10) is True
    assert bt_queue.load_overlay()[doc.key]["attempts"] == 0, "not a failed attempt"


def test_a_run_that_achieved_nothing_counts_an_attempt(doc, monkeypatch):
    monkeypatch.setattr(jobs, "start", start_that_writes())  # writes nothing

    assert worker.run_one(doc, chunk_size=10) is False
    overlay = bt_queue.load_overlay()[doc.key]
    assert overlay["attempts"] == 1
    assert "no chunks advanced" in overlay["last_error"]


def test_progress_resets_earlier_failures(doc, monkeypatch):
    monkeypatch.setattr(jobs, "start", start_that_writes())
    worker.run_one(doc, chunk_size=10)
    worker.run_one(doc, chunk_size=10)
    assert bt_queue.load_overlay()[doc.key]["attempts"] == 2

    monkeypatch.setattr(jobs, "start", start_that_writes((0, 9)))
    worker.run_one(doc, chunk_size=10)
    assert bt_queue.load_overlay()[doc.key]["attempts"] == 0


def test_a_book_that_never_progresses_is_eventually_stalled(doc, monkeypatch):
    """So one bad document cannot block the queue forever."""
    monkeypatch.setattr(jobs, "start", start_that_writes())
    for _ in range(MAX_ATTEMPTS):
        worker.run_one(doc, chunk_size=10)

    attempts = bt_queue.load_overlay()[doc.key]["attempts"]
    assert attempts == MAX_ATTEMPTS

    stalled = DocState(pdf=doc.pdf, out_dir=doc.out_dir, chunks_total=5, attempts=attempts)
    assert stalled.status == "stalled"


def test_refusing_to_start_is_not_counted_against_the_book(doc, monkeypatch):
    """jobs.start() raises when another run appears between check and launch.

    That says nothing about this document, so it must not burn an attempt.
    """
    def refuse(spec):
        raise RuntimeError("a job is already running")

    monkeypatch.setattr(jobs, "start", refuse)

    assert worker.run_one(doc, chunk_size=10) is False
    assert doc.key not in bt_queue.load_overlay()


def test_dry_run_launches_nothing(doc, monkeypatch):
    def explode(spec):
        raise AssertionError("dry run must not start a job")

    monkeypatch.setattr(jobs, "start", explode)
    assert worker.run_one(doc, chunk_size=10, dry_run=True) is False


# --------------------------------------------------------------------------
# supervision: is the job actually working?
#
# These wait on real child processes, so they cost seconds rather than
# milliseconds and are marked slow. The judgement tests above -- the part that
# decides whether a night's work counted -- stay in the fast suite.
# --------------------------------------------------------------------------
@pytest.fixture
def fast_watchdog(monkeypatch):
    """Shrink the timers, and neuter the global pkill in stop_pids.

    The logic does not care about absolute durations, only about whether output
    moved within the window, so the timers are cut to keep these quick. They
    still wait on real processes, which is why this section is marked slow.
    """
    monkeypatch.setattr(worker, "STALL_TIMEOUT", 1.5)
    monkeypatch.setattr(worker, "WATCH_POLL", 0.3)

    killed: list[list[int]] = []

    def safe_stop(pids, kill_inference=True):
        killed.append(list(pids))
        for pid in pids:  # only ever the pid this test spawned
            try:
                subprocess.run(["kill", "-9", str(pid)], capture_output=True)
            except OSError:
                pass
        return True

    monkeypatch.setattr(jobs, "stop_pids", safe_stop)
    return killed


@pytest.mark.slow
def test_a_job_that_finishes_is_reported_as_exited(tmp_path, fast_watchdog):
    log = tmp_path / "run.log"
    log.write_text("start")
    proc = subprocess.Popen(["sleep", "1"])
    try:
        assert worker.supervise(proc.pid, tmp_path, log) == "exited"
        assert fast_watchdog == [], "a finished job must not be killed"
    finally:
        proc.wait()


@pytest.mark.slow
def test_a_silent_job_is_stopped_as_stalled(tmp_path, fast_watchdog):
    """REGRESSION: a job waiting on a dead model server looks perfectly healthy.

    It sat for 14 minutes using 1m23s of CPU, producing nothing, and would have
    waited out the full 1800s startup timeout. "Process alive" is not evidence
    of work.
    """
    (tmp_path / "chunks").mkdir()
    log = tmp_path / "run.log"
    log.write_text("start")
    proc = subprocess.Popen(["sleep", "120"], start_new_session=True)
    try:
        assert worker.supervise(proc.pid, tmp_path, log) == "stalled"
        assert fast_watchdog == [[proc.pid]], "only the supervised pid"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


@pytest.mark.slow
def test_a_job_still_writing_output_is_left_alone(tmp_path, fast_watchdog):
    """Liveness is judged by output, so a slow-but-working job is not killed."""
    (tmp_path / "chunks").mkdir()
    log = tmp_path / "run.log"
    log.write_text("start")

    # runs comfortably past the stall timeout, but never goes quiet
    proc = subprocess.Popen(
        ["sh", "-c", f"for i in 1 2 3 4 5; do echo tick >> {log}; sleep 0.7; done"],
        start_new_session=True,
    )
    try:
        started = time.time()
        assert worker.supervise(proc.pid, tmp_path, log) == "exited"
        assert time.time() - started > worker.STALL_TIMEOUT, "ran past the timeout"
        assert fast_watchdog == [], "a working job must never be killed"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
