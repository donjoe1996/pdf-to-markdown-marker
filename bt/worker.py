"""Keep transcribing, unattended, until everything is done.

    uv run python -m bt.worker

Without this the machine idles between books: a run that ends at midnight
wastes the hours until someone notices. The worker picks the next document as
soon as the previous one stops, so overnight time turns into progress.

It runs as its own process rather than inside the Streamlit app because
Streamlit only executes its script while a browser session is connected -- with
the tab closed nothing would tick.

The loop is built around a fact about this machine rather than an ideal one:
runs here **stop early on purpose**. ``llama-server`` drives multi-GB swap onto
the boot disk, ``transcribe.py`` halts below ``MIN_FREE_GB``, and the space
comes back a few minutes after the process exits. So finishing a book is
normally many short cycles, not one long run, and the worker is written to make
those cycles cheap: wait for the disk to recover, resume the same book, and
judge success by whether chunks advanced rather than by exit status.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
import time
from datetime import datetime

from bt import jobs, queue as bt_queue
from bt.transcribe import MIN_FREE_GB

# Another pipeline is running (possibly started by hand) -- check back soon.
POLL_BUSY = 30.0
# Nothing to do. Cheap, but often enough to notice a PDF dropped in uploads/.
POLL_IDLE = 120.0
# After a run exits, before starting another. Swap is released on exit but the
# disk does not recover instantly; starting immediately would trip the guard
# again and burn an attempt for nothing.
COOLDOWN = 180.0
# Disk is below the floor: wait longer, there is nothing useful to do.
BACKOFF_DISK = 300.0

# Start a run only with room to do real work. The run itself stops at
# MIN_FREE_GB, so beginning one at 2.1 GB would manage a chunk or two at best.
START_FREE_GB = MIN_FREE_GB + 1.5

_stop = False


def _handle_signal(signum, frame) -> None:  # pragma: no cover - signal path
    del frame
    global _stop
    _stop = True
    log(f"signal {signum} received; finishing the current wait and exiting")


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def _sleep(seconds: float) -> None:
    """Sleep in slices so a signal is noticed promptly."""
    deadline = time.time() + seconds
    while not _stop and time.time() < deadline:
        time.sleep(min(2.0, max(0.0, deadline - time.time())))


def run_one(state: bt_queue.DocState, chunk_size: int, dry_run: bool = False) -> bool:
    """Run one cycle on one document. Returns whether any chunk was added."""
    before = bt_queue.chunks_in(state.out_dir)
    log(
        f"starting {state.key} ({before}/{state.chunks_total} chunks, "
        f"{'split' if state.split else 'no split'}, "
        f"{'OCR' if state.ocr else 'text layer'})"
    )
    if dry_run:
        log("  dry run -- not launching")
        return False

    spec = bt_queue.spec_for(state, chunk_size)
    try:
        started = jobs.start(spec)
    except RuntimeError as exc:  # another run appeared between check and start
        log(f"  could not start: {exc}")
        return False

    if started.pid:
        jobs.wait_for_pid(started.pid)

    after = bt_queue.chunks_in(state.out_dir)
    progressed = after > before
    bt_queue.record_attempt(
        state.key, progressed, error="" if progressed else "no chunks advanced"
    )
    log(
        f"  finished {state.key}: {before} -> {after} chunks"
        + ("" if progressed else "  (no progress)")
    )
    return progressed


def loop(chunk_size: int, max_cycles: int | None = None, dry_run: bool = False) -> int:
    if not jobs.claim_worker_lock():
        log(f"a worker is already running (pid {jobs.worker_pid()}) -- exiting")
        return 1

    log(f"worker started; chunk size {chunk_size}, disk floor {START_FREE_GB:.1f} GB")
    cycles = 0
    try:
        while not _stop:
            if max_cycles is not None and cycles >= max_cycles:
                log(f"reached max cycles ({max_cycles})")
                break

            running = jobs.active_runs()
            if running:
                log(f"another job is running ({running[0].name}); waiting")
                _sleep(POLL_BUSY)
                continue

            free = free_gb()
            if free < START_FREE_GB:
                log(f"only {free:.1f} GB free (need {START_FREE_GB:.1f}); waiting")
                _sleep(BACKOFF_DISK)
                continue

            state = bt_queue.next_pending(chunk_size)
            if state is None:
                _sleep(POLL_IDLE)  # watch for new PDFs
                continue

            run_one(state, chunk_size, dry_run=dry_run)
            cycles += 1
            if dry_run:
                break
            _sleep(COOLDOWN)
    finally:
        jobs.release_worker_lock()
        log("worker stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Process the transcription queue.")
    ap.add_argument("--chunk-size", type=int, default=bt_queue.DEFAULT_CHUNK_SIZE)
    ap.add_argument(
        "--once", action="store_true", help="run a single cycle, then exit"
    )
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would run without launching anything",
    )
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    return loop(
        args.chunk_size,
        max_cycles=1 if args.once else args.max_cycles,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
