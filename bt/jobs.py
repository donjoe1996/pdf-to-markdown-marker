"""Run the pipeline as a background subprocess, and report progress from disk.

Streamlit re-executes its script on every interaction, so a long job cannot
live inside the app: a click, a refresh, or a reconnect would lose it. Instead
the GUI launches the existing CLI as a detached subprocess and reads progress
back off the filesystem.

That works because progress is *already* durable. ``bt.transcribe`` writes each
completed chunk atomically to ``chunks/NNNN-NNNN.md`` and skips finished chunks
on restart, so the chunk directory is an accurate record of what is done. The
GUI holds no state at all -- close the browser, restart the app, and the same
run is still there.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOCK_NAME = ".run.lock"
LOG_NAME = "run.log"

# Translation is a separate job with its own lock, log and chunk directory. It
# is network-bound rather than memory-bound, so it deliberately does NOT count
# as a pipeline process: blocking an OCR run because a translation is in flight
# would be wrong, and the two can share the machine.
TRANSLATE_LOCK = ".translate.lock"
TRANSLATE_LOG = "translate.log"
TRANSLATE_DIR = "translation"

# `python -m bt.run …` / `python -m bt.transcribe …`, however it was launched.
PIPELINE_CMD = re.compile(r"-m\s+bt\.(run|transcribe)\b")


@dataclass
class JobSpec:
    pdf: str
    out_dir: str
    split: bool = True
    ocr: bool = True
    dpi: int = 300
    chunk_size: int = 10
    mode: str = "fast"
    pages: str | None = None
    total_pages: int = 0  # pages after splitting; drives the progress bar
    images: bool = False

    def command(self) -> list[str]:
        cmd = [
            sys.executable, "-u", "-m", "bt.run",
            "--pdf", self.pdf,
            "--out-dir", self.out_dir,
            "--chunk-size", str(self.chunk_size),
            "--dpi", str(self.dpi),
            "--mode", self.mode,
            "--skip-preflight",
        ]
        if not self.split:
            cmd.append("--no-split")
        if not self.ocr:
            cmd.append("--no-ocr")
        if self.pages:
            cmd += ["--pages", self.pages]
        if self.images:
            cmd.append("--images")
        return cmd


@dataclass
class JobStatus:
    running: bool = False
    pid: int | None = None
    spec: JobSpec | None = None
    chunks_done: int = 0
    chunks_total: int = 0
    started_at: float = 0.0
    finished: bool = False
    stopped_early: bool = False
    last_lines: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        if self.chunks_total <= 0:
            return 0.0
        return min(1.0, self.chunks_done / self.chunks_total)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0


def lock_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / LOCK_NAME


def log_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / LOG_NAME


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 tests existence without touching the process
    except OSError:
        return False

    # A finished child that nobody has reaped stays in the process table as a
    # zombie, and signal 0 still succeeds for it. Since the worker launches jobs
    # as its own children, treating a zombie as running would leave it waiting
    # on a job that already finished -- and the watchdog would eventually
    # "stop" a process that exited long ago.
    try:
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.stdout.strip().startswith("Z"):
            try:  # best effort: reap it if it is ours, so it stops piling up
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass
            return False
    except (OSError, subprocess.SubprocessError):
        pass  # cannot tell; assume alive rather than kill a live job
    return True


def is_alive(pid: int) -> bool:
    """Public alias -- other modules need this without reaching for a private."""
    return _alive(pid)


# surya records each model server it spawns here, so a later run can attach to
# one that is already up instead of paying the start-up cost again.
SURYA_SENTINEL_DIR = Path("~/.cache/datalab/surya").expanduser()


def clear_stale_sentinels() -> list[str]:
    """Delete sentinels whose server process is gone.

    When a run ends, ``shutdown_models()`` stops the servers but the sentinel
    files can outlive them. The next run then reads a sentinel, tries to attach
    to a dead port, and waits on a health check that can never succeed -- for
    the full ``*_SERVER_STARTUP_TIMEOUT``, which this project raises to 1800s to
    survive slow first-run downloads. The symptom is a job that looks perfectly
    healthy (process alive, no error) while doing nothing for half an hour.

    Observed exactly that: a sentinel naming a dead pid, nothing listening on
    its port, and the job burning 1m23s of CPU across 14 minutes.
    """
    removed: list[str] = []
    try:
        files = sorted(SURYA_SENTINEL_DIR.glob("*_server.json"))
    except OSError:
        return removed

    for path in files:
        try:
            pid = int(json.loads(path.read_text(encoding="utf-8")).get("pid", 0))
        except (ValueError, OSError, TypeError, KeyError):
            pid = 0  # unreadable sentinel is no use to anyone either
        if pid and _alive(pid):
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    return removed


def find_pipeline_processes(exclude_pid: int | None = None) -> list[tuple[int, str]]:
    """Any pipeline process on this machine, however it was started.

    The lock file only knows about jobs the GUI launched. A run started from a
    terminal has no lock, and starting a second one alongside it is the worst
    thing you can do on this hardware: two llama-servers push the machine into
    swap, where throughput collapses (see QnA.md Q2). So look for the real
    processes rather than trusting the lock alone.
    """
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return []

    hits: list[tuple[int, str]] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        # Match an actual module invocation, not any command line that happens
        # to contain the text. A loose substring test matched a transient
        # process of its own inspection command; a false positive here makes the
        # worker sit and wait for a job that does not exist.
        if not PIPELINE_CMD.search(cmd):
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in (os.getpid(), exclude_pid):
            continue
        hits.append((pid, cmd.strip()))
    return hits


@dataclass
class ActiveRun:
    """A pipeline running anywhere on this machine, identified from its argv."""

    pids: list[int]
    pdf: str = ""
    out_dir: str = ""

    @property
    def name(self) -> str:
        return Path(self.pdf).name if self.pdf else "unknown document"

    @property
    def chunks_done(self) -> int:
        if not self.out_dir:
            return 0
        return len(list((Path(self.out_dir) / "chunks").glob("[0-9]*-[0-9]*.md")))


def parse_command(cmd: str) -> dict:
    """Pull --pdf and --out-dir back out of a running process's command line."""
    parts = cmd.split()
    found: dict = {}
    for flag, key in (("--pdf", "pdf"), ("--out-dir", "out_dir")):
        if flag in parts:
            i = parts.index(flag)
            if i + 1 < len(parts):
                found[key] = parts[i + 1]
    return found


def active_runs() -> list[ActiveRun]:
    """Distinct pipeline jobs currently running, however they were started.

    One job shows up as several processes (the ``uv run`` wrapper plus the
    python child), so they are grouped by the document they are working on --
    stopping a job has to signal all of them, not just the one you happened to
    find first.
    """
    grouped: dict[tuple[str, str], ActiveRun] = {}
    for pid, cmd in find_pipeline_processes():
        info = parse_command(cmd)
        key = (info.get("pdf", ""), info.get("out_dir", ""))
        run = grouped.setdefault(
            key, ActiveRun(pids=[], pdf=key[0], out_dir=key[1])
        )
        run.pids.append(pid)
    return list(grouped.values())


def _kill_group(pid: int, sig: int) -> bool:
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except OSError:
        try:
            os.kill(pid, sig)
            return True
        except OSError:
            return False


def stop_pids(pids: list[int], kill_inference: bool = True) -> bool:
    """Stop a job by pid, then clean up the inference server it left behind.

    marker spawns ``llama-server`` in its own session, so it does not die with
    the python process -- observed surviving a parent kill and holding ~2.5 GB
    indefinitely. SIGTERM first so the pipeline can shut down cleanly and finish
    writing the chunk in flight; only then force the leftovers.
    """
    stopped = any(_kill_group(pid, signal.SIGTERM) for pid in pids)

    for _ in range(20):  # up to ~5s for a clean exit
        if not any(_alive(p) for p in pids):
            break
        time.sleep(0.25)
    for pid in pids:
        if _alive(pid):
            _kill_group(pid, signal.SIGKILL)

    if kill_inference:
        try:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    return stopped


def start(spec: JobSpec) -> JobStatus:
    """Launch the pipeline. Refuses if a job is already running here."""
    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    current = status(spec.out_dir)
    if current.running:
        raise RuntimeError(f"a job is already running (pid {current.pid})")
    foreign = find_pipeline_processes()
    if foreign:
        pids = ", ".join(str(p) for p, _ in foreign)
        raise RuntimeError(
            f"another pipeline process is already running (pid {pids}). "
            "Running two at once pushes this machine into swap and slows both. "
            "Wait for it, or stop it first."
        )

    log = log_path(out).open("ab")
    proc = subprocess.Popen(
        spec.command(),
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).resolve().parent.parent,
        # Detach into its own process group so it is not killed when the GUI
        # (or the terminal that started the GUI) goes away.
        start_new_session=True,
    )
    lock_path(out).write_text(
        json.dumps({"pid": proc.pid, "started": time.time(), "spec": asdict(spec)}),
        encoding="utf-8",
    )
    return status(spec.out_dir)


def _read_status(lock: Path, log: Path, chunk_dir: Path, spec_type) -> JobStatus:
    """Reconstruct a job's state from its lock, log and chunk directory.

    Shared by the OCR pipeline and the translator because they record progress
    the same way -- atomically written chunk files -- and differ only in where
    those files live.
    """
    st = JobStatus()

    if lock.exists():
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
            st.pid = data.get("pid")
            st.started_at = data.get("started", 0.0)
            st.spec = spec_type(**data["spec"]) if data.get("spec") else None
        except (ValueError, KeyError, TypeError):
            st.spec = None
    if st.pid:
        st.running = _alive(st.pid)

    if chunk_dir.is_dir():
        st.chunks_done = len(list(chunk_dir.glob("[0-9]*-[0-9]*.md")))

    if log.exists():
        tail = log.read_bytes()[-8000:].decode("utf-8", "replace")
        lines = [ln for ln in tail.replace("\r", "\n").split("\n") if ln.strip()]
        st.last_lines = lines[-12:]
        st.stopped_early = any("STOPPING" in ln for ln in lines)
        st.finished = (not st.running) and any("Concatenated" in ln for ln in lines)
    return st


def status(out_dir: str | Path) -> JobStatus:
    """Current state of the OCR pipeline, reconstructed entirely from disk."""
    out = Path(out_dir)
    st = _read_status(lock_path(out), log_path(out), out / "chunks", JobSpec)
    if st.spec and st.spec.total_pages and st.spec.chunk_size:
        st.chunks_total = -(-st.spec.total_pages // st.spec.chunk_size)  # ceil
    return st


# --------------------------------------------------------------------------
# translation: the same launch-and-watch shape, a separate job
# --------------------------------------------------------------------------
@dataclass
class TranslateSpec:
    src: str
    out: str
    target: str = "English"
    provider: str = "openrouter"
    model: str = ""  # empty means the provider's own default
    pages_per_chunk: int = 10
    total_pages: int = 0  # pages in the source markdown; drives the progress bar

    @property
    def chunk_dir(self) -> Path:
        return Path(self.src).parent / TRANSLATE_DIR / "chunks"

    def command(self) -> list[str]:
        cmd = [
            sys.executable, "-u", "-m", "bt.translate",
            self.src, self.out,
            "--target", self.target,
            "--provider", self.provider,
            "--pages-per-chunk", str(self.pages_per_chunk),
            "--chunk-dir", str(self.chunk_dir),
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd


def translate_lock_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / TRANSLATE_LOCK


def translate_log_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / TRANSLATE_LOG


def translate_status(out_dir: str | Path) -> JobStatus:
    """Translation progress, read from its own chunk directory.

    Deliberately not ``out_dir/chunks``: ``queue.survey()`` judges a book
    finished by counting files there, so a translation chunk landing in it
    would make a half-transcribed book look complete and the worker would move
    on for good.
    """
    out = Path(out_dir)
    st = _read_status(
        translate_lock_path(out),
        translate_log_path(out),
        out / TRANSLATE_DIR / "chunks",
        TranslateSpec,
    )
    if st.spec and st.spec.total_pages and st.spec.pages_per_chunk:
        st.chunks_total = -(-st.spec.total_pages // st.spec.pages_per_chunk)
    return st


def start_translate(spec: TranslateSpec, out_dir: str | Path) -> JobStatus:
    """Launch the translator detached, as ``start()`` does for the pipeline.

    Same reasoning: Streamlit re-runs its script on every interaction, and a
    book takes hours. Refuses only a second translation of the *same* document
    -- an OCR run elsewhere on the machine is no obstacle, since this job is
    waiting on the network rather than holding the memory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    current = translate_status(out)
    if current.running:
        raise RuntimeError(f"a translation is already running (pid {current.pid})")

    log = translate_log_path(out).open("ab")
    proc = subprocess.Popen(
        spec.command(),
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).resolve().parent.parent,
        start_new_session=True,
    )
    translate_lock_path(out).write_text(
        json.dumps({"pid": proc.pid, "started": time.time(), "spec": asdict(spec)}),
        encoding="utf-8",
    )
    return translate_status(out)


def stop(out_dir: str | Path) -> bool:
    """Stop a running job. Chunks already written are kept."""
    st = status(out_dir)
    if not (st.running and st.pid):
        return False
    return stop_pids([st.pid])


def clear_lock(out_dir: str | Path) -> None:
    """Remove a stale lock left by a crashed run."""
    lock_path(out_dir).unlink(missing_ok=True)


WORKER_LOCK = "output/.worker.lock"


def worker_lock_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parent.parent
    return base / WORKER_LOCK


def worker_pid() -> int | None:
    """The queue worker's pid, if one is alive."""
    lock = worker_lock_path()
    if not lock.exists():
        return None
    try:
        pid = int(json.loads(lock.read_text(encoding="utf-8"))["pid"])
    except (ValueError, KeyError, OSError, TypeError):
        return None
    return pid if _alive(pid) else None


def claim_worker_lock() -> bool:
    """Register this process as the worker. False if one is already running."""
    if worker_pid() is not None:
        return False
    lock = worker_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8"
    )
    return True


def release_worker_lock() -> None:
    worker_lock_path().unlink(missing_ok=True)


def worker_command() -> list[str]:
    return [sys.executable, "-u", "-m", "bt.worker"]


def worker_log_path() -> Path:
    return worker_lock_path().parent / "worker.log"


def start_worker(wait: float = 5.0) -> list[str]:
    """Launch the queue worker detached, as the GUI's start button does.

    Detached for the same reason as a job: the worker must outlive the browser
    tab and the Streamlit process. Returns the command it ran so the GUI can
    show exactly what happened. Waits briefly for the worker to claim its lock,
    so the page that reruns next already shows it as running.
    """
    running = worker_pid()
    if running is not None:
        raise RuntimeError(f"a worker is already running (pid {running})")

    cmd = worker_command()
    log_file = worker_log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parent.parent,
            start_new_session=True,
        )

    deadline = time.time() + wait
    while time.time() < deadline and worker_pid() is None and proc.poll() is None:
        time.sleep(0.2)
    return cmd


def stop_worker(wait: float = 5.0) -> list[str] | None:
    """Ask the worker to exit. Returns the equivalent shell command, or None.

    SIGTERM only, and only to the worker's own pid: it finishes its current
    wait, releases the lock and exits within a couple of seconds. A job it
    already launched lives in its own session and keeps running -- stopping
    that is a separate, deliberate step (``stop_pids``).
    """
    pid = worker_pid()
    if pid is None:
        return None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return None

    deadline = time.time() + wait
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    return ["kill", "-TERM", str(pid)]


def wait_for_pid(pid: int, poll: float = 5.0) -> None:
    """Block until a process exits."""
    while _alive(pid):
        time.sleep(poll)


def per_page_seconds(st: JobStatus) -> float | None:
    """Most recent seconds-per-page, parsed from the log."""
    import re

    for line in reversed(st.last_lines):
        m = re.search(r"([\d.]+)s/page", line)
        if m:
            return float(m.group(1))
    return None
