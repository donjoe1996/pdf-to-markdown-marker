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
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOCK_NAME = ".run.lock"
LOG_NAME = "run.log"


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
    return True


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
        if not (" bt.run" in f" {cmd}" or "bt.transcribe" in cmd):
            continue
        if "ps -axo" in cmd:
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


def status(out_dir: str | Path) -> JobStatus:
    """Current state, reconstructed entirely from disk."""
    out = Path(out_dir)
    st = JobStatus()

    lock = lock_path(out)
    if lock.exists():
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
            st.pid = data.get("pid")
            st.started_at = data.get("started", 0.0)
            st.spec = JobSpec(**data["spec"]) if data.get("spec") else None
        except (ValueError, KeyError, TypeError):
            st.spec = None
    if st.pid:
        st.running = _alive(st.pid)

    chunk_dir = out / "chunks"
    if chunk_dir.is_dir():
        st.chunks_done = len(list(chunk_dir.glob("[0-9]*-[0-9]*.md")))
    if st.spec and st.spec.total_pages and st.spec.chunk_size:
        st.chunks_total = -(-st.spec.total_pages // st.spec.chunk_size)  # ceil

    log = log_path(out)
    if log.exists():
        tail = log.read_bytes()[-8000:].decode("utf-8", "replace")
        lines = [ln for ln in tail.replace("\r", "\n").split("\n") if ln.strip()]
        st.last_lines = lines[-12:]
        st.stopped_early = any("STOPPING" in ln for ln in lines)
        st.finished = (not st.running) and any("Concatenated" in ln for ln in lines)
    return st


def stop(out_dir: str | Path) -> bool:
    """Stop a running job. Chunks already written are kept."""
    st = status(out_dir)
    if not (st.running and st.pid):
        return False
    return stop_pids([st.pid])


def clear_lock(out_dir: str | Path) -> None:
    """Remove a stale lock left by a crashed run."""
    lock_path(out_dir).unlink(missing_ok=True)


def per_page_seconds(st: JobStatus) -> float | None:
    """Most recent seconds-per-page, parsed from the log."""
    import re

    for line in reversed(st.last_lines):
        m = re.search(r"([\d.]+)s/page", line)
        if m:
            return float(m.group(1))
    return None
