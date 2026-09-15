"""Process handling, locks and sentinels.

These tests spawn their own short-lived children and only ever signal those.
Nothing here touches a real pid, the real sentinel directory or the real worker
lock -- the autouse ``isolate`` fixture redirects the last two, and
``stop_pids`` is always called with ``kill_inference=False`` so its global
``pkill -f llama-server`` can never reach a live OCR server.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

from bt import jobs
from bt.jobs import PIPELINE_CMD, JobSpec, parse_command  # noqa: F401


# --------------------------------------------------------------------------
# which processes count as a pipeline run
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cmd", "matches"),
    [
        ("/path/.venv/bin/python3 -u -m bt.run --pdf x.pdf", True),
        ("uv run --quiet python -u -m bt.run --chunk-size 10", True),
        ("python -m bt.transcribe out/pages.pdf out/raw.md", True),
        # REGRESSION: a loose substring test matched a transient process
        # belonging to its own inspection command. A false positive here leaves
        # the worker waiting forever for a job that does not exist.
        ("python -c from bt import jobs; jobs.active_runs()", False),
        # The worker must never match itself, or it would wait on itself.
        ("python -m bt.worker --chunk-size 10", False),
        ("python -m bt.queue", False),
        ("streamlit run app.py", False),
    ],
)
def test_pipeline_command_matching(cmd, matches):
    assert bool(PIPELINE_CMD.search(cmd)) is matches


def test_parse_command_recovers_the_document():
    info = parse_command("python -m bt.run --pdf /books/a.pdf --out-dir /out/a --chunk-size 10")
    assert info == {"pdf": "/books/a.pdf", "out_dir": "/out/a"}


def test_parse_command_tolerates_missing_flags():
    assert parse_command("python -m bt.run") == {}


# --------------------------------------------------------------------------
# the command a job is launched with
# --------------------------------------------------------------------------
def test_jobspec_command_carries_the_settings():
    cmd = " ".join(JobSpec(pdf="a.pdf", out_dir="out", chunk_size=5, dpi=192).command())
    assert "-m bt.run" in cmd
    assert "--pdf a.pdf" in cmd
    assert "--chunk-size 5" in cmd
    assert "--dpi 192" in cmd


def test_jobspec_flags_are_opt_out():
    """split/ocr are on by default; the flags only appear when disabled."""
    on = " ".join(JobSpec(pdf="a.pdf", out_dir="o").command())
    assert "--no-split" not in on and "--no-ocr" not in on

    off = " ".join(JobSpec(pdf="a.pdf", out_dir="o", split=False, ocr=False).command())
    assert "--no-split" in off and "--no-ocr" in off


# --------------------------------------------------------------------------
# liveness
# --------------------------------------------------------------------------
def test_exited_child_is_not_alive():
    """REGRESSION: a reaped-less child stays in the table as a zombie.

    os.kill(pid, 0) still succeeds for it, so _alive() reported finished jobs as
    running. The worker launches jobs as its own children, so every completed
    job would have hung it until the stall timeout fired.
    """
    proc = subprocess.Popen(["true"])
    for _ in range(50):
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    assert jobs.is_alive(proc.pid) is False


def test_running_child_is_alive():
    proc = subprocess.Popen(["sleep", "5"])
    try:
        assert jobs.is_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()


def test_stop_pids_terminates_our_own_child():
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        # kill_inference=False: never run the global pkill from a test.
        assert jobs.stop_pids([proc.pid], kill_inference=False) is True
        for _ in range(40):
            if not jobs.is_alive(proc.pid):
                break
            time.sleep(0.1)
        assert jobs.is_alive(proc.pid) is False
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# --------------------------------------------------------------------------
# sentinels
# --------------------------------------------------------------------------
def test_clear_stale_sentinels_removes_dead_keeps_live():
    """REGRESSION: a sentinel naming a dead server made the next run wait on a
    health check that could never pass -- for the full 1800s startup timeout."""
    import os

    d = jobs.SURYA_SENTINEL_DIR
    (d / "live_server.json").write_text(json.dumps({"pid": os.getpid()}))
    (d / "dead_server.json").write_text(json.dumps({"pid": 999_999}))
    (d / "broken_server.json").write_text("not json")

    removed = jobs.clear_stale_sentinels()

    assert set(removed) == {"dead_server.json", "broken_server.json"}
    assert (d / "live_server.json").exists(), "a live server must still be reusable"


def test_clear_stale_sentinels_handles_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "SURYA_SENTINEL_DIR", tmp_path / "absent")
    assert jobs.clear_stale_sentinels() == []


# --------------------------------------------------------------------------
# worker lock
# --------------------------------------------------------------------------
def test_worker_lock_is_exclusive():
    assert jobs.claim_worker_lock() is True
    assert jobs.worker_pid() is not None
    assert jobs.claim_worker_lock() is False, "a second worker must not start"
    jobs.release_worker_lock()
    assert jobs.worker_pid() is None


def test_worker_lock_from_a_dead_process_is_ignored():
    """A crashed worker leaves a lock behind; it must not block the next one."""
    jobs.worker_lock_path().write_text(json.dumps({"pid": 999_999, "started": 0}))
    assert jobs.worker_pid() is None
    assert jobs.claim_worker_lock() is True
    jobs.release_worker_lock()


# --------------------------------------------------------------------------
# starting and stopping the worker from the GUI
# --------------------------------------------------------------------------
def test_worker_command_is_not_mistaken_for_a_job():
    """A GUI-started worker must not match PIPELINE_CMD, or it would show up as
    a running job and the worker would wait on itself."""
    cmd = " ".join(jobs.worker_command())
    assert "-m bt.worker" in cmd
    assert not PIPELINE_CMD.search(cmd)


def test_start_worker_refuses_a_second_worker():
    """Two workers means two llama-servers -- the swap collapse in QnA.md Q2."""
    assert jobs.claim_worker_lock() is True  # this test process is "the worker"
    with pytest.raises(RuntimeError):
        jobs.start_worker(wait=0)
    jobs.release_worker_lock()


def test_start_worker_launches_detached_and_logs(monkeypatch):
    # Never launch the real bt.worker from a test: it would claim the repo's
    # lock and start transcribing. A sleep stands in for it.
    monkeypatch.setattr(jobs, "worker_command", lambda: ["sleep", "30"])
    cmd = jobs.start_worker(wait=0)
    pids = [p for p in _pids_of("sleep 30")]
    try:
        assert cmd[:2] == ["sleep", "30"]
        assert pids, "the worker process was not started"
        assert jobs.worker_log_path().exists()
    finally:
        for pid in pids:
            jobs.stop_pids([pid], kill_inference=False)


def test_stop_worker_terminates_only_the_worker():
    proc = subprocess.Popen(["sleep", "30"])
    jobs.worker_lock_path().write_text(json.dumps({"pid": proc.pid, "started": 0}))
    try:
        cmd = jobs.stop_worker(wait=5)
        assert cmd == ["kill", "-TERM", str(proc.pid)]
        assert jobs.is_alive(proc.pid) is False
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_stop_worker_without_a_worker_does_nothing():
    assert jobs.stop_worker(wait=0) is None


def _pids_of(pattern: str) -> list[int]:
    """Children of this test process matching a command line."""
    import os

    out = subprocess.run(
        ["pgrep", "-P", str(os.getpid()), "-f", pattern], capture_output=True, text=True
    )
    return [int(p) for p in out.stdout.split()]


# --------------------------------------------------------------------------
# status from disk
# --------------------------------------------------------------------------
def test_status_counts_chunks_and_reads_the_spec(out_dir, write_lock, make_chunks, tmp_path):
    write_lock(out_dir, tmp_path / "book.pdf", pid=999_999, total_pages=40)
    make_chunks(out_dir, [(0, 9), (10, 19)])

    st = jobs.status(out_dir)
    assert st.chunks_done == 2
    assert st.chunks_total == 4  # 40 pages / chunk_size 10
    assert st.running is False  # the pid is dead
    assert st.fraction == 0.5


# --------------------------------------------------------------------------
# the translation job
# --------------------------------------------------------------------------
def test_translation_is_not_mistaken_for_a_pipeline_process():
    """`bt.translate` shares a prefix with `bt.transcribe`; it must not match.

    PIPELINE_CMD gates whether an OCR run may start. If a translation counted
    as a pipeline process, the GUI and the worker would both refuse to start
    OCR while one was in flight -- and a translation runs for hours. The
    reverse matters too: translation is network-bound, so it does not need the
    machine's memory and has no business holding that lock.
    """
    assert not PIPELINE_CMD.search("python -u -m bt.translate output/book/book.md")
    assert PIPELINE_CMD.search("python -u -m bt.transcribe pages.pdf raw.md")


def test_translate_command_carries_the_settings(tmp_path):
    spec = jobs.TranslateSpec(
        src=str(tmp_path / "book.md"),
        out=str(tmp_path / "book.english.md"),
        target="Indonesian",
        provider="groq",
        model="llama-3.3-70b-versatile",
        pages_per_chunk=4,
    )
    cmd = " ".join(spec.command())
    assert "-m bt.translate" in cmd
    assert "--target Indonesian" in cmd
    assert "--provider groq" in cmd
    assert "--model llama-3.3-70b-versatile" in cmd
    assert "--pages-per-chunk 4" in cmd


def test_an_unset_model_leaves_the_provider_default_alone(tmp_path):
    """Free-tier model ids are retired often; the preset is the fallback.

    Passing an empty --model would override the provider's default with
    nothing, and the request would name a model that does not exist.
    """
    spec = jobs.TranslateSpec(src=str(tmp_path / "b.md"), out=str(tmp_path / "o.md"))
    assert "--model" not in spec.command()


def test_translate_status_counts_its_own_chunks_not_the_ocr_chunks(out_dir, make_chunks):
    """The two chunk directories must never be confused for one another.

    `queue.survey()` judges a book finished by counting `chunks/`. Translation
    chunks landing there would make a half-transcribed book look done, and the
    worker would move on and never come back to it.
    """
    make_chunks(out_dir, [(0, 9)])  # OCR chunks
    tchunks = out_dir / jobs.TRANSLATE_DIR / "chunks"
    tchunks.mkdir(parents=True)
    for name in ("0000-0009.md", "0010-0019.md"):
        (tchunks / name).write_text("translated", encoding="utf-8")

    assert jobs.status(out_dir).chunks_done == 1
    assert jobs.translate_status(out_dir).chunks_done == 2


def test_translate_status_is_not_running_without_a_lock(out_dir):
    st = jobs.translate_status(out_dir)
    assert st.running is False and st.chunks_done == 0


# --------------------------------------------------------------------------
# an API key pasted into the GUI
# --------------------------------------------------------------------------
@pytest.fixture
def fake_popen(monkeypatch):
    """Capture the Popen call instead of launching a translator."""
    calls = []

    class _Proc:
        pid = 424242

    def _popen(cmd, **kw):
        calls.append((cmd, kw))
        return _Proc()

    monkeypatch.setattr(jobs.subprocess, "Popen", _popen)
    return calls


def test_a_pasted_key_reaches_the_child_through_the_environment(out_dir, fake_popen):
    """The GUI must be able to supply the key the detached run reads.

    ``OpenAICompatTranslator`` reads ``os.environ[key_env]``, so handing the
    child an environment is the whole mechanism: pasting a key and exporting
    one become the same code path, and the backend needs no key argument at
    all. The env var name comes from the provider, not the caller, so the GUI
    and the CLI cannot disagree about where groq's key lives.
    """
    spec = jobs.TranslateSpec(
        src=str(out_dir / "book.md"), out=str(out_dir / "book.en.md"), provider="groq"
    )
    jobs.start_translate(spec, out_dir, api_key="gsk-secret")

    (_cmd, kw) = fake_popen[0]
    assert kw["env"]["GROQ_API_KEY"] == "gsk-secret"
    # Inherited, not replaced: PATH and the HF vars must survive.
    assert kw["env"]["PATH"] == os.environ["PATH"]


def test_a_pasted_key_never_reaches_the_command_line(out_dir, fake_popen):
    """`ps` is readable by every user on the machine -- and by this project.

    ``find_pipeline_processes()`` exists precisely because command lines are
    public. A ``--api-key`` flag would publish the secret to anyone running
    ``ps``, for the hours the run takes.
    """
    spec = jobs.TranslateSpec(
        src=str(out_dir / "book.md"), out=str(out_dir / "book.en.md"), provider="groq"
    )
    jobs.start_translate(spec, out_dir, api_key="gsk-secret")

    assert "gsk-secret" not in " ".join(fake_popen[0][0])


def test_a_pasted_key_never_reaches_the_lock_file(out_dir, fake_popen):
    """``start_translate`` writes ``asdict(spec)`` to disk, under ``output/``.

    So the key must not live on the spec: that file is written next to the
    book, survives the run, and gets copied around with the output folder.
    """
    spec = jobs.TranslateSpec(
        src=str(out_dir / "book.md"), out=str(out_dir / "book.en.md"), provider="groq"
    )
    jobs.start_translate(spec, out_dir, api_key="gsk-secret")

    assert "gsk-secret" not in jobs.translate_lock_path(out_dir).read_text()


def test_an_empty_key_leaves_an_exported_one_alone(out_dir, fake_popen, monkeypatch):
    """A blank field means "use what is already exported", not "unset it".

    Overriding with "" would make the child fail with a missing-key error on a
    machine where the key was exported before the app started.
    """
    monkeypatch.setenv("GROQ_API_KEY", "from-the-shell")
    spec = jobs.TranslateSpec(
        src=str(out_dir / "book.md"), out=str(out_dir / "book.en.md"), provider="groq"
    )
    jobs.start_translate(spec, out_dir, api_key="")

    assert fake_popen[0][1]["env"]["GROQ_API_KEY"] == "from-the-shell"


def test_a_key_pasted_for_a_keyless_provider_is_dropped(out_dir, fake_popen):
    """`local` and `ollama` have no key_env; there is nowhere to put it.

    Inventing a variable name would either do nothing or send a secret to a
    server that never asked for one.
    """
    spec = jobs.TranslateSpec(
        src=str(out_dir / "book.md"), out=str(out_dir / "book.en.md"), provider="local"
    )
    jobs.start_translate(spec, out_dir, api_key="gsk-secret")

    env = fake_popen[0][1]["env"]
    assert "gsk-secret" not in env.values()
