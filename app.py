"""Streamlit GUI for the PDF -> Markdown pipeline.

Run with:  uv run streamlit run app.py

The app is a launcher and monitor, not the pipeline itself. Long runs happen in
a detached subprocess (``bt.jobs``) and progress is read back off disk, so the
browser can be closed and reopened -- or the app restarted -- without losing a
run. See bt/jobs.py for why that matters.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pymupdf
import streamlit as st

from bt import auth, jobs
from bt import queue as bt_queue
from bt.analyze import analyze
from bt.split_spreads import _ink_profile, find_gutter

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"

# Set only in the Docker image this repo ships for Hugging Face Spaces (see
# Dockerfile / README "Deploy your own copy"). Local, personal use of this
# app -- the normal case documented in CLAUDE.md -- never sets it and gets
# none of the login/quota gate below: one trusted user, one machine, no need
# for accounts.
PUBLIC_MODE = os.environ.get("BT_PUBLIC_MODE") == "1"

st.set_page_config(page_title="PDF → Markdown", page_icon="📄", layout="wide")


def _require_login() -> str:
    """Render the login/signup/forgot-code screen; return the signed-in username.

    Only called in PUBLIC_MODE. Not real authentication -- see bt/auth.py's
    module docstring for exactly what this protects (per-visitor storage and
    a page cap on a shared free container) and what it does not (identity).
    """
    user = st.session_state.get("user")
    if user:
        return user

    st.title("📄 PDF → Markdown")
    st.caption(
        "Free demo on a shared CPU server. No real accounts -- this just "
        "keeps your files separate from everyone else's and caps each "
        f"document at {auth.MAX_PAGES_PER_DOCUMENT} pages."
    )
    tab_signup, tab_login, tab_forgot = st.tabs(["New here", "Log in", "Forgot your code"])

    with tab_signup:
        with st.form("signup"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email (not verified — only used to recover your code)")
            if st.form_submit_button("Create account", type="primary"):
                try:
                    code = auth.signup(new_username, new_email)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(
                        f"Your one-time login code is **{code}**. It is shown "
                        "only here, once — save it now. You'll need your "
                        "username and this code to come back.",
                        icon=":material/key:",
                    )

    with tab_login:
        with st.form("login"):
            username = st.text_input("Username", key="login_username")
            code = st.text_input("Login code", key="login_code")
            if st.form_submit_button("Log in", type="primary"):
                if auth.login(username, code):
                    st.session_state["user"] = auth.normalize_username(username)
                    st.rerun()
                else:
                    st.error("Wrong username or code.")

    with tab_forgot:
        st.caption(
            "No email is sent — if your username and email match, a new code "
            "is shown right here and your old code stops working."
        )
        with st.form("forgot"):
            forgot_username = st.text_input("Username", key="forgot_username")
            forgot_email = st.text_input("Email you signed up with", key="forgot_email")
            if st.form_submit_button("Get a new code"):
                new_code = auth.reset_code(forgot_username, forgot_email)
                if new_code:
                    st.success(f"Your new login code is **{new_code}**.", icon=":material/key:")
                else:
                    st.error("No account matches that username and email.")

    st.stop()


if PUBLIC_MODE:
    current_user = _require_login()
    with st.sidebar:
        who = st.container(horizontal=True, vertical_alignment="center")
        who.caption(f"Signed in as **{current_user}**")
        if who.button("Log out", icon=":material/logout:"):
            del st.session_state["user"]
            st.rerun()
else:
    current_user = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Inspecting PDF…")
def cached_analysis(path: str, mtime: float, size: int):
    """Analysis is deterministic per file; key the cache on identity, not name."""
    del mtime, size  # only present to invalidate the cache
    return analyze(path)


@st.cache_data(show_spinner=False)
def render_page(path: str, index: int, dpi: int, mtime: float) -> bytes:
    del mtime
    with pymupdf.open(path) as doc:
        index = max(0, min(index, doc.page_count - 1))
        return doc[index].get_pixmap(dpi=dpi).tobytes("png")


@st.cache_data(show_spinner=False)
def render_halves(path: str, index: int, dpi: int, mtime: float):
    """Render the two halves a spread would be split into, plus the cut point."""
    del mtime
    with pymupdf.open(path) as doc:
        page = doc[max(0, min(index, doc.page_count - 1))]
        rect = page.rect
        gutter, band = find_gutter(_ink_profile(page))
        cut = rect.x0 + gutter * rect.width
        left = page.get_pixmap(
            dpi=dpi, clip=pymupdf.Rect(rect.x0, rect.y0, cut, rect.y1)
        ).tobytes("png")
        right = page.get_pixmap(
            dpi=dpi, clip=pymupdf.Rect(cut, rect.y0, rect.x1, rect.y1)
        ).tobytes("png")
    return left, right, gutter, band


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def find_pdfs(dirs: list[Path] | None = None) -> list[Path]:
    """PDFs in the given directories (default: the project root and uploads/).

    Uploaded files are saved to disk so they survive a reload -- without
    listing that folder they would disappear from the picker as soon as the
    uploader widget cleared, stranding work already done on them. In
    PUBLIC_MODE the caller passes just the current user's own upload folder,
    so one visitor never sees another's filenames.
    """
    search = dirs if dirs is not None else [ROOT, ROOT / "uploads"]
    found = [p for d in search for p in d.glob("*.pdf")]
    return sorted(
        (p for p in found if not p.name.startswith(".")), key=lambda p: p.name.lower()
    )


# --------------------------------------------------------------------------
# sidebar: choose a document
# --------------------------------------------------------------------------
st.sidebar.title("📄 PDF → Markdown")

user_upload_dir = auth.account_dir(current_user) if PUBLIC_MODE else ROOT / "uploads"
pdfs = find_pdfs([user_upload_dir]) if PUBLIC_MODE else find_pdfs()
# Label by path relative to the project, not bare filename: uploads/ files live
# in a subfolder, and resolving a bare name against the root would miss them.
by_label = {str(p.relative_to(ROOT)): p for p in pdfs}
labels = list(by_label) + ["Upload a file…"]
choice = st.sidebar.selectbox("Document", labels, index=0 if pdfs else len(labels) - 1)

pdf_path: Path | None = None
if choice == "Upload a file…":
    up = st.sidebar.file_uploader("Choose a PDF", type="pdf")
    if up is not None:
        dest = user_upload_dir / up.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != up.size:
            dest.write_bytes(up.getbuffer())
        pdf_path = dest
else:
    pdf_path = by_label.get(choice)

if pdf_path is None or not pdf_path.exists():
    st.info("Choose a PDF in the sidebar to begin.")
    st.stop()

stat = pdf_path.stat()
st.sidebar.caption(f"{stat.st_size / 1e6:.1f} MB")


chunks_in = bt_queue.chunks_in

if PUBLIC_MODE:
    # Fixed and namespaced by account, not user-editable: a free-text output
    # folder would let one visitor point at (and read) another's in-progress
    # or finished work by guessing their folder name.
    out_dir = OUTPUT_ROOT / current_user / pdf_path.stem
    st.sidebar.caption(f"Output: `{out_dir.relative_to(ROOT)}`")
else:
    # Resolving this in bt.queue keeps the GUI and the worker agreeing on
    # where a document's chunks live. If they disagreed, one could resume
    # onto the other's output -- see queue.folder_belongs_to for why
    # ownership comes from the lock.
    default_out = bt_queue.resolve_out_dir(pdf_path)

    # Shown relative to the project when it lives there, which is the normal
    # case and much easier to read. An output folder elsewhere is shown in
    # full rather than raising -- relative_to() is strict, and an unhandled
    # ValueError here takes down the whole page.
    try:
        default_label = str(default_out.relative_to(ROOT))
    except ValueError:
        default_label = str(default_out)

    out_text = st.sidebar.text_input(
        "Output folder",
        value=default_label,
        help="Holds chunks/, raw.md and the final Markdown. Point it at an existing "
        "folder to resume that run.",
    )
    out_dir = Path(out_text) if Path(out_text).is_absolute() else ROOT / out_text

existing_chunks = chunks_in(out_dir)
if existing_chunks:
    st.sidebar.success(f"{existing_chunks} chunks already done here", icon=":material/history:")

# A pipeline running anywhere on this machine matters, not just one we started.
# Only one may run at a time, so this is effectively the session state: show it
# up front, and let it be cancelled from here rather than leaving the user stuck
# with a disabled button and no way out.
runs = jobs.active_runs()
job = jobs.status(out_dir)

if runs:
    with st.container(border=True):
        st.markdown("**Session in progress** — only one job runs at a time.")
        for run in runs:
            row = st.container(horizontal=True, vertical_alignment="center")
            same_doc = str(pdf_path) == run.pdf
            # In PUBLIC_MODE, a filename identifies a stranger's document --
            # show it only for the account it belongs to.
            run_is_others = PUBLIC_MODE and not same_doc and not str(
                Path(run.pdf)
            ).startswith(str(user_upload_dir))
            name = "another user's document" if run_is_others else run.name
            row.markdown(
                f":material/sync: **{name}**"
                f"{'  ·  _this document_' if same_doc else ''}  \n"
                f"{run.chunks_done} chunks done"
                + ("" if run_is_others else f" · `{Path(run.out_dir).name or 'output'}/`")
            )
            if row.button(
                "Stop this job",
                key=f"stop-{run.pids[0]}",
                type="secondary",
                icon=":material/stop_circle:",
            ):
                jobs.stop_pids(run.pids)
                st.toast(f"Stopped {run.name}", icon=":material/stop_circle:")
                st.rerun()
        st.caption(
            "Stopping keeps every finished chunk — the job resumes from where "
            "it left off whenever you come back to it."
        )

# --------------------------------------------------------------------------
# queue: what the worker will do next, unattended
# --------------------------------------------------------------------------
# Not shown in PUBLIC_MODE: it lists and can start unattended processing of
# every user's uploads at once, on hardware sized for one job at a time, and
# an unattended worker run has no way to apply the per-account page cap.
if not PUBLIC_MODE:
    st.subheader("Queue")

    worker = jobs.worker_pid()
    if worker:
        st.success(
            f"Worker running (pid {worker}). When a book finishes or stops, the "
            "next one starts on its own.",
            icon=":material/autoplay:",
        )
    else:
        st.warning(
            "No worker running — books will not advance on their own.",
            icon=":material/pause_circle:",
        )

    # The worker is detached, so the button only sends a command; the page
    # state still comes from disk. The command is echoed so a click is never
    # a mystery.
    try:
        worker_log = str(jobs.worker_log_path().relative_to(ROOT))
    except ValueError:
        worker_log = str(jobs.worker_log_path())
    start_preview = shlex.join(jobs.worker_command()) + f" >> {worker_log} 2>&1 &"

    worker_row = st.container(horizontal=True, vertical_alignment="center")
    if worker:
        # Stopping the worker alone leaves its current job (and llama-server's
        # ~2.5 GB) running, which is rarely what "stop" is for.
        also_job = bool(runs) and worker_row.checkbox(
            "Also stop the running job",
            value=True,
            help="Frees the job's memory, llama-server included. Finished chunks are kept.",
        )
        if worker_row.button("Stop worker", icon=":material/stop_circle:"):
            executed = [shlex.join(jobs.stop_worker() or ["# worker had already exited"])]
            if also_job:
                for run in runs:
                    jobs.stop_pids(run.pids)
                    executed.append(shlex.join(["kill", "-TERM", *map(str, run.pids)]))
                executed.append("pkill -f llama-server")
            st.session_state["worker_cmd"] = "\n".join(executed)
            st.toast("Worker stopped", icon=":material/stop_circle:")
            st.rerun()
    else:
        if worker_row.button("Start worker", type="primary", icon=":material/play_circle:"):
            try:
                jobs.start_worker()
                st.session_state["worker_cmd"] = f"cd {shlex.quote(str(ROOT))}\n{start_preview}"
                st.toast("Worker started", icon=":material/play_circle:")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc), icon=":material/error:")

    if "worker_cmd" in st.session_state:
        st.caption("Last command run from this page")
        st.code(st.session_state["worker_cmd"], language="bash")
    elif worker:
        stop_preview = [f"kill -TERM {worker}"]
        if also_job:
            stop_preview += [shlex.join(["kill", "-TERM", *map(str, r.pids)]) for r in runs]
            stop_preview.append("pkill -f llama-server")
        st.caption("Stop runs")
        st.code("\n".join(stop_preview), language="bash")
    else:
        st.caption(f"Start runs (output goes to `{worker_log}`)")
        st.code(start_preview, language="bash")

    queue_states = bt_queue.survey()
    rows = [
        {
            "Document": s.key,
            "Status": s.status,
            "Progress": s.fraction,
            "Chunks": f"{s.chunks_done}/{s.chunks_total}" if s.chunks_total else "—",
            "Skip": s.skip,
        }
        for s in queue_states
    ]
    edited = st.data_editor(
        rows,
        hide_index=True,
        width="stretch",
        disabled=["Document", "Status", "Progress", "Chunks"],
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "Skip": st.column_config.CheckboxColumn(
                "Skip", help="Leave this document out of the queue"
            ),
        },
        key="queue_editor",
    )

    # Persist only what changed; writing every row on every rerun would churn
    # the file and fight the user's next click.
    # strict=True deliberately: these must stay aligned row for row. If they
    # ever diverged, a Skip toggle would be written to the wrong document.
    for original, row in zip(queue_states, edited, strict=True):
        if bool(row.get("Skip")) != original.skip:
            bt_queue.set_skip(original.key, bool(row.get("Skip")))

    nxt = bt_queue.next_pending()
    stalled = [s for s in queue_states if s.status == "stalled"]
    st.caption(
        f"Next up: **{nxt.key}**" if nxt else "Nothing pending — every document is done or skipped."
    )
    if stalled:
        st.warning(
            "Stopped making progress after repeated attempts: "
            + ", ".join(s.key for s in stalled)
            + ". The worker has moved on; clear the attempt count by unskipping or "
            "investigate that document.",
            icon=":material/error:",
        )

# --------------------------------------------------------------------------
# 1. analysis
# --------------------------------------------------------------------------
st.header(pdf_path.name)
info = cached_analysis(str(pdf_path), stat.st_mtime, stat.st_size)

if PUBLIC_MODE and info.output_pages > auth.MAX_PAGES_PER_DOCUMENT:
    st.error(
        f"This document would produce {info.output_pages} pages, over this "
        f"free demo's {auth.MAX_PAGES_PER_DOCUMENT}-page limit per document. "
        "Try a shorter excerpt.",
        icon=":material/block:",
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages in file", info.pages)
c2.metric("Pages to process", info.output_pages)
c3.metric("Text layer", info.text_verdict.replace("_", " "))
c4.metric("Estimated", info.estimate_human())

with st.expander("Why these settings", expanded=True):
    st.markdown(
        f"- **Spreads:** {'yes' if info.is_spread else 'no'} — {info.spread_reason}\n"
        f"- **Text layer:** `{info.text_verdict}` — {info.text_reason}\n"
        f"- **Producer:** `{info.producer or 'none'}`  ·  "
        f"{info.chars_per_page:.0f} chars/page  ·  "
        f"{info.scanned_ratio:.0%} full-page images"
    )

# --------------------------------------------------------------------------
# 2. settings (pre-filled from the analysis, still editable)
# --------------------------------------------------------------------------
st.subheader("Settings")
s1, s2, s3 = st.columns(3)
split = s1.checkbox(
    "Split two-page spreads",
    value=info.recommend_split,
    help="Cuts each page in two at the gutter. Correct for scanned spreads; "
    "on a normal PDF it would cut every page in half.",
)
ocr = s2.checkbox(
    "Run OCR",
    value=info.recommend_ocr,
    help="Off reads the PDF's existing text layer instead — seconds rather than "
    "hours, but only right when that layer is trustworthy.",
)
dpi = s3.select_slider("DPI", [150, 192, 300, 400], value=300,
                       help="Does not change speed — the cost is tokens generated, not pixels read.")
a1, a2 = st.columns(2)
chunk_size = a1.number_input("Pages per chunk", 1, 100, 10,
                             help="Smaller chunks resume more finely and check disk more often.")
pages = a2.text_input("Page range (optional)", "", placeholder="e.g. 0-9 or 8,40-41")

if not ocr and info.recommend_ocr:
    st.warning(
        "The analysis says this document needs OCR. Reading its text layer "
        "instead will reproduce whatever the previous OCR got wrong.",
        icon="⚠️",
    )
if split and not info.is_spread:
    st.warning(
        "No spreads were detected. Splitting will cut every page in half.", icon="⚠️"
    )

# --------------------------------------------------------------------------
# 3. preview -- catch a wrong guess before a multi-hour run
# --------------------------------------------------------------------------
st.subheader("Preview")
# Default to the middle of the document: front matter is often blank or a
# title page, which tells you nothing about whether the settings are right.
idx = st.slider("Page", 0, max(0, info.pages - 1), max(0, info.pages // 2))
if split:
    left, right, gutter, band = render_halves(str(pdf_path), idx, 110, stat.st_mtime)
    st.caption(f"Cut at {gutter:.1%} of width (blank band {band:.1%} wide)")
    p1, p2 = st.columns(2)
    p1.image(left, caption=f"page {idx} · left", width="stretch")
    p2.image(right, caption=f"page {idx} · right", width="stretch")
else:
    st.image(
        render_page(str(pdf_path), idx, 110, stat.st_mtime),
        caption=f"page {idx}",
        width="stretch",
    )

# --------------------------------------------------------------------------
# 4. run and monitor
# --------------------------------------------------------------------------
st.subheader("Run")

busy = bool(runs)
if busy:
    others = [r.name for r in runs if str(pdf_path) != r.pdf]
    st.info(
        (
            f"**{others[0]}** is still running. Stop it in the panel above to "
            "free the machine for this document."
            if others
            else "This document is already running — see the panel above."
        ),
        icon=":material/hourglass_top:",
    )

total_pages = info.pages * 2 if split else info.pages
spec = jobs.JobSpec(
    pdf=str(pdf_path),
    out_dir=str(out_dir),
    split=split,
    ocr=ocr,
    dpi=int(dpi),
    chunk_size=int(chunk_size),
    pages=pages.strip() or None,
    total_pages=total_pages,
)

b1, b2, _ = st.columns([1, 1, 3])
if job.running:
    # Stopping is handled by the session panel above, so this is only a hint.
    b1.button("Running…", disabled=True, icon=":material/sync:")
else:
    if job.chunks_done:
        total_chunks = -(-total_pages // int(chunk_size)) if chunk_size else 0
        label = f"Resume ({job.chunks_done}/{total_chunks or '?'})"
        st.caption(
            f"{job.chunks_done} chunks are already transcribed in "
            f"`{out_dir.name}/chunks/` and will be skipped."
        )
    else:
        label = "Start"
    if b1.button(label, type="primary", disabled=busy, icon=":material/play_arrow:"):
        try:
            jobs.start(spec)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if job.chunks_done and b2.button("Clear progress"):
        for f in (out_dir / "chunks").glob("*.md"):
            f.unlink()
        jobs.clear_lock(out_dir)
        st.rerun()


@st.fragment(run_every=5)
def progress_panel() -> None:
    """Re-read status from disk every few seconds.

    A fragment reruns on its own without re-executing the whole script, so the
    page does not flicker and the analysis is not recomputed on every tick.
    """
    live = jobs.status(out_dir)

    if live.running or live.chunks_done:
        done, total = live.chunks_done, live.chunks_total or spec_total_chunks()
        st.progress(
            min(1.0, done / total) if total else 0.0,
            text=f"{done}/{total} chunks · {done * int(chunk_size)} pages",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Status", "running" if live.running else "stopped")
        rate = jobs.per_page_seconds(live)
        m2.metric("Seconds/page", f"{rate:.0f}" if rate else "—")
        if rate and total:
            remaining = max(0, total - done) * int(chunk_size) * rate
            m3.metric("Remaining", human_time(remaining))
        else:
            m3.metric("Elapsed", human_time(live.elapsed))

    if live.stopped_early and not live.running:
        st.warning(
            "Stopped early to protect free disk space. Press **Resume** to "
            "continue — completed chunks are kept.",
            icon="💾",
        )

    if live.last_lines:
        with st.expander("Log", expanded=live.running):
            st.code("\n".join(live.last_lines), language="text")


def spec_total_chunks() -> int:
    return -(-total_pages // int(chunk_size)) if chunk_size else 0


progress_panel()

# --------------------------------------------------------------------------
# 5. result
# --------------------------------------------------------------------------
final_md = out_dir / f"{pdf_path.stem}.md"
raw_md = out_dir / "raw.md"
result = final_md if final_md.exists() else raw_md

if result.exists():
    st.subheader("Result")
    text = result.read_text(encoding="utf-8", errors="replace")
    st.caption(f"`{result.relative_to(ROOT)}` · {len(text):,} characters")

    from bt.verify import run_all

    source_for_check = out_dir / "pages.pdf" if split else pdf_path
    findings = run_all(
        text,
        pdf_path=str(source_for_check) if source_for_check.exists() else None,
        fresh_ocr=ocr,
    )
    cols = st.columns(len(findings))
    for col, f in zip(cols, findings, strict=True):  # cols built from len(findings)
        col.metric(f.name, "ok" if f.ok else "FAIL", help=f.detail)
    for f in findings:
        if not f.ok:
            st.error(f"**{f.name}** — {f.detail}", icon="⚠️")

    st.download_button(
        "Download Markdown",
        text,
        file_name=final_md.name,
        mime="text/markdown",
        type="primary",
    )
    with st.expander("Preview text"):
        st.text(text[:4000] + ("\n\n… truncated" if len(text) > 4000 else ""))
