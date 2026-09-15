"""Streamlit GUI for the PDF -> Markdown pipeline.

Run with:  uv run streamlit run app.py

The app is a launcher and monitor, not the pipeline itself. Long runs happen in
a detached subprocess (``bt.jobs``) and progress is read back off disk, so the
browser can be closed and reopened -- or the app restarted -- without losing a
run. See bt/jobs.py for why that matters.
"""

from __future__ import annotations

import io
import os
import shlex
import zipfile
from pathlib import Path

import pymupdf
import streamlit as st

from bt import jobs
from bt import queue as bt_queue
from bt.analyze import analyze
from bt.split_spreads import _ink_profile, find_gutter

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"


def shown(path: Path, base: Path) -> str:
    """Path relative to the project when it lives there, in full when it does not.

    ``relative_to`` is strict, and an unhandled ValueError here takes down the
    whole page -- which is what an output folder outside the project used to do.
    """
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)

st.set_page_config(page_title="PDF → Markdown", page_icon="📄", layout="wide")


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


@st.cache_data(show_spinner="Building bundle…")
def bundle_zip(md_path: str, image_paths: tuple[str, ...]) -> bytes:
    """Markdown plus its images, laid out so the links still resolve.

    Keyed on the file list rather than the folder so a new figure invalidates
    it; the zip is rebuilt, not served stale.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_path, Path(md_path).name)
        for image in image_paths:
            zf.write(image, f"images/{Path(image).name}")
    return buf.getvalue()


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# --------------------------------------------------------------------------
# sidebar: choose a document
# --------------------------------------------------------------------------
st.sidebar.title("📄 PDF → Markdown")

# Discovery lives in bt.queue so the GUI and the unattended worker cannot
# disagree about what the library holds -- app.py had a second copy of the same
# glob. Uploaded files are listed because they are saved under uploads/; without
# that they would vanish from the picker the moment the uploader cleared,
# stranding work already done on them.
LIBRARY = bt_queue.ROOT
pdfs = bt_queue.discover()
# Label by path relative to the library, not bare filename: uploads/ files live
# in a subfolder, and resolving a bare name against the root would miss them.
by_label = {shown(p, LIBRARY): p for p in pdfs}
labels = list(by_label) + ["Upload a file…"]
choice = st.sidebar.selectbox("Document", labels, index=0 if pdfs else len(labels) - 1)

pdf_path: Path | None = None
if choice == "Upload a file…":
    up = st.sidebar.file_uploader("Choose a PDF", type="pdf")
    if up is not None:
        dest = LIBRARY / "uploads" / up.name
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


# Resolving this in bt.queue keeps the GUI and the worker agreeing on where a
# document's chunks live. If they disagreed, one could resume onto the other's
# output -- see queue.folder_belongs_to for why ownership comes from the lock.
default_out = bt_queue.resolve_out_dir(pdf_path)
chunks_in = bt_queue.chunks_in

default_label = shown(default_out, LIBRARY)

out_text = st.sidebar.text_input(
    "Output folder",
    value=default_label,
    help="Holds chunks/, raw.md and the final Markdown. Point it at an existing "
    "folder to resume that run.",
)
out_dir = Path(out_text) if Path(out_text).is_absolute() else LIBRARY / out_text
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
            row.markdown(
                f":material/sync: **{run.name}**"
                f"{'  ·  _this document_' if same_doc else ''}  \n"
                f"{run.chunks_done} chunks done · `{Path(run.out_dir).name or 'output'}/`"
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

# The worker is detached, so the button only sends a command; the page state
# still comes from disk. The command is echoed so a click is never a mystery.
worker_log = shown(jobs.worker_log_path(), ROOT)
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

# Persist only what changed; writing every row on every rerun would churn the
# file and fight the user's next click.
# strict=True deliberately: these must stay aligned row for row. If they ever
# diverged, a Skip toggle would be written to the wrong document.
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
images = s3.checkbox(
    "Extract figures",
    value=False,
    help="Saves charts, plates and figures to images/ beside the Markdown and "
    "links them from it. On a text-only book every 'figure' found is a false "
    "positive costing disk, so this is off unless the document has them.",
)
a1, a2, a3 = st.columns(3)
dpi = a1.select_slider("DPI", [150, 192, 300, 400], value=300,
                       help="Does not change speed — the cost is tokens generated, not pixels read.")
chunk_size = a2.number_input("Pages per chunk", 1, 100, 10,
                             help="Smaller chunks resume more finely and check disk more often.")
pages = a3.text_input("Page range (optional)", "", placeholder="e.g. 0-9 or 8,40-41")

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
if info.pages > 1:
    idx = st.slider("Page", 0, info.pages - 1, info.pages // 2)
else:
    # st.slider raises when min == max, which took the whole page down on any
    # one-page document -- there is nothing to choose between, so do not ask.
    idx = 0
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
    images=images,
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


@st.fragment(run_every=5)
def translation_panel(folder: Path, translated_md: Path) -> None:
    """Translation progress and result, re-read from disk like the OCR panel.

    Same reasoning as ``progress_panel``: the job is detached, so the page owns
    no state and a reload costs nothing.
    """
    live = jobs.translate_status(folder)

    if live.running or live.chunks_done:
        done, total = live.chunks_done, live.chunks_total
        st.progress(
            min(1.0, done / total) if total else 0.0,
            text=f"{done}/{total or '?'} chunks",
        )
        st.caption(
            f"{'running' if live.running else 'stopped'} · "
            f"{human_time(live.elapsed)} elapsed"
        )

    if live.stopped_early and not live.running:
        st.warning(
            "Stopped part-way through — the log below says why. Press **Resume** "
            "to carry on; finished chunks are kept.",
            icon=":material/pause_circle:",
        )

    if live.last_lines:
        with st.expander("Translation log", expanded=live.running):
            st.code("\n".join(live.last_lines), language="text")

    if translated_md.exists():
        body = translated_md.read_text(encoding="utf-8", errors="replace")
        st.caption(f"`{translated_md.name}` · {len(body):,} characters")
        st.download_button(
            "Download translation",
            body,
            file_name=translated_md.name,
            mime="text/markdown",
            icon=":material/download:",
            key="download-translation",
        )
        with st.expander("Preview translation"):
            st.text(body[:4000] + ("\n\n… truncated" if len(body) > 4000 else ""))


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
    st.caption(f"`{shown(result, LIBRARY)}` · {len(text):,} characters")

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

    figures = sorted((out_dir / "images").glob("*.*"))

    downloads = st.container(horizontal=True, vertical_alignment="center")
    downloads.download_button(
        "Download Markdown",
        text,
        file_name=final_md.name,
        mime="text/markdown",
        type="primary",
    )
    if figures:
        # The Markdown links images by relative path, so the file alone is not
        # the document -- a bundle is what someone can actually open elsewhere.
        downloads.download_button(
            "Download bundle",
            bundle_zip(str(result), tuple(str(f) for f in figures)),
            file_name=f"{pdf_path.stem}.zip",
            mime="application/zip",
            icon=":material/folder_zip:",
        )

    if figures:
        with st.expander(f"Figures ({len(figures)})"):
            st.caption(f"`{out_dir.name}/images/` — linked from the Markdown by name.")
            grid = st.container(horizontal=True)
            for f in figures[:8]:
                try:
                    grid.image(str(f), caption=f.name, width=180)
                except Exception:  # noqa: BLE001 -- unreadable file, not a page error
                    # One undecodable figure must not take the page down with
                    # it. The Markdown still links the file; the reader can see
                    # for themselves that it is damaged.
                    grid.warning(f"{f.name} could not be read", icon=":material/broken_image:")
            if len(figures) > 8:
                st.caption(f"…and {len(figures) - 8} more.")

    with st.expander("Preview text"):
        st.text(text[:4000] + ("\n\n… truncated" if len(text) > 4000 else ""))

    # ----------------------------------------------------------------------
    # 6. translation -- optional, and a separate job from the pipeline
    # ----------------------------------------------------------------------
    from bt.translate import DEFAULT_PROVIDER, PROVIDERS, default_out_path

    st.subheader("Translate")
    with st.container(border=True):
        t1, t2 = st.columns(2)
        target = t1.text_input(
            "Target language", "English", help="Any language; the source is detected."
        )
        provider_names = sorted(PROVIDERS)
        provider_name = t2.selectbox(
            "Provider",
            provider_names,
            index=provider_names.index(DEFAULT_PROVIDER),
            format_func=lambda n: f"{n} — {PROVIDERS[n].note}",
            help="All of these are free. The local ones need no account and no "
            "network but run on this machine; the hosted ones need a free key "
            "and a request budget.",
        )
        provider = PROVIDERS[provider_name]

        t3, t4 = st.columns([3, 1])
        model = t3.text_input(
            "Model",
            value=provider.model,
            key=f"model-{provider_name}",
            help="Free-tier model ids get retired; if one stops working, put a "
            "current one here. A local server ignores the name and serves "
            "whatever it loaded.",
        )
        t_chunk = t4.number_input("Pages per chunk", 1, 50, 10)

        translated_md = default_out_path(result, target)
        t_spec = jobs.TranslateSpec(
            src=str(result),
            out=str(translated_md),
            target=target,
            provider=provider_name,
            model=model.strip(),
            pages_per_chunk=int(t_chunk),
            total_pages=text.count("<!-- page ") or 1,
        )
        t_job = jobs.translate_status(out_dir)

        pages_to_do = t_spec.total_pages
        if provider.key_env:
            st.caption(
                f"Sends the text of `{result.name}` to {provider.base_url}, one "
                f"request per page ({pages_to_do} pages), paced to "
                f"{provider.rpm}/min. Page anchors, footnote ids and image links "
                "are kept out of the request and re-attached here."
            )
        else:
            st.caption(
                f"Runs against a server on this machine ({provider.base_url}), "
                f"one request per page ({pages_to_do} pages). Nothing leaves the "
                "machine. Page anchors, footnote ids and image links are kept "
                "out of the request and re-attached here."
            )

        if provider.key_env and not os.environ.get(provider.key_env):
            st.warning(
                f"`{provider.key_env}` is not set. It is free to obtain from "
                f"{provider_name}; export it and restart the app.",
                icon=":material/key_off:",
            )
        elif not provider.key_env:
            st.info(
                "Needs a server already running and serving an instruct model — "
                "this is not the one marker spawns for OCR.",
                icon=":material/dns:",
            )

        row = st.container(horizontal=True, vertical_alignment="center")
        if t_job.running:
            row.button("Translating…", disabled=True, icon=":material/sync:")
            if row.button("Stop", icon=":material/stop_circle:", key="stop-translate"):
                # Not kill_inference: llama-server belongs to the OCR job, which
                # may well be running a different book right now.
                jobs.stop_pids([t_job.pid], kill_inference=False)
                st.rerun()
        else:
            label = (
                f"Resume ({t_job.chunks_done}/{t_job.chunks_total or '?'})"
                if t_job.chunks_done
                else f"Translate to {target or 'English'}"
            )
            if row.button(label, type="primary", icon=":material/translate:"):
                try:
                    jobs.start_translate(t_spec, out_dir)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc), icon=":material/error:")
            if t_job.chunks_done and row.button("Clear progress", key="clear-translate"):
                for f in t_spec.chunk_dir.glob("*.md"):
                    f.unlink()
                jobs.translate_lock_path(out_dir).unlink(missing_ok=True)
                st.rerun()

        translation_panel(out_dir, translated_md)
