"""Streamlit GUI for the PDF -> Markdown pipeline.

Run with:  uv run streamlit run app.py

The app is a launcher and monitor, not the pipeline itself. Long runs happen in
a detached subprocess (``bt.jobs``) and progress is read back off disk, so the
browser can be closed and reopened -- or the app restarted -- without losing a
run. See bt/jobs.py for why that matters.

Layout
------
Two scopes share this page, and they used to share one scroll column -- which
is why no section could be told from the next. They are separated here:

* the **machine**: one worker, one OCR job at a time, a queue of books. Global,
  outlives any document, and lives in the sidebar (read-only status) and the
  last tab (its controls).
* the **document** in front of you: analyse, configure, run, read, translate.
  A linear workflow, one step per tab.

Above the tabs sits a status strip that carries the one thing that must not
wait for navigation -- a running job, stoppable from wherever you are.

Tabs rather than ``st.navigation`` pages, deliberately. A page renders only its
own code, so the AppTest smoke suite would stop executing every branch in a
single run, and catching a NameError in a branch the happy path never reaches
is the entire reason that suite exists (see tests/test_app.py). Every tab's
children execute on every run, so the net is unchanged by this layout.
"""

from __future__ import annotations

import hashlib
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


st.set_page_config(
    page_title="PDF → Markdown",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


@st.cache_data(ttl=600, max_entries=8, show_spinner=False)
def list_translation_models(provider_name: str, key_fingerprint: str, _api_key: str):
    """The ids a provider serves right now, or the reason it could not be asked.

    Returns ``(ids, error)`` rather than raising: a failed listing is not a
    failed page, and the panel still has a preset to fall back to.

    ``key_fingerprint`` is in the cache key so that changing the key refetches;
    ``_api_key`` is underscore-prefixed so Streamlit does not hash it, which
    keeps the secret itself out of the cache key. Ten minutes is long enough
    that switching providers back and forth is free, short enough that a
    retirement shows up the same session.
    """
    from bt.translate import PROVIDERS, TranslationError, list_models

    del key_fingerprint
    try:
        return list_models(PROVIDERS[provider_name], _api_key), ""
    except TranslationError as exc:
        return [], str(exc)


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


def human_size(nbytes: int) -> str:
    """Bytes at a scale a reader can use.

    One decimal of MB renders every PDF under a megabyte as "0.0 MB", which is
    the size the sidebar shows for a page range pulled out for testing.
    """
    if nbytes < 1_000_000:
        return f"{nbytes / 1e3:.0f} kB"
    return f"{nbytes / 1e6:.1f} MB"


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def ceil_div(total: int, size: int) -> int:
    """Chunks needed for ``total`` pages -- 0 when the size is not yet known."""
    return -(-total // size) if size else 0


# --------------------------------------------------------------------------
# live panels -- fragments, so a five-second tick redraws one panel instead of
# re-running the whole script (which would re-analyse the PDF each time).
#
# Arguments rather than closures: these are called from inside a tab, and a
# fragment that reads module-level names defined further down is a NameError
# waiting for the first auto-rerun.
# --------------------------------------------------------------------------
@st.fragment(run_every=5)
def progress_panel(folder: Path, pages_per_chunk: int, fallback_chunks: int) -> None:
    """OCR progress, re-read from disk. Chunk files *are* the progress record."""
    live = jobs.status(folder)

    if live.running or live.chunks_done:
        done, total = live.chunks_done, live.chunks_total or fallback_chunks
        st.progress(
            min(1.0, done / total) if total else 0.0,
            text=f"{done}/{total} chunks · {done * pages_per_chunk} pages",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Status", "running" if live.running else "stopped", border=True)
        rate = jobs.per_page_seconds(live)
        m2.metric("Seconds/page", f"{rate:.0f}" if rate else "—", border=True)
        if rate and total:
            remaining = max(0, total - done) * pages_per_chunk * rate
            m3.metric("Remaining", human_time(remaining), border=True)
        else:
            m3.metric("Elapsed", human_time(live.elapsed), border=True)
    else:
        st.caption("Not started. Nothing has been transcribed into this folder yet.")

    if live.stopped_early and not live.running:
        st.warning(
            "Stopped early to protect free disk space. Press **Resume** to "
            "continue — completed chunks are kept.",
            icon=":material/hard_drive:",
        )

    if live.last_lines:
        with st.expander("Log", expanded=live.running):
            st.code("\n".join(live.last_lines), language="text")


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


# --------------------------------------------------------------------------
# sidebar: the document scope -- what to work on, and where its output goes
# --------------------------------------------------------------------------
st.sidebar.markdown("### :material/menu_book: PDF → Markdown")
st.sidebar.caption("Scanned pages in, clean Markdown out.")

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
# Kept first among the sidebar's selectboxes on purpose: the smoke tests address
# it as `at.sidebar.selectbox[0]`, because the tabs below add pickers of their own.
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
    st.info(
        "Choose a PDF in the sidebar to begin — or upload one.",
        icon=":material/upload_file:",
    )
    st.stop()

stat = pdf_path.stat()
st.sidebar.caption(human_size(stat.st_size))

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
    st.sidebar.success(
        f"{existing_chunks} chunks already done here", icon=":material/history:"
    )

# --------------------------------------------------------------------------
# sidebar: the machine scope -- read-only here. Every control that changes it
# lives in the Queue tab, so there is exactly one place to act and one place to
# glance; split across both, neither would be trustworthy.
# --------------------------------------------------------------------------
runs = jobs.active_runs()
job = jobs.status(out_dir)
worker = jobs.worker_pid()

st.sidebar.divider()
st.sidebar.markdown("**Machine**")
if worker:
    st.sidebar.badge(f"Worker running · pid {worker}", icon=":material/autoplay:", color="green")
else:
    st.sidebar.badge("Worker stopped", icon=":material/pause_circle:", color="gray")
if runs:
    st.sidebar.badge(
        f"{len(runs)} job running" if len(runs) == 1 else f"{len(runs)} jobs running",
        icon=":material/sync:",
        color="blue",
    )
else:
    st.sidebar.badge("No job running", icon=":material/check_circle:", color="gray")
st.sidebar.caption("Start, stop and queue controls are in the **Queue** tab.")


# --------------------------------------------------------------------------
# header: which document is open, and the facts every setting below derives from
# --------------------------------------------------------------------------
st.header(pdf_path.name, divider="gray")
st.caption(
    f":material/description: `{shown(pdf_path, LIBRARY)}`  →  "
    f":material/folder: `{shown(out_dir, LIBRARY)}/`"
)

info = cached_analysis(str(pdf_path), stat.st_mtime, stat.st_size)

# Bordered metrics rather than a bare row: at a glance this is one card of
# document facts, not four numbers floating above whatever section follows.
f1, f2, f3, f4 = st.columns(4)
f1.metric("Pages in file", info.pages, border=True)
f2.metric("Pages to process", info.output_pages, border=True)
f3.metric("Text layer", info.text_verdict.replace("_", " "), border=True)
f4.metric("Estimated", info.estimate_human(), border=True)

# --------------------------------------------------------------------------
# status strip: above the tabs, so a running job can be stopped from anywhere.
# Only one pipeline may run at a time on this machine, and a run started from a
# terminal has no lock file -- so this reports every one found, not just ours.
# --------------------------------------------------------------------------
if runs:
    with st.container(border=True):
        st.markdown(
            ":material/sync: **Job in progress** — only one runs at a time."
        )
        for run in runs:
            row = st.container(horizontal=True, vertical_alignment="center")
            same_doc = str(pdf_path) == run.pdf
            row.markdown(
                f"**{run.name}**"
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

# Paths the Result and Translate tabs both need, resolved once.
final_md = out_dir / f"{pdf_path.stem}.md"
raw_md = out_dir / "raw.md"
result = final_md if final_md.exists() else raw_md

tab_transcribe, tab_result, tab_translate, tab_queue = st.tabs(
    [
        ":material/play_circle: Transcribe",
        ":material/description: Result",
        ":material/translate: Translate",
        ":material/queue: Queue",
    ],
    key="section",
)

# ==========================================================================
# Transcribe -- settings, the preview that checks them, and the run itself.
# One tab, because configuring and launching are one decision: splitting them
# would put a navigation click between "this looks right" and "go".
# ==========================================================================
with tab_transcribe:
    settings_col, preview_col = st.columns([2, 3], gap="medium")

    with settings_col:
        st.subheader("Settings")
        with st.container(border=True):
            # Pre-filled from the analysis, still editable -- the analysis is a
            # strong guess, not an authority.
            split = st.checkbox(
                "Split two-page spreads",
                value=info.recommend_split,
                help="Cuts each page in two at the gutter. Correct for scanned "
                "spreads; on a normal PDF it would cut every page in half.",
            )
            ocr = st.checkbox(
                "Run OCR",
                value=info.recommend_ocr,
                help="Off reads the PDF's existing text layer instead — seconds "
                "rather than hours, but only right when that layer is trustworthy.",
            )
            images = st.checkbox(
                "Extract figures",
                value=False,
                help="Saves charts, plates and figures to images/ beside the "
                "Markdown and links them from it. On a text-only book every "
                "'figure' found is a false positive costing disk, so this is off "
                "unless the document has them.",
            )
            st.divider()
            dpi = st.select_slider(
                "DPI",
                [150, 192, 300, 400],
                value=300,
                help="Does not change speed — the cost is tokens generated, not "
                "pixels read.",
            )
            chunk_size = st.number_input(
                "Pages per chunk",
                1,
                100,
                10,
                help="Smaller chunks resume more finely and check disk more often.",
            )
            pages = st.text_input(
                "Page range (optional)", "", placeholder="e.g. 0-9 or 8,40-41"
            )

        if not ocr and info.recommend_ocr:
            st.warning(
                "The analysis says this document needs OCR. Reading its text "
                "layer instead will reproduce whatever the previous OCR got wrong.",
                icon=":material/warning:",
            )
        if split and not info.is_spread:
            st.warning(
                "No spreads were detected. Splitting will cut every page in half.",
                icon=":material/warning:",
            )

        with st.expander("Why these settings", expanded=False):
            st.markdown(
                f"- **Spreads:** {'yes' if info.is_spread else 'no'} — {info.spread_reason}\n"
                f"- **Text layer:** `{info.text_verdict}` — {info.text_reason}\n"
                f"- **Producer:** `{info.producer or 'none'}`  ·  "
                f"{info.chars_per_page:.0f} chars/page  ·  "
                f"{info.scanned_ratio:.0%} full-page images"
            )

    with preview_col:
        # Beside the settings, not below them: the preview exists to catch a
        # wrong guess before a multi-hour run, which it can only do if the guess
        # and its consequence are on screen together.
        st.subheader("Preview")
        with st.container(border=True):
            # Default to the middle of the document: front matter is often blank
            # or a title page, which tells you nothing about the settings.
            if info.pages > 1:
                idx = st.slider("Page", 0, info.pages - 1, info.pages // 2)
            else:
                # st.slider raises when min == max, which took the whole page
                # down on any one-page document -- there is nothing to choose
                # between, so do not ask.
                idx = 0
            if split:
                left, right, gutter, band = render_halves(
                    str(pdf_path), idx, 110, stat.st_mtime
                )
                p1, p2 = st.columns(2)
                p1.image(left, caption=f"page {idx} · left", width="stretch")
                p2.image(right, caption=f"page {idx} · right", width="stretch")
                st.caption(f"Cut at {gutter:.1%} of width (blank band {band:.1%} wide)")
            else:
                st.image(
                    render_page(str(pdf_path), idx, 110, stat.st_mtime),
                    caption=f"page {idx}",
                    width="stretch",
                )

    st.subheader("Run")
    total_pages = info.pages * 2 if split else info.pages
    total_chunks = ceil_div(total_pages, int(chunk_size))
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

    with st.container(border=True):
        busy = bool(runs)
        if busy:
            others = [r.name for r in runs if str(pdf_path) != r.pdf]
            st.info(
                (
                    f"**{others[0]}** is still running. Stop it in the strip "
                    "above to free the machine for this document."
                    if others
                    else "This document is already running — see the strip above."
                ),
                icon=":material/hourglass_top:",
            )

        controls = st.container(horizontal=True, vertical_alignment="center")
        if job.running:
            # Stopping is handled by the status strip above, so this is a hint.
            controls.button("Running…", disabled=True, icon=":material/sync:")
        else:
            if job.chunks_done:
                label = f"Resume ({job.chunks_done}/{total_chunks or '?'})"
            else:
                label = "Start"
            if controls.button(
                label, type="primary", disabled=busy, icon=":material/play_arrow:"
            ):
                try:
                    jobs.start(spec)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc), icon=":material/error:")
            if job.chunks_done and controls.button(
                "Clear progress", icon=":material/delete_sweep:"
            ):
                for f in (out_dir / "chunks").glob("*.md"):
                    f.unlink()
                jobs.clear_lock(out_dir)
                st.rerun()
        if job.chunks_done and not job.running:
            st.caption(
                f"{job.chunks_done} chunks are already transcribed in "
                f"`{out_dir.name}/chunks/` and will be skipped."
            )

        progress_panel(out_dir, int(chunk_size), total_chunks)

# ==========================================================================
# Result -- the Markdown, what verification makes of it, and how to take it away
# ==========================================================================
with tab_result:
    if not result.exists():
        st.info(
            "No Markdown yet. Start the run in **Transcribe**; output appears "
            "here as soon as the first chunks are written.",
            icon=":material/hourglass_empty:",
        )
    else:
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
            col.metric(f.name, "ok" if f.ok else "FAIL", help=f.detail, border=True)
        for f in findings:
            if not f.ok:
                st.error(f"**{f.name}** — {f.detail}", icon=":material/warning:")

        figures = sorted((out_dir / "images").glob("*.*"))

        downloads = st.container(horizontal=True, vertical_alignment="center")
        downloads.download_button(
            "Download Markdown",
            text,
            file_name=final_md.name,
            mime="text/markdown",
            type="primary",
            icon=":material/download:",
        )
        if figures:
            # The Markdown links images by relative path, so the file alone is
            # not the document -- a bundle is what someone can open elsewhere.
            downloads.download_button(
                "Download bundle",
                bundle_zip(str(result), tuple(str(f) for f in figures)),
                file_name=f"{pdf_path.stem}.zip",
                mime="application/zip",
                icon=":material/folder_zip:",
            )

        if figures:
            with st.expander(f"Figures ({len(figures)})"):
                st.caption(
                    f"`{out_dir.name}/images/` — linked from the Markdown by name."
                )
                grid = st.container(horizontal=True)
                for f in figures[:8]:
                    try:
                        grid.image(str(f), caption=f.name, width=180)
                    except Exception:  # noqa: BLE001 -- unreadable file, not a page error
                        # One undecodable figure must not take the page down
                        # with it. The Markdown still links the file; the reader
                        # can see for themselves that it is damaged.
                        grid.warning(
                            f"{f.name} could not be read",
                            icon=":material/broken_image:",
                        )
                if len(figures) > 8:
                    st.caption(f"…and {len(figures) - 8} more.")

        with st.expander("Preview text"):
            st.text(text[:4000] + ("\n\n… truncated" if len(text) > 4000 else ""))

# ==========================================================================
# Translate -- optional stage 4, and a separate job from the pipeline
# ==========================================================================
with tab_translate:
    if not result.exists():
        st.info(
            "Nothing to translate yet — this stage runs on the finished "
            "Markdown, not on the PDF.",
            icon=":material/translate:",
        )
    else:
        from bt.translate import DEFAULT_PROVIDER, PROVIDERS, default_out_path

        st.caption(
            f"Source: `{result.name}` · {len(text):,} characters. Not part of "
            "the pipeline and not something the worker does — translating a "
            "transcription you have not read only multiplies its mistakes."
        )

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
                help="All of these are free. The local ones need no account and "
                "no network but run on this machine; the hosted ones need a free "
                "key and a request budget.",
            )
            provider = PROVIDERS[provider_name]

            # The key, pasted rather than exported. Exporting one means
            # restarting the app, which on a machine mid-run is the most
            # expensive way to supply a string. This field is the same key by a
            # shorter path: jobs.start_translate() puts it in the child's
            # environment, which is exactly where an exported one would have
            # been read from.
            #
            # Keyed per provider on purpose. One shared box would carry a groq
            # key into a gemini run -- refused by the server, and a secret sent
            # to a service it was not issued for.
            #
            # It sits above the model picker because the picker depends on it:
            # the list of models is fetched from the provider, and the provider
            # will not answer without the key.
            api_key = ""
            if provider.key_env:
                api_key = (
                    st.text_input(
                        f"API key for {provider_name}",
                        type="password",
                        key=f"apikey-{provider_name}",
                        placeholder=f"Paste it here, or export ${provider.key_env}",
                        icon=":material/key:",
                        help="Free to obtain from the provider. Kept in this "
                        "browser session only: it is never written to disk, never "
                        "put on a command line, and is handed to the run through "
                        "its environment. Leave it empty to use the exported "
                        "variable.",
                    )
                    or ""
                ).strip()

            env_key = os.environ.get(provider.key_env or "", "")
            if not provider.key_env:
                st.info(
                    "Needs a server already running and serving an instruct "
                    "model — this is not the one marker spawns for OCR.",
                    icon=":material/dns:",
                )
            elif api_key:
                st.caption(
                    f"Using the key above for this session. `{provider.key_env}` "
                    "is not modified, and the key is not saved anywhere."
                )
            elif env_key:
                st.caption(f"Using `{provider.key_env}` from the environment.")
            else:
                st.warning(
                    f"No key yet. `{provider.key_env}` is not set either — paste "
                    f"one above; it is free to obtain from {provider_name}.",
                    icon=":material/key_off:",
                )

            # Ask the provider what it serves rather than trusting the preset. A
            # retired id is the normal way this stage breaks -- groq dropped
            # llama-3.3-70b-versatile while it was still the default here -- and
            # the answer is one GET away on the same OpenAI-compatible surface
            # the translation itself uses.
            #
            # Only when there is a key to ask with: a keyless request just earns
            # a 401, and showing the user that instead of the preset is worse
            # than not having asked. A local server needs no key and is asked
            # always.
            listed: list[str] = []
            list_error = ""
            usable_key = api_key or env_key
            if usable_key or not provider.key_env:
                listed, list_error = list_translation_models(
                    provider_name,
                    hashlib.sha256(usable_key.encode()).hexdigest()[:16],
                    usable_key,
                )

            t3, t4 = st.columns([3, 1])
            # The preset stays available when nothing was listed, and remains the
            # default when it is still served. accept_new_options keeps the field
            # typeable either way: a list is a better starting point than a
            # constant, but it is not a constraint -- a provider can serve a
            # model its own listing omits.
            options = listed or [provider.model]
            model = t3.selectbox(
                "Model",
                options,
                index=options.index(provider.model) if provider.model in options else 0,
                key=f"model-{provider_name}",
                accept_new_options=True,
                help="Listed by the provider itself, filtered to models that can "
                "take a page of text and give one back. Type an id to use one "
                "that is not listed. A local server ignores the name and serves "
                "whatever it loaded.",
            )
            t_chunk = t4.number_input("Pages per chunk", 1, 50, 10)

            refresh = st.container(horizontal=True, vertical_alignment="center")
            if listed:
                refresh.caption(
                    f"{len(listed)} models listed by {provider_name}, cached for "
                    "10 minutes."
                )
            elif list_error:
                refresh.caption(
                    f"Could not list models ({list_error.split('.')[0]}). Showing "
                    "the built-in default; type an id to override it."
                )
            elif provider.key_env:
                refresh.caption(
                    "Add a key above to list the models this provider serves."
                )
            if refresh.button("Refresh", icon=":material/refresh:", key="refresh-models"):
                list_translation_models.clear()
                st.rerun()

            translated_md = default_out_path(result, target)
            t_spec = jobs.TranslateSpec(
                src=str(result),
                out=str(translated_md),
                target=target,
                provider=provider_name,
                model=(model or provider.model).strip(),
                pages_per_chunk=int(t_chunk),
                total_pages=text.count("<!-- page ") or 1,
            )
            t_job = jobs.translate_status(out_dir)

            pages_to_do = t_spec.total_pages
            if provider.key_env:
                st.caption(
                    f"Sends the text of `{result.name}` to {provider.base_url}, "
                    f"one request per page ({pages_to_do} pages), paced to "
                    f"{provider.rpm}/min. Page anchors, footnote ids and image "
                    "links are kept out of the request and re-attached here."
                )
            else:
                st.caption(
                    f"Runs against a server on this machine ({provider.base_url}), "
                    f"one request per page ({pages_to_do} pages). Nothing leaves "
                    "the machine. Page anchors, footnote ids and image links are "
                    "kept out of the request and re-attached here."
                )

            row = st.container(horizontal=True, vertical_alignment="center")
            if t_job.running:
                row.button("Translating…", disabled=True, icon=":material/sync:")
                if row.button("Stop", icon=":material/stop_circle:", key="stop-translate"):
                    # Not kill_inference: llama-server belongs to the OCR job,
                    # which may well be running a different book right now.
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
                        jobs.start_translate(t_spec, out_dir, api_key=api_key)
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc), icon=":material/error:")
                if t_job.chunks_done and row.button(
                    "Clear progress", key="clear-translate"
                ):
                    for f in t_spec.chunk_dir.glob("*.md"):
                        f.unlink()
                    jobs.translate_lock_path(out_dir).unlink(missing_ok=True)
                    st.rerun()

            translation_panel(out_dir, translated_md)

# ==========================================================================
# Queue -- the machine scope: the unattended worker and what it will do next
# ==========================================================================
with tab_queue:
    st.subheader("Worker")
    with st.container(border=True):
        if worker:
            st.success(
                f"Worker running (pid {worker}). When a book finishes or stops, "
                "the next one starts on its own.",
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
        worker_log = shown(jobs.worker_log_path(), ROOT)
        start_preview = shlex.join(jobs.worker_command()) + f" >> {worker_log} 2>&1 &"

        also_job = False
        worker_row = st.container(horizontal=True, vertical_alignment="center")
        if worker:
            # Stopping the worker alone leaves its current job (and
            # llama-server's ~2.5 GB) running, which is rarely what "stop" is for.
            also_job = bool(runs) and worker_row.checkbox(
                "Also stop the running job",
                value=True,
                help="Frees the job's memory, llama-server included. Finished "
                "chunks are kept.",
            )
            if worker_row.button("Stop worker", icon=":material/stop_circle:"):
                executed = [
                    shlex.join(jobs.stop_worker() or ["# worker had already exited"])
                ]
                if also_job:
                    for run in runs:
                        jobs.stop_pids(run.pids)
                        executed.append(shlex.join(["kill", "-TERM", *map(str, run.pids)]))
                    executed.append("pkill -f llama-server")
                st.session_state["worker_cmd"] = "\n".join(executed)
                st.toast("Worker stopped", icon=":material/stop_circle:")
                st.rerun()
        else:
            if worker_row.button(
                "Start worker", type="primary", icon=":material/play_circle:"
            ):
                try:
                    jobs.start_worker()
                    st.session_state["worker_cmd"] = (
                        f"cd {shlex.quote(str(ROOT))}\n{start_preview}"
                    )
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
                stop_preview += [
                    shlex.join(["kill", "-TERM", *map(str, r.pids)]) for r in runs
                ]
                stop_preview.append("pkill -f llama-server")
            st.caption("Stop runs")
            st.code("\n".join(stop_preview), language="bash")
        else:
            st.caption(f"Start runs (output goes to `{worker_log}`)")
            st.code(start_preview, language="bash")

    st.subheader("Queue")
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
        f"Next up: **{nxt.key}**"
        if nxt
        else "Nothing pending — every document is done or skipped."
    )
    if stalled:
        st.warning(
            "Stopped making progress after repeated attempts: "
            + ", ".join(s.key for s in stalled)
            + ". The worker has moved on; clear the attempt count by unskipping "
            "or investigate that document.",
            icon=":material/error:",
        )
