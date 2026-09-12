"""Streamlit GUI for the PDF -> Markdown pipeline.

Run with:  uv run streamlit run app.py

The app is a launcher and monitor, not the pipeline itself. Long runs happen in
a detached subprocess (``bt.jobs``) and progress is read back off disk, so the
browser can be closed and reopened -- or the app restarted -- without losing a
run. See bt/jobs.py for why that matters.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import streamlit as st

from bt import jobs
from bt.analyze import analyze
from bt.split_spreads import _ink_profile, find_gutter

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"

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


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def find_pdfs() -> list[Path]:
    return sorted(
        p for p in ROOT.glob("*.pdf") if not p.name.startswith(".")
    )


# --------------------------------------------------------------------------
# sidebar: choose a document
# --------------------------------------------------------------------------
st.sidebar.title("📄 PDF → Markdown")

pdfs = find_pdfs()
labels = [p.name for p in pdfs] + ["Upload a file…"]
choice = st.sidebar.selectbox("Document", labels, index=0 if pdfs else len(labels) - 1)

pdf_path: Path | None = None
if choice == "Upload a file…":
    up = st.sidebar.file_uploader("Choose a PDF", type="pdf")
    if up is not None:
        dest = ROOT / "uploads" / up.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != up.size:
            dest.write_bytes(up.getbuffer())
        pdf_path = dest
else:
    pdf_path = ROOT / choice

if pdf_path is None or not pdf_path.exists():
    st.info("Choose a PDF in the sidebar to begin.")
    st.stop()

stat = pdf_path.stat()
out_dir = OUTPUT_ROOT / pdf_path.stem
st.sidebar.caption(f"{stat.st_size / 1e6:.1f} MB\n\nOutput: `output/{pdf_path.stem}/`")

# A pipeline running anywhere on this machine matters, not just one we started.
foreign = jobs.find_pipeline_processes()
job = jobs.status(out_dir)

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
idx = st.slider("Page", 0, max(0, info.pages - 1), min(1, max(0, info.pages - 1)))
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

if foreign and not job.running:
    st.error(
        f"A pipeline is already running on this machine (pid "
        f"{', '.join(str(p) for p, _ in foreign)}). Two at once push this "
        "machine into swap and slow both down.",
        icon="🚫",
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
    if b1.button("Stop", type="secondary"):
        jobs.stop(out_dir)
        st.rerun()
else:
    label = "Resume" if job.chunks_done else "Start"
    if b1.button(label, type="primary", disabled=bool(foreign)):
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
    for col, f in zip(cols, findings):
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
