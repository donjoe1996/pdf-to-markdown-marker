"""Stage 2: run marker over the split single-page PDF.

Notes on marker 2.0 (the API changed substantially from 1.x):

* Models no longer run in-process. marker auto-spawns a surya VLM inference
  server -- ``llama-server`` on Apple Silicon. ``SURYA_INFERENCE_KEEP_ALIVE``
  keeps it up between chunks so startup is paid once, not once per chunk.
* ``PYTORCH_ENABLE_MPS_FALLBACK`` is set by marker's *CLI* but not by the
  library, so it has to be set here -- and before torch is imported.
* On MPS, ``mode`` defaults to ``"fast"``, which reads text from the PDF's
  existing text layer via pdftext. This book's embedded layer is bad Acrobat
  OCR, so ``force_ocr`` + ``strip_existing_ocr`` are mandatory, not optional
  tuning. See ``verify.py`` for the check that they actually took effect.
"""

from __future__ import annotations

import os

# Must precede any torch/marker import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # transformers uses .isin
os.environ.setdefault("SURYA_INFERENCE_KEEP_ALIVE", "1")  # reuse the VLM server
# See bt/warmup.py: the Hub's Xet CDN bridge can hang at 0 bytes. Set here too
# in case marker reaches for a weight the warmup stage did not cover.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# marker starts each model in its own server subprocess and waits for a health
# check. The defaults (300s) assume weights are already cached; a first run that
# is still downloading blows through them and the server gets force-killed. The
# warmup stage makes this moot, but these raise the ceiling for a slow link or a
# cold page cache.
os.environ.setdefault("OCR_ERROR_SERVER_STARTUP_TIMEOUT", "1800")
os.environ.setdefault("DETECTOR_SERVER_STARTUP_TIMEOUT", "1800")
os.environ.setdefault("FAST_LAYOUT_SERVER_STARTUP_TIMEOUT", "1800")
os.environ.setdefault("SURYA_INFERENCE_STARTUP_TIMEOUT", "1800")

import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from bt.images import IMAGE_DIR_NAME, stage_chunk  # noqa: E402

DEFAULT_CHUNK_SIZE = 20

# llama-server's working set (~2.4 GB) drives heavy swap on a full volume, and
# macOS grows swapfiles on the same disk. Over a long unattended run that can
# fill it, so the chunk loop stops cleanly while there is still room rather than
# failing mid-page. Completed chunks are already on disk, so a stop costs
# nothing: re-running resumes where it left off.
MIN_FREE_GB = 2.0


def build_config(
    page_range: str | None = None,
    mode: str = "fast",
    dpi: int = 300,
    use_llm: bool = False,
    ocr: bool = True,
    images: bool = True,
) -> dict:
    """Config dict for ``ConfigParser``.

    ``page_range`` is a *string* here because ConfigParser runs it through
    ``parse_range_str``; a hand-built config would need a list[int] instead.

    ``images=False`` turns figure extraction off. It is on because a map or a
    plate is part of the document, and marker will not hand the images back
    later without re-running the OCR. The escape hatch exists because layout
    detection can read a noisy scan's whole page as a picture; the figure count
    in the post-processing report is what makes that visible.

    ``ocr=False`` is the born-digital fast path: marker reads the PDF's existing
    text layer instead of running the vision model, which turns hours into
    seconds. Only correct when that layer is trustworthy -- ``bt.analyze`` makes
    that call. For a scan it would reproduce whatever the old OCR got wrong.
    """
    config: dict = {
        "output_format": "markdown",  # ConfigParser KeyErrors without this
        "highres_image_dpi": dpi,
        "paginate_output": True,  # keeps book pages addressable downstream
        # -> extract_images. Off means the figures are simply dropped: marker
        # writes the caption and nothing else, so the loss is not obvious.
        "disable_image_extraction": not images,
        "mode": mode,
    }
    if ocr:
        # Ignore any embedded text layer and re-read the page from the image.
        config["force_ocr"] = True
        config["strip_existing_ocr"] = True
    else:
        # Pure text-layer extraction; turns off all VLM calls.
        config["disable_ocr"] = True
    if page_range is not None:
        config["page_range"] = page_range
    if use_llm:
        # Off by default. Wiring is kept so a bad-Greek page range can be
        # redone later without restructuring anything.
        config["use_llm"] = True
    return config


def make_converter(config: dict, artifacts: dict):
    """Build a PdfConverter over an already-created artifact dict.

    ``artifacts`` is created once per run and reused: it owns the handles to the
    model servers, so rebuilding it per chunk would re-do that setup. The
    converter itself is cheap and has to be rebuilt because ``page_range`` is
    baked into its config.
    """
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter

    parser = ConfigParser(config)
    return PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=artifacts,
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service(),
    )


def convert_range(converter, pdf_path: Path) -> tuple[str, dict]:
    """Return the chunk's Markdown *and* the figures it refers to.

    The images were discarded here before, which is why an illustrated book
    came out with captions standing over nothing.
    """
    from marker.output import text_from_rendered

    rendered = converter(str(pdf_path))
    text, _ext, images = text_from_rendered(rendered)
    return text, images or {}


def transcribe(
    pdf_path: Path,
    out_md: Path,
    chunk_dir: Path,
    total_pages: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mode: str = "fast",
    dpi: int = 300,
    use_llm: bool = False,
    resume: bool = True,
    ocr: bool = True,
    images: bool = True,
) -> Path:
    """OCR ``pdf_path`` in resumable chunks and concatenate to ``out_md``.

    A full run is measured in hours, so each chunk is written as it completes
    and an existing chunk file is skipped on re-run.

    Extracted figures go to ``<out_md.parent>/images/`` and are linked relative
    to the finished Markdown. They are saved *before* the chunk file is renamed
    into place, because the chunk file is the record that its pages are done:
    written the other way round, a kill in between would leave a resumed run
    permanently missing those figures with nothing to notice it.
    """
    from marker.models import create_model_dict, shutdown_models

    chunk_dir.mkdir(parents=True, exist_ok=True)
    bounds = [
        (s, min(s + chunk_size - 1, total_pages - 1))
        for s in range(0, total_pages, chunk_size)
    ]

    pending = [
        (a, b) for a, b in bounds if not (resume and _chunk_path(chunk_dir, a, b).exists())
    ]
    print(
        f"{len(bounds)} chunks of {chunk_size} pages; "
        f"{len(bounds) - len(pending)} already done, {len(pending)} to run."
    )

    if not pending:
        return concatenate(chunk_dir, out_md, bounds)

    image_dir = out_md.parent / IMAGE_DIR_NAME
    artifacts = create_model_dict()
    stopped_early = False
    try:
        for n, (first, last) in enumerate(pending, 1):
            free = shutil.disk_usage("/").free / 1e9
            if free < MIN_FREE_GB:
                print(
                    f"\n  STOPPING: only {free:.1f} GB free (floor {MIN_FREE_GB} GB).\n"
                    f"  {n - 1} of {len(pending)} chunks done and saved. "
                    "Free some space and re-run the same command to resume."
                )
                stopped_early = True
                break

            target = _chunk_path(chunk_dir, first, last)
            started = time.time()
            config = build_config(
                f"{first}-{last}",
                mode=mode,
                dpi=dpi,
                use_llm=use_llm,
                ocr=ocr,
                images=images,
            )
            converter = make_converter(config, artifacts)
            text, figures = convert_range(converter, pdf_path)
            if figures:
                text = stage_chunk(text, figures, image_dir, target.stem)
            # Write via a temp file and rename: a chunk killed mid-write would
            # otherwise look complete on resume and silently truncate the book.
            tmp = target.with_suffix(".partial")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(target)
            elapsed = time.time() - started
            pages = last - first + 1
            figure_note = f", {len(figures)} figures" if figures else ""
            print(
                f"  [{n}/{len(pending)}] pages {first}-{last} -> {target.name} "
                f"({elapsed:.0f}s, {elapsed / pages:.1f}s/page{figure_note})"
            )
    finally:
        shutdown_models(artifacts)

    # A disk-guard stop leaves later chunks unwritten; join what exists so the
    # partial result is still usable, rather than failing on the missing ones.
    return concatenate(chunk_dir, out_md, None if stopped_early else bounds)


def _chunk_path(chunk_dir: Path, first: int, last: int) -> Path:
    return chunk_dir / f"{first:04d}-{last:04d}.md"


def concatenate(
    chunk_dir: Path, out_md: Path, bounds: list[tuple[int, int]] | None = None
) -> Path:
    """Join chunk files in page order.

    ``bounds`` restricts the join to the chunks this run expects. Without it a
    changed ``--chunk-size`` would leave stale chunks with different boundaries
    in the directory (say 0000-0019.md alongside 0000-0009.md), and globbing
    would concatenate overlapping page ranges into a duplicated book.
    """
    if bounds is not None:
        chunks = [_chunk_path(chunk_dir, a, b) for a, b in bounds]
        missing = [p.name for p in chunks if not p.exists()]
        if missing:
            raise RuntimeError(f"missing chunk output: {', '.join(missing)}")
    else:
        # Match the chunk naming exactly, so an unrelated .md sitting in the
        # directory (an output file, a stray note) is ignored rather than
        # crashing the sort or being spliced into the book.
        named = [
            (int(m.group(1)), p)
            for p in chunk_dir.glob("*.md")
            if (m := re.fullmatch(r"(\d{4})-(\d{4})", p.stem))
        ]
        chunks = [p for _, p in sorted(named)]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n\n".join(c.read_text(encoding="utf-8").strip() for c in chunks) + "\n",
        encoding="utf-8",
    )
    print(f"Concatenated {len(chunks)} chunks -> {out_md}")
    return out_md


def main(argv: list[str] | None = None) -> int:
    import argparse

    import pymupdf

    ap = argparse.ArgumentParser(description="OCR a split PDF to Markdown via marker.")
    ap.add_argument("pdf", type=Path, help="split single-page PDF (from stage 1)")
    ap.add_argument("out", type=Path, help="output Markdown file")
    ap.add_argument("--chunk-dir", type=Path, default=None)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument(
        "--mode",
        choices=["fast", "balanced"],
        default="fast",
        help="marker mode; 'fast' matches 'balanced' on multi-column layout "
        "(76.0 vs 76.6 on marker's benchmark) at much lower cost",
    )
    ap.add_argument("--dpi", type=int, default=300, help="highres_image_dpi")
    ap.add_argument("--use-llm", action="store_true", help="enable LLM hybrid mode")
    ap.add_argument("--no-resume", action="store_true", help="redo completed chunks")
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="drop figures instead of saving them next to the Markdown",
    )
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="read the existing text layer instead of OCRing (born-digital PDFs "
        "only -- on a scan this reproduces the old OCR's mistakes)",
    )
    args = ap.parse_args(argv)

    with pymupdf.open(args.pdf) as doc:
        total = doc.page_count

    chunk_dir = args.chunk_dir or args.out.parent / "chunks"
    transcribe(
        args.pdf,
        args.out,
        chunk_dir,
        total_pages=total,
        chunk_size=args.chunk_size,
        mode=args.mode,
        dpi=args.dpi,
        use_llm=args.use_llm,
        resume=not args.no_resume,
        ocr=not args.no_ocr,
        images=not args.no_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
