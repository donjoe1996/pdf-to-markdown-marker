"""Stage 1: split scanned 2-page spreads into single book pages.

Every page of the source PDF is one landscape scan holding two facing book
pages. Feeding that to any layout model invites cross-gutter line merging and
makes per-page headers/footnotes ambiguous, so the spreads are cut first.

The cut is found per page rather than assumed at 50%: measured gutter centres
across this book land between 0.497 and 0.514 of page width.

Halves are emitted with ``set_cropbox``, which is lossless -- the page content
is untouched, only the visible box changes. That matters for quality: marker
then resamples the original 300 DPI scan exactly once. Rasterising here and
letting marker rasterise again would resample twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pymupdf

# Ink is projected at this DPI purely to locate the gutter; it needs to be
# cheap, not precise. 36 DPI over a ~760 pt page gives ~380 columns.
PROBE_DPI = 36
# The gutter is searched only in the middle of the page so that the wide blank
# outer margins can never win.
SEARCH_LO, SEARCH_HI = 0.35, 0.65
# A half holding less than this share of the spread's ink is blank (6 such
# halves exist in this book: source pages 0, 2, 8, 9, 32, 250).
BLANK_INK_FRACTION = 0.02


@dataclass
class PageRecord:
    """Where an output page came from, for tracing problems back."""

    out_page: int
    src_page: int
    half: str  # "left" | "right" | "full"
    gutter_frac: float


def _ink_profile(page: pymupdf.Page) -> np.ndarray:
    """Count dark pixels per pixel-column."""
    pm = page.get_pixmap(dpi=PROBE_DPI, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.stride)
    arr = arr[:, : pm.width]  # drop row padding
    return (arr < 128).sum(axis=0)


def find_gutter(ink: np.ndarray) -> float:
    """Return the gutter position as a fraction of page width.

    Uses the *widest* run of zero-ink columns in the central band. Taking
    ``argmin`` instead does not work: the entire gutter reads zero, so argmin
    returns whichever zero it meets first and drifts toward the band edge.
    """
    width = len(ink)
    lo, hi = int(width * SEARCH_LO), int(width * SEARCH_HI)
    blank = ink[lo:hi] == 0

    best_len = best_start = 0
    run = start = 0
    for i, is_blank in enumerate(blank):
        if is_blank:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0

    if best_len == 0:
        return 0.5  # no clear gutter; fall back to the midpoint
    return (lo + best_start + best_len / 2) / width


def split_document(
    src_path: Path,
    out_pdf: Path,
    pagemap_path: Path | None = None,
    page_range: list[int] | None = None,
) -> list[PageRecord]:
    """Split every spread in ``src_path``; write single pages to ``out_pdf``."""
    src = pymupdf.open(src_path)
    out = pymupdf.open()
    records: list[PageRecord] = []

    pages = page_range if page_range is not None else range(src.page_count)
    for src_idx in pages:
        page = src[src_idx]
        rect = page.rect
        ink = _ink_profile(page)
        gutter = find_gutter(ink)

        cut_px = int(gutter * len(ink))
        left_ink, right_ink = int(ink[:cut_px].sum()), int(ink[cut_px:].sum())
        total_ink = left_ink + right_ink
        if total_ink == 0:
            continue  # fully blank spread

        cut_x = rect.x0 + gutter * rect.width
        halves = [
            ("left", pymupdf.Rect(rect.x0, rect.y0, cut_x, rect.y1), left_ink),
            ("right", pymupdf.Rect(cut_x, rect.y0, rect.x1, rect.y1), right_ink),
        ]

        for name, clip, half_ink in halves:
            if half_ink < total_ink * BLANK_INK_FRACTION:
                continue
            out.insert_pdf(src, from_page=src_idx, to_page=src_idx)
            out[-1].set_cropbox(clip)
            records.append(
                PageRecord(len(out) - 1, src_idx, name, round(gutter, 4))
            )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # garbage=3 dedupes the spread image shared by both halves.
    out.save(out_pdf, garbage=3, deflate=True)
    out.close()
    src.close()

    if pagemap_path:
        pagemap_path.write_text(
            json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
        )
    return records


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Split 2-page spreads into book pages.")
    ap.add_argument("src", type=Path, help="source PDF of scanned spreads")
    ap.add_argument("out", type=Path, help="destination PDF of single pages")
    ap.add_argument(
        "--pagemap",
        type=Path,
        default=None,
        help="write the output->source page map here (default: <out>.pagemap.json)",
    )
    ap.add_argument(
        "--pages",
        default=None,
        help="source pages to process, e.g. '8,40-41' (default: all)",
    )
    args = ap.parse_args(argv)

    page_range = parse_range(args.pages) if args.pages else None
    pagemap = args.pagemap or args.out.with_suffix(".pagemap.json")

    records = split_document(args.src, args.out, pagemap, page_range)

    print(f"Wrote {len(records)} book pages to {args.out}")
    print(f"Page map: {pagemap}")
    gutters = [r.gutter_frac for r in records]
    if gutters:
        print(
            f"Gutter position: min {min(gutters):.3f} "
            f"max {max(gutters):.3f} mean {sum(gutters) / len(gutters):.3f}"
        )
    return 0


def parse_range(spec: str) -> list[int]:
    """Parse '8,40-41' into [8, 40, 41]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
