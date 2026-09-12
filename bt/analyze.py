"""Inspect a PDF and recommend pipeline settings.

The pipeline has two expensive, hard-to-undo choices: whether to split two-page
spreads, and whether to OCR at all. Both are cheap to get right by looking at
the file first, and costly to get wrong -- splitting a single-page PDF cuts
every page in half, and OCRing a born-digital PDF burns hours reproducing text
that could be extracted in seconds.

``analyze(path)`` answers both, plus a rough time estimate, so the GUI can
pre-fill settings instead of asking the user to guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from bt.split_spreads import detect_spreads
from bt.verify import RUN_ON

# Producers that indicate the text layer was made by OCR rather than typeset
# directly. Such a layer may be fine, but it is the case where it can silently
# be garbage -- this file's own layer came from "Paper Capture".
OCR_PRODUCERS = re.compile(
    r"paper\s*capture|abbyy|finereader|tesseract|scansoft|omnipage|readiris|ocr"
    r"|luradocument|luratech|djvu|scanned",
    re.IGNORECASE,
)

# The decisive signal. If a raster image covers essentially the whole page, the
# page *is* a photograph of paper -- so any text on it was produced by OCR, no
# matter how clean it looks or what the producer string claims. Producer names
# alone are too weak: a scan of Marcus Aurelius reports "Recoded by
# LuraDocument", has only 1225 chars/page and zero run-ons, and would otherwise
# be misread as typeset.
PAGE_IMAGE_COVERAGE = 0.9

# Below this many characters per page there is effectively no text layer.
MIN_CHARS_PER_PAGE = 120
# Run-on words per 1000 characters above which the layer looks damaged. Lost
# inter-word spacing is the signature failure of old OCR.
RUNON_PER_1K_BAD = 1.0

# Seconds per page, measured on this project: ~44 s/page on an M2 via llama.cpp
# when memory is not contended. Only an order-of-magnitude guide.
OCR_SECONDS_PER_PAGE = 45.0
EXTRACT_SECONDS_PER_PAGE = 0.2


@dataclass
class Analysis:
    path: Path
    pages: int = 0
    page_size: tuple[float, float] = (0.0, 0.0)
    # spreads
    is_spread: bool = False
    spread_reason: str = ""
    # text layer
    text_verdict: str = "no_text"  # born_digital | bad_ocr | no_text
    text_reason: str = ""
    chars_per_page: float = 0.0
    runons_per_1k: float = 0.0
    scanned_ratio: float = 0.0
    producer: str = ""
    # recommendations
    recommend_split: bool = False
    recommend_ocr: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def output_pages(self) -> int:
        """Pages after splitting -- what the OCR cost is actually based on."""
        return self.pages * 2 if self.recommend_split else self.pages

    @property
    def estimated_seconds(self) -> float:
        per = OCR_SECONDS_PER_PAGE if self.recommend_ocr else EXTRACT_SECONDS_PER_PAGE
        return self.output_pages * per

    def estimate_human(self) -> str:
        s = self.estimated_seconds
        if s < 90:
            return f"~{s:.0f} seconds"
        if s < 5400:
            return f"~{s / 60:.0f} minutes"
        return f"~{s / 3600:.1f} hours"

    def summary(self) -> str:
        lines = [
            f"file          : {self.path.name}",
            f"pages         : {self.pages} ({self.page_size[0]:.0f}x{self.page_size[1]:.0f} pt)",
            f"producer      : {self.producer or '(none)'}",
            f"spreads       : {self.is_spread} -- {self.spread_reason}",
            f"text layer    : {self.text_verdict} -- {self.text_reason}",
            f"  chars/page  : {self.chars_per_page:.0f}",
            f"  run-ons/1k  : {self.runons_per_1k:.2f}",
            f"  full-page images: {self.scanned_ratio:.0%} of sampled pages",
            "",
            f"recommend split : {self.recommend_split}",
            f"recommend OCR   : {self.recommend_ocr}",
            f"pages to process: {self.output_pages}",
            f"estimated time  : {self.estimate_human()}",
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def _page_image_coverage(page: pymupdf.Page) -> float:
    """Largest fraction of the page covered by a single raster image."""
    rect = page.rect
    area = rect.width * rect.height
    if area <= 0:
        return 0.0
    best = 0.0
    for img in page.get_images(full=True):
        for placed in page.get_image_rects(img[0]):
            best = max(best, (placed.width * placed.height) / area)
    return best


def _sample_scanned(doc: pymupdf.Document, sample: int = 8) -> float:
    """Fraction of sampled pages that are essentially a full-page image."""
    total = doc.page_count
    if total == 0:
        return 0.0
    step = max(1, total // sample)
    idxs = list(range(0, total, step))[:sample]
    hits = sum(
        1 for i in idxs if _page_image_coverage(doc[i]) >= PAGE_IMAGE_COVERAGE
    )
    return hits / len(idxs)


def _sample_text(doc: pymupdf.Document, sample: int = 12) -> tuple[float, float]:
    """Return (chars per page, run-on words per 1000 chars) over sampled pages."""
    total = doc.page_count
    if total == 0:
        return 0.0, 0.0
    step = max(1, total // sample)
    idxs = list(range(0, total, step))[:sample]

    chars = runons = 0
    for i in idxs:
        text = doc[i].get_text("text")
        chars += len(text)
        runons += len(RUN_ON.findall(text))
    per_page = chars / len(idxs)
    per_1k = (runons / chars * 1000) if chars else 0.0
    return per_page, per_1k


def analyze(path: str | Path, sample: int = 20) -> Analysis:
    """Inspect ``path`` and recommend settings."""
    path = Path(path)
    doc = pymupdf.open(path)
    try:
        a = Analysis(path=path, pages=doc.page_count)
        if doc.page_count:
            rect = doc[0].rect
            a.page_size = (rect.width, rect.height)
        a.producer = (doc.metadata or {}).get("producer", "") or ""

        spread = detect_spreads(doc, sample=sample)
        a.is_spread = bool(spread["is_spread"])
        a.spread_reason = spread["reason"]
        a.recommend_split = a.is_spread

        a.chars_per_page, a.runons_per_1k = _sample_text(doc)
        a.scanned_ratio = _sample_scanned(doc)
        ocr_producer = bool(OCR_PRODUCERS.search(a.producer))
        is_scan = a.scanned_ratio >= 0.6

        if a.chars_per_page < MIN_CHARS_PER_PAGE:
            a.text_verdict = "no_text"
            a.text_reason = (
                f"only {a.chars_per_page:.0f} chars/page -- scanned images, "
                "OCR is required"
            )
            a.recommend_ocr = True
        elif is_scan:
            # Text over a full-page image can only have come from OCR.
            a.text_verdict = "bad_ocr"
            a.text_reason = (
                f"{a.scanned_ratio:.0%} of sampled pages are a full-page image, "
                "so the text layer is someone else's OCR -- re-OCR and ignore it"
            )
            a.recommend_ocr = True
        elif a.runons_per_1k >= RUNON_PER_1K_BAD or ocr_producer:
            a.text_verdict = "bad_ocr"
            why = []
            if a.runons_per_1k >= RUNON_PER_1K_BAD:
                why.append(f"{a.runons_per_1k:.1f} run-on words per 1000 chars")
            if ocr_producer:
                why.append(f"producer looks like OCR software ({a.producer})")
            a.text_reason = (
                "text layer exists but looks unreliable: "
                + " and ".join(why)
                + " -- re-OCR and ignore it"
            )
            a.recommend_ocr = True
        else:
            a.text_verdict = "born_digital"
            a.text_reason = (
                f"{a.chars_per_page:.0f} chars/page, few run-ons -- the existing "
                "text looks typeset, so OCR would be wasted work"
            )
            a.recommend_ocr = False

        if a.recommend_split:
            a.notes.append(
                f"{a.pages} spreads will become ~{a.pages * 2} book pages"
            )
        if not a.recommend_ocr:
            a.notes.append(
                "extraction takes seconds; switch OCR on only if the result "
                "looks wrong"
            )
        return a
    finally:
        doc.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a PDF and recommend settings.")
    ap.add_argument("pdf", type=Path)
    args = ap.parse_args(argv)
    print(analyze(args.pdf).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
