"""Quality checks on transcribed output.

The headline check is ``check_text_layer_replaced``. On MPS marker defaults to
``mode="fast"``, which extracts text from the PDF's existing text layer instead
of OCRing. This PDF's embedded layer is poor Acrobat "Paper Capture" output, so
if that path is ever taken the result is silently bad -- readable Markdown made
of the wrong characters. The embedded layer has distinctive damage, which makes
it easy to detect in the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Verbatim damage observed in the embedded Acrobat OCR layer of this file.
# None of these are real English or German words, so any hit means marker
# reused the old text layer rather than re-OCRing.
STALE_LAYER_MARKERS = [
    "itse1f",  # digit 1 substituted for letter l
    "sorne",  # rn read as m, in reverse
    "certainwayof",  # collapsed inter-word spacing
    "whichisDasein",
    "cornpletely",
    "sirnply",
]

# Long runs of letters with no space: the other signature of the stale layer.
RUN_ON = re.compile(r"[A-Za-z']{28,}")


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str


def check_text_layer_replaced(text: str) -> Finding:
    hits = [m for m in STALE_LAYER_MARKERS if m.lower() in text.lower()]
    if hits:
        return Finding(
            "fresh OCR",
            False,
            f"found stale-layer artefacts {hits} -- marker reused the embedded "
            "Acrobat text layer. Confirm force_ocr/strip_existing_ocr are set, "
            "or re-run the split with --rasterize.",
        )
    return Finding("fresh OCR", True, "no stale-layer artefacts found")


def check_run_ons(text: str) -> Finding:
    hits = RUN_ON.findall(text)
    if len(hits) > 5:
        sample = ", ".join(hits[:3])
        return Finding(
            "word spacing", False, f"{len(hits)} run-on words (e.g. {sample})"
        )
    return Finding("word spacing", True, f"{len(hits)} run-on words")


def check_greek(text: str) -> Finding:
    """The opening page and many footnotes carry polytonic Greek."""
    greek = re.findall(r"[Ͱ-Ͽἀ-῿]+", text)
    if not greek:
        return Finding(
            "greek", False, "no Greek characters found -- expected on book page 1"
        )
    return Finding("greek", True, f"{len(greek)} Greek runs recognised")


def check_italics(text: str) -> Finding:
    n = len(re.findall(r"(?<!\*)\*(?!\*)[^*\n]+\*(?!\*)", text))
    if n == 0:
        return Finding("italics", False, "no italic spans -- emphasis was lost")
    return Finding("italics", True, f"{n} italic spans")


def check_footnotes(text: str) -> Finding:
    n = len(re.findall(r"\[\^[^\]]+\]|<sup>", text))
    if n == 0:
        return Finding("footnotes", False, "no footnote markers found")
    return Finding("footnotes", True, f"{n} footnote markers")


def run_all(text: str) -> list[Finding]:
    return [
        check_text_layer_replaced(text),
        check_run_ons(text),
        check_greek(text),
        check_italics(text),
        check_footnotes(text),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check transcribed Markdown.")
    ap.add_argument("md", type=Path, help="Markdown file to check")
    args = ap.parse_args(argv)

    text = args.md.read_text(encoding="utf-8")
    print(f"Checking {args.md} ({len(text):,} chars)\n")
    findings = run_all(text)
    for f in findings:
        print(f"  [{'ok  ' if f.ok else 'FAIL'}] {f.name:<13} {f.detail}")

    failed = [f for f in findings if not f.ok]
    print("\nAll checks passed." if not failed else f"\n{len(failed)} check(s) failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
