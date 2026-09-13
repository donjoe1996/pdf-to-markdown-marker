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


def check_text_layer_replaced(text: str, markers: list[str] | None = None) -> Finding:
    """Look for artefacts of a *known* bad text layer.

    The default list is specific to the Being and Time scan. For another
    document, pass markers observed in its own embedded layer, or rely on
    ``check_not_embedded_layer`` which needs no prior knowledge.
    """
    markers = STALE_LAYER_MARKERS if markers is None else markers
    hits = [m for m in markers if m.lower() in text.lower()]
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


def check_not_embedded_layer(text: str, pdf_path, sample: int = 8) -> Finding:
    """Generic version of the stale-layer check -- needs no prior knowledge.

    Compares the output against the PDF's *own* embedded text. If OCR really
    ran, the two differ in the places the old layer got wrong. Near-identical
    output means marker read the embedded layer instead of the page image.

    Only meaningful when fresh OCR was requested; for a born-digital extraction
    high similarity is the correct outcome, not a fault.
    """
    import difflib

    import pymupdf

    from bt.postprocess import split_pages

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    # Compare like with like. marker's paginated output labels each page with
    # its source index ("{7}----"), so the output page can be matched to the
    # very page it came from. Comparing whole-document blobs instead makes the
    # result depend on which pages happened to be sampled -- different pages
    # always look different, which would pass this check for the wrong reason.
    pages = [(n, body) for n, body in split_pages(text) if norm(body)]
    if not pages:
        return Finding("fresh OCR", True, "no paginated output to compare")

    step = max(1, len(pages) // sample)
    picked = pages[::step][:sample]

    ratios = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, body in picked:
            if page_no >= doc.page_count:
                continue
            embedded = norm(doc[page_no].get_text("text"))
            if len(embedded) < 200:
                continue
            ratios.append(
                difflib.SequenceMatcher(None, embedded, norm(body)[:20000]).ratio()
            )

    if not ratios:
        return Finding(
            "fresh OCR", True, "no meaningful embedded text to compare against"
        )

    avg = sum(ratios) / len(ratios)
    if avg > 0.90:
        return Finding(
            "fresh OCR",
            False,
            f"output is {avg:.0%} identical to the PDF's own text layer across "
            f"{len(ratios)} pages -- marker probably reused it instead of OCRing",
        )
    return Finding(
        "fresh OCR",
        True,
        f"output differs from the embedded layer ({avg:.0%} similar over "
        f"{len(ratios)} aligned pages)",
    )


# These three describe *content*, so their absence is only a fault when the
# document is known to contain them. Plenty of PDFs have no Greek, no italics
# and no footnotes; failing on that would make the report meaningless.
def check_greek(text: str, required: bool = False) -> Finding:
    greek = re.findall(r"[Ͱ-Ͽἀ-῿]+", text)
    if not greek:
        return Finding("greek", not required, "no Greek characters found")
    return Finding("greek", True, f"{len(greek)} Greek runs recognised")


def check_italics(text: str, required: bool = False) -> Finding:
    n = len(re.findall(r"(?<!\*)\*(?!\*)[^*\n]+\*(?!\*)", text))
    if n == 0:
        return Finding("italics", not required, "no italic spans found")
    return Finding("italics", True, f"{n} italic spans")


def check_footnotes(text: str, required: bool = False) -> Finding:
    n = len(re.findall(r"\[\^[^\]]+\]|<sup>", text))
    if n == 0:
        return Finding("footnotes", not required, "no footnote markers found")
    return Finding("footnotes", True, f"{n} footnote markers")


def check_images(text: str, base_dir) -> Finding:
    """Every figure the Markdown links to must exist on disk.

    The same chunk writes the link and the file, so a missing one means
    something dropped it between here and there. A dead image link is invisible
    in the Markdown source and glaring to a reader, which makes it worth a
    check rather than a hope. Relative links only: an http(s) figure is not
    ours to account for.
    """
    from bt.images import MD_IMAGE

    base = Path(base_dir)
    links = [
        target
        for _alt, target in MD_IMAGE.findall(text)
        if not target.startswith(("http://", "https://", "data:"))
    ]
    if not links:
        return Finding("figures", True, "no figures linked")

    missing = [t for t in dict.fromkeys(links) if not (base / t).exists()]
    if missing:
        shown = ", ".join(missing[:3])
        return Finding(
            "figures",
            False,
            f"{len(missing)} of {len(links)} linked figures are not on disk "
            f"(e.g. {shown}) -- the Markdown points at files that were never "
            "written, or the images directory was left behind when the book "
            "was copied",
        )
    return Finding("figures", True, f"{len(links)} figures linked, all present")


def run_all(
    text: str,
    pdf_path=None,
    markers: list[str] | None = None,
    expect: tuple[str, ...] = (),
    fresh_ocr: bool = True,
    base_dir=None,
) -> list[Finding]:
    """Check transcribed Markdown.

    ``pdf_path`` enables the generic embedded-layer comparison, which needs no
    prior knowledge of the document; without it only the hard-coded artefact
    list is available. ``expect`` names content that must be present
    ("greek", "italics", "footnotes") -- anything unnamed is reported but
    cannot fail. ``fresh_ocr=False`` skips the OCR checks entirely, since for a
    born-digital extraction matching the embedded layer is correct.
    ``base_dir`` is the directory the Markdown will be read from; giving it
    enables the check that extracted figures are actually there.
    """
    findings: list[Finding] = []
    if fresh_ocr:
        if pdf_path is not None:
            findings.append(check_not_embedded_layer(text, pdf_path))
        else:
            findings.append(check_text_layer_replaced(text, markers))
    findings.append(check_run_ons(text))
    findings.append(check_greek(text, "greek" in expect))
    findings.append(check_italics(text, "italics" in expect))
    findings.append(check_footnotes(text, "footnotes" in expect))
    if base_dir is not None:
        findings.append(check_images(text, base_dir))
    return findings


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check transcribed Markdown.")
    ap.add_argument("md", type=Path, help="Markdown file to check")
    args = ap.parse_args(argv)

    text = args.md.read_text(encoding="utf-8")
    print(f"Checking {args.md} ({len(text):,} chars)\n")
    findings = run_all(text, base_dir=args.md.parent)
    for f in findings:
        print(f"  [{'ok  ' if f.ok else 'FAIL'}] {f.name:<13} {f.detail}")

    failed = [f for f in findings if not f.ok]
    print("\nAll checks passed." if not failed else f"\n{len(failed)} check(s) failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
