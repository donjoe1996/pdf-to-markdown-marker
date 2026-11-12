"""Find the pages where the transcription lost or invented text.

``verify.py`` asks questions about the *document* (was fresh OCR run? are there
run-on words?). Those pass on a book in which one page in fifty is missing half
its text, because the failure is local and the checks are global. This module
asks the same kind of question page by page, so that a bad page has a number.

Why pages go wrong silently -- read from marker/surya's source, not guessed:

* A page whose full-page output loops or fails to parse is redone block by
  block; a block that loops again is **dropped** with a log line
  (``OcrBuilder.clean_html``). The Markdown simply has less text on that page.
* Output that reaches the token budget is accepted as it stands -- a truncated
  page reads exactly like a page that ended there.
* With ``--no-ocr`` a page that is a scanned image has no text layer, so it
  comes out empty, in a book that otherwise extracted perfectly.

None of these needs a model to detect. The PDF is its own witness:

1. **Its text layer.** Even bad OCR gets the *amount* of text on a page roughly
   right, so the count of letters and digits is a reliable yardstick where the
   characters themselves are not. Output far shorter than the layer lost text;
   far longer invented or repeated it.
2. **Its ink.** Where there is no layer, a page that is visibly not blank and
   came out empty is worth a look.

Run it on ``raw.md`` (or the final Markdown) against the PDF that was actually
OCRed -- ``pages.pdf`` for a split book, the source otherwise:

    uv run python -m bt.audit output/NAME/raw.md --pdf output/NAME/pages.pdf
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from bt.postprocess import PAGE_MARK, split_pages

# Below this many letters/digits a text layer is a caption or a page number,
# and a ratio against it says nothing. Same floor verify uses.
MIN_WITNESS = 200
# Output/witness length. The spread on healthy pages is narrow (markup is
# stripped, running heads are a few dozen characters), so these are wide enough
# that a hit means a block went missing, not that a head was removed.
SHORT_RATIO = 0.6
LONG_RATIO = 1.6

# Mean darkness (0..1) above which a page is not blank. A page of body text
# measures around 0.05-0.10; scanner specks and a binding shadow, well under
# 0.01.
INK_PRESENT = 0.01
INK_DPI = 36
EMPTY_CHARS = 20
# With no text layer, the only yardstick is the document itself: characters
# per unit of ink is roughly constant across body pages.
SPARSE_SHARE = 0.25
SPARSE_MIN_PAGES = 8

# A unit repeated this often, making up this much of a page, is a decoder loop.
LOOP_REPEATS = 4
LOOP_SHARE = 0.3
LOOP_MIN_CHARS = 12

_MARKUP = re.compile(
    r"<!--.*?-->"  # page anchors
    r"|!?\[[^\]]*\]\([^)]*\)"  # image and link targets
    r"|\[\^[^\]]+\]:?"  # footnote ids
    r"|<[^>]+>",  # html tags
    re.DOTALL,
)
_UNIT_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class PageIssue:
    page: int
    kind: str  # missing | extra | repeated | empty | sparse
    detail: str
    # Strong evidence fails verification. Weak evidence is reported only: a
    # page of ink and no text is usually a plate, and failing every illustrated
    # book would teach people to ignore the check.
    strong: bool

    def render(self) -> str:
        return f"page {self.page}: {self.detail}"


def text_amount(text: str) -> int:
    """Letters and digits that are content, with markup taken out."""
    return sum(ch.isalnum() for ch in _MARKUP.sub(" ", text))


def find_repetition(body: str) -> tuple[str, int] | None:
    """The sentence or line a page is stuck repeating, if it is.

    marker catches a loop that runs to the end of the output. One that recovers
    mid-page -- the same line a dozen times, then on with the text -- is still
    in the Markdown.
    """
    units = [re.sub(r"\s+", " ", u).strip().lower() for u in _UNIT_BREAK.split(body)]
    units = [u for u in units if sum(ch.isalnum() for ch in u) >= LOOP_MIN_CHARS]
    if not units:
        return None
    unit, n = Counter(units).most_common(1)[0]
    total = sum(len(u) for u in units)
    if n >= LOOP_REPEATS and n * len(unit) >= LOOP_SHARE * total:
        return unit, n
    return None


def page_ink(page) -> float:
    """Mean darkness of a page, 0 (blank) to 1 (solid black).

    Summed darkness, not a count of dark pixels: at this resolution a bitonal
    scan's thin strokes average to light grey and a threshold sees no ink at
    all -- the same trap as gutter detection.
    """
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(dpi=INK_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    grey = np.frombuffer(pix.samples, dtype=np.uint8)
    return float((255 - grey.astype(np.float64)).mean() / 255) if grey.size else 0.0


def audit_pages(text: str, pdf_path=None) -> list[PageIssue]:
    """Issues found, in page order. Empty means nothing looks wrong.

    Without ``pdf_path`` only repetition can be checked -- there is nothing to
    measure the page against.
    """
    pages = split_pages(text)
    issues: list[PageIssue] = []

    for number, body in pages:
        loop = find_repetition(body)
        if loop:
            unit, n = loop
            issues.append(
                PageIssue(
                    number,
                    "repeated",
                    f"the same text appears {n} times ({unit[:40]!r}) -- the "
                    "OCR model looped",
                    strong=True,
                )
            )

    # Unpaginated text is one "page 0" holding the whole document; measuring
    # that against the PDF's first page would flag every book.
    if pdf_path is None or not PAGE_MARK.search(text):
        return sorted(issues, key=lambda i: i.page)

    import pymupdf

    unwitnessed: list[tuple[int, int, float]] = []  # page, amount, ink
    with pymupdf.open(pdf_path) as doc:
        for number, body in pages:
            if number >= doc.page_count:
                continue
            page = doc[number]
            got = text_amount(body)
            # Clipped to the visible page: a split book's pages are crop boxes
            # over a spread, and the other half's text is still in the file.
            expected = text_amount(page.get_text("text", clip=page.rect))

            if expected >= MIN_WITNESS:
                ratio = got / expected
                if ratio < SHORT_RATIO:
                    issues.append(
                        PageIssue(
                            number,
                            "missing",
                            f"{got} characters where the PDF's text layer has "
                            f"{expected} ({ratio:.0%}) -- text was dropped",
                            strong=True,
                        )
                    )
                elif ratio > LONG_RATIO:
                    issues.append(
                        PageIssue(
                            number,
                            "extra",
                            f"{got} characters where the PDF's text layer has "
                            f"{expected} ({ratio:.0%}) -- text was repeated or "
                            "invented",
                            strong=True,
                        )
                    )
                continue

            ink = page_ink(page)
            if ink < INK_PRESENT:
                continue  # a blank page that came out blank
            if got < EMPTY_CHARS:
                issues.append(
                    PageIssue(
                        number,
                        "empty",
                        "no text, but the page is not blank -- a full-page "
                        "figure, or a page the run could not read",
                        strong=False,
                    )
                )
            else:
                unwitnessed.append((number, got, ink))

    if len(unwitnessed) >= SPARSE_MIN_PAGES:
        typical = median(got / ink for _, got, ink in unwitnessed)
        for number, got, ink in unwitnessed:
            if got / ink < SPARSE_SHARE * typical:
                issues.append(
                    PageIssue(
                        number,
                        "sparse",
                        f"{got} characters, about {got / ink / typical:.0%} of "
                        "what this much ink carries elsewhere in the document "
                        "-- a figure, or dropped text",
                        strong=False,
                    )
                )

    return sorted(issues, key=lambda i: i.page)


def summarise(issues: list[PageIssue], limit: int = 6) -> str:
    """One line for a report: counts first, then the first few pages."""
    if not issues:
        return "no page looks incomplete or repeated"
    strong = [i for i in issues if i.strong]
    weak = [i for i in issues if not i.strong]
    parts = []
    if strong:
        parts.append(f"{len(strong)} page(s) lost or repeated text")
    if weak:
        parts.append(f"{len(weak)} page(s) worth a look")
    shown = "; ".join(i.render() for i in (strong + weak)[:limit])
    more = f"; and {len(issues) - limit} more" if len(issues) > limit else ""
    return f"{', '.join(parts)} -- {shown}{more}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="List the pages where the transcription lost or repeated text."
    )
    ap.add_argument("md", type=Path, help="raw.md or the final Markdown")
    ap.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="the PDF that was OCRed (default: pages.pdf beside the Markdown)",
    )
    args = ap.parse_args(argv)

    pdf = args.pdf
    if pdf is None and (args.md.parent / "pages.pdf").exists():
        pdf = args.md.parent / "pages.pdf"
    if pdf is None:
        print("note: no --pdf given; only repetition can be checked\n")

    issues = audit_pages(args.md.read_text(encoding="utf-8"), pdf)
    for issue in issues:
        print(f"  [{'FAIL' if issue.strong else 'look'}] {issue.render()}")
    strong = sum(i.strong for i in issues)
    print(
        f"\n{strong} page(s) lost or repeated text, "
        f"{len(issues) - strong} worth a look."
        if issues
        else "\nNo page looks incomplete or repeated."
    )
    return 1 if any(i.strong for i in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
