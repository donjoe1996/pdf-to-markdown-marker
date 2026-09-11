"""Stage 3: clean marker's Markdown into a readable, citable text.

Four independent transforms, each individually switchable so they can be tuned
against real output rather than guessed at:

1. drop the running heads ("82  Being and Time  I. 2")
2. turn the marginal Niemeyer page numbers into inline ``[H. 56]`` anchors
3. namespace footnote markers per page (footnote "1" recurs on ~every page, so
   un-namespaced markers collide hundreds of times in one document)
4. rejoin words hyphenated across line and page breaks

Run with ``--report`` to see what each would change without writing anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# marker's paginated markdown marks each page as "{N}" followed by a rule,
# e.g. "{3}------------------------------------------------" -- not a bare rule,
# so a dashes-only pattern silently matches nothing and the whole book collapses
# into one "page" (which breaks per-page footnote namespacing).
PAGE_SEPARATOR = re.compile(r"^\{(\d+)\}-{10,}$", re.MULTILINE)

# Running heads look like "82  Being and Time  I. 2" (verso) and
# "INT. I  Being and Time  27" (recto). Rather than enumerate the section-label
# formats (INT. I, I. 2, II. 1, ...), which is brittle against OCR noise, a head
# is identified structurally: a short line carrying the title and a page number.
HEAD_TITLE = re.compile(r"being\s+and\s+time", re.IGNORECASE)
HEAD_MAX_LEN = 60


def is_running_head(line: str) -> bool:
    stripped = line.strip().strip("*_# ")
    if not stripped or len(stripped) > HEAD_MAX_LEN:
        return False
    if not HEAD_TITLE.search(stripped):
        return False
    # Every running head carries a page number; the title page ("BEING AND
    # TIME" alone) does not, and must survive.
    if not re.search(r"\d", stripped):
        return False
    # A citation such as "See Being and Time, p. 42." ends in sentence
    # punctuation; a running head never does.
    return stripped[-1] not in ".,;:"

# A marginal Niemeyer number sits alone on its own line once marker has pulled
# it out of the margin. Heidegger's German pagination runs 1..437.
MARGIN_NUMBER = re.compile(r"^\s*\**(\d{1,3})\**\s*$")

# Footnote markers, in the several shapes marker may emit them.
SUP_MARKER = re.compile(r"<sup>([0-9ivxIVX]{1,4})</sup>")
MD_FOOTNOTE_REF = re.compile(r"\[\^([0-9ivxIVX]{1,4})\]")

# Hyphen at end of line, continued on the next. Keeps genuine em/en dashes.
HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")


@dataclass
class Stats:
    pages: int = 0
    heads_removed: int = 0
    margins_converted: int = 0
    footnotes_namespaced: int = 0
    hyphens_joined: int = 0
    margin_sequence: list[int] = field(default_factory=list)

    def render(self) -> str:
        seq = self.margin_sequence
        gaps = [
            (a, b) for a, b in zip(seq, seq[1:]) if b != a + 1
        ]
        lines = [
            f"pages processed      : {self.pages}",
            f"running heads removed: {self.heads_removed}",
            f"[H. n] anchors made  : {self.margins_converted}",
            f"footnotes namespaced : {self.footnotes_namespaced}",
            f"hyphen joins         : {self.hyphens_joined}",
        ]
        if seq:
            lines.append(f"H-number range       : {seq[0]}..{seq[-1]}")
            if gaps:
                shown = ", ".join(f"{a}->{b}" for a, b in gaps[:10])
                lines.append(
                    f"H-number breaks      : {len(gaps)} "
                    f"(non-consecutive: {shown}{' ...' if len(gaps) > 10 else ''})"
                )
            else:
                lines.append("H-number breaks      : none (strictly consecutive)")
        return "\n".join(lines)


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split paginated markdown into ``(page_number, body)`` pairs.

    Uses marker's own page numbers rather than a running counter, so footnote
    ids stay stable when only part of the book is re-run.
    """
    marks = list(PAGE_SEPARATOR.finditer(text))
    if not marks:
        return [(0, text)]

    pages: list[tuple[int, str]] = []
    preamble = text[: marks[0].start()].strip()
    if preamble:
        pages.append((0, preamble))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages.append((int(m.group(1)), text[m.end() : end]))
    return pages


def strip_running_head(page: str, stats: Stats) -> str:
    out = []
    for line in page.split("\n"):
        if is_running_head(line):
            stats.heads_removed += 1
            continue
        out.append(line)
    return "\n".join(out)


def convert_margin_numbers(page: str, stats: Stats) -> str:
    """Replace standalone marginal numbers with an inline ``[H. n]`` anchor."""
    out = []
    for line in page.split("\n"):
        m = MARGIN_NUMBER.match(line)
        if m:
            n = int(m.group(1))
            # 1..437 is the Niemeyer range; anything else is not pagination.
            if 1 <= n <= 437:
                stats.margins_converted += 1
                stats.margin_sequence.append(n)
                out.append(f"[H. {n}]")
                continue
        out.append(line)
    return "\n".join(out)


def namespace_footnotes(page: str, page_no: int, stats: Stats) -> str:
    """Make footnote ids unique per page.

    The book restarts footnote numbering on every page and uses two series
    (arabic for the translators' notes, roman for Heidegger's marginalia), so
    a document-wide ``[^1]`` would be defined hundreds of times.
    """

    def sup(m: re.Match) -> str:
        stats.footnotes_namespaced += 1
        return f"[^p{page_no}-{m.group(1)}]"

    def ref(m: re.Match) -> str:
        stats.footnotes_namespaced += 1
        return f"[^p{page_no}-{m.group(1)}]"

    page = SUP_MARKER.sub(sup, page)
    page = MD_FOOTNOTE_REF.sub(ref, page)
    # A marker that opens a line is the note's definition, not a reference to
    # it; Markdown needs a colon there for the footnote to resolve.
    page = re.sub(r"(?m)^(\[\^p\d+-[0-9ivxIVX]{1,4}\])(?!:)\s+", r"\1: ", page)
    return page


def join_hyphens(text: str, stats: Stats) -> str:
    joined, n = HYPHEN_BREAK.subn(r"\1\2", text)
    stats.hyphens_joined += n
    return joined


def process(
    text: str,
    heads: bool = True,
    # Off by default: marker's MarginaliaProcessor discards the outer-margin
    # Niemeyer numbers before the markdown is written, so there is nothing left
    # for this to convert. Kept (and enablable with --margins) because it is
    # harmless and would catch any number that does survive.
    margins: bool = False,
    footnotes: bool = True,
    hyphens: bool = True,
) -> tuple[str, Stats]:
    stats = Stats()
    pages = split_pages(text)
    stats.pages = len(pages)

    done = []
    for page_no, page in pages:
        if heads:
            page = strip_running_head(page, stats)
        if margins:
            page = convert_margin_numbers(page, stats)
        if footnotes:
            page = namespace_footnotes(page, page_no, stats)
        done.append(page.strip())

    out = "\n\n".join(p for p in done if p)
    if hyphens:
        out = join_hyphens(out, stats)
    # Collapse the runs of blank lines left behind by removed lines.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out, stats


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Clean marker Markdown output.")
    ap.add_argument("src", type=Path, help="raw Markdown from stage 2")
    ap.add_argument("out", type=Path, nargs="?", help="cleaned output")
    ap.add_argument("--report", action="store_true", help="show stats, write nothing")
    ap.add_argument("--no-heads", action="store_true")
    ap.add_argument(
        "--margins",
        action="store_true",
        help="convert stray marginal numbers to [H. n] anchors; off by default "
        "because marker discards them before the markdown is written",
    )
    ap.add_argument("--no-footnotes", action="store_true")
    ap.add_argument("--no-hyphens", action="store_true")
    args = ap.parse_args(argv)

    text = args.src.read_text(encoding="utf-8")
    cleaned, stats = process(
        text,
        heads=not args.no_heads,
        margins=args.margins,
        footnotes=not args.no_footnotes,
        hyphens=not args.no_hyphens,
    )
    print(stats.render())

    if args.report:
        return 0
    if not args.out:
        ap.error("an output path is required unless --report is given")
    args.out.write_text(cleaned, encoding="utf-8")
    print(f"\nWrote {args.out} ({len(cleaned):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
