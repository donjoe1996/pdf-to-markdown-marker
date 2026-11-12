"""Stage 3: clean marker's Markdown into a readable, citable text.

Independent transforms, each individually switchable so they can be tuned
against real output rather than guessed at:

1. normalise characters (ligatures, decomposed accents, invisible characters)
2. drop the running heads ("82  Being and Time  I. 2"), including the ones
   that change with the chapter
3. turn the marginal Niemeyer page numbers into inline ``[H. 56]`` anchors
4. namespace footnote markers per page (footnote "1" recurs on ~every page, so
   un-namespaced markers collide hundreds of times in one document)
5. rejoin words hyphenated across a line break -- deciding per word, from the
   document's own vocabulary, whether the hyphen belongs to the word
6. reflow each paragraph onto one line (marker keeps the PDF's own line breaks;
   on some born-digital PDFs that is a break after every word)
7. rejoin the sentence or word a page break cut in two

Every one of these repairs damage the *page layout* did to the text. None of
them can repair a character the OCR misread -- that is what ``bt.audit`` is for.

Run with ``--report`` to see what each would change without writing anything.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# marker's paginated markdown marks each page as "{N}" followed by a rule,
# e.g. "{3}------------------------------------------------" -- not a bare rule,
# so a dashes-only pattern silently matches nothing and the whole book collapses
# into one "page" (which breaks per-page footnote namespacing).
PAGE_SEPARATOR = re.compile(r"^\{(\d+)\}-{10,}$", re.MULTILINE)

# What post-processing writes in place of marker's separator. An HTML comment
# renders as nothing, so the anchor is invisible to a reader while staying
# machine-readable -- which keeps verify's page alignment working on the
# finished document, and lets any passage be traced back to a page.
#
# Deleting the markers entirely, as an earlier version did, silently disabled
# the most important check in verify.py: it reported "ok / no meaningful
# embedded text" across a whole 256-page book, a pass that proved nothing.
PAGE_ANCHOR_FMT = "<!-- page {n} -->"
PAGE_ANCHOR = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$", re.MULTILINE)

# Either form, so the same code reads marker's raw output and our own.
PAGE_MARK = re.compile(
    r"^(?:\{(\d+)\}-{10,}|<!--\s*page\s+(\d+)\s*-->)\s*$", re.MULTILINE
)

# Running heads are found by *repetition*, not by matching a known title. A head
# like "82  Being and Time  I. 2" varies only in its numbers from page to page,
# so normalising digits away makes it identical across the book -- whereas real
# body text essentially never recurs. This works on any document; matching a
# hard-coded title only ever worked on one.
HEAD_MAX_LEN = 60
HEAD_FREQUENCY = 0.30  # share of pages a line must appear on to count as a head
HEAD_SCAN_LINES = 2  # heads live at the very top or bottom of a page
HEAD_MIN_PAGES = 4  # below this, repetition proves nothing


def normalise_head(line: str) -> str:
    """Collapse a line to its page-invariant form (digits and case removed)."""
    s = line.strip().strip("*_# ")
    s = re.sub(r"\d+", "#", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _could_be_head(line: str) -> bool:
    """Shape test applied before frequency is even considered.

    Repetition alone is not enough. Body text can repeat too -- a templated
    sentence normalises to the same form on every page, and would otherwise be
    stripped as a head. A running head is a short *label*: it does not end in
    sentence punctuation and carries no footnote markup.
    """
    s = line.strip().strip("*_# ")
    if not s or len(s) > HEAD_MAX_LEN:
        return False
    if s[-1] in ".,;:!?":
        return False  # a sentence, not a label
    if "<sup>" in s or re.search(r"\[\^", s):
        return False  # footnote text, not a head
    if "](" in s:
        # An image or link, not a head. Extracted figures are named by page and
        # figure number, so `images/0000-0009_page_3_Figure_2.jpeg` normalises
        # to exactly the same form on every page -- a book with a chart on most
        # pages would have every figure deleted as a running head, leaving text
        # that still reads perfectly with the pictures silently gone.
        return False
    return True


def _edge_lines(body: str) -> list[str]:
    lines = [ln for ln in body.split("\n") if ln.strip()]
    if not lines:
        return []
    return lines[:HEAD_SCAN_LINES] + lines[-HEAD_SCAN_LINES:]


def find_running_heads(pages: list[tuple[int, str]]) -> set[str]:
    """Normalised forms that recur at page edges often enough to be heads."""
    if len(pages) < HEAD_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for _, body in pages:
        # Count each distinct form once per page, so a line repeated twice on
        # one page cannot manufacture a false majority.
        seen = {normalise_head(ln) for ln in _edge_lines(body) if _could_be_head(ln)}
        counts.update(seen)

    threshold = max(HEAD_MIN_PAGES, int(len(pages) * HEAD_FREQUENCY))
    return {form for form, n in counts.items() if form and n >= threshold}


def is_running_head(line: str, heads: set[str]) -> bool:
    return _could_be_head(line) and normalise_head(line) in heads


# The frequency rule above needs a line on 30% of the book's pages, and a head
# that changes with the chapter never gets there: ten chapters put each title
# on a tenth of the pages, and every one of them was kept. What makes a line a
# head is not its share of the whole book but that it repeats at the edge of
# *neighbouring* pages -- so that is what is tested. A gap of two allows the
# usual alternation (book title on the left page, chapter title on the right).
HEAD_LOCAL_RUN = 3  # pages in a row (allowing the gap) before it is a head
HEAD_LOCAL_GAP = 2
_STRUCTURE_START = re.compile(r"^(?:[-*+]\s|\d+[.)]\s|>|\||```|~~~|\$\$|<)")


def _could_be_local_head(line: str) -> bool:
    return _could_be_head(line) and not _STRUCTURE_START.match(line.strip())


def find_local_heads(pages: list[tuple[int, str]]) -> dict[int, set[str]]:
    """Heads proven by repetition on neighbouring pages.

    Returns ``{index into pages: normalised forms to strip there}``. Only the
    very first and last line of a page are considered -- local evidence is
    thinner than whole-book frequency, so the position test is stricter.

    A run usually begins on the chapter's own title page, where the same words
    are the *heading*; ``strip_running_head`` keeps that one.
    """
    seen: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, (_, body) in enumerate(pages):
        lines = [ln for ln in body.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue  # a page that is one line has no body for it to head
        for ln in dict.fromkeys((lines[0], lines[-1])):
            if _could_be_local_head(ln):
                seen[normalise_head(ln)].append((idx, ln))

    found: dict[int, set[str]] = defaultdict(set)
    for form, hits in seen.items():
        if not form:
            continue
        run: list[tuple[int, str]] = []
        for hit in [*hits, (10**9, "")]:  # sentinel flushes the last run
            if run and hit[0] - run[-1][0] > HEAD_LOCAL_GAP:
                if len(run) >= HEAD_LOCAL_RUN:
                    for idx, _ in run:
                        found[idx].add(form)
                run = []
            run.append(hit)
    return dict(found)

# A marginal Niemeyer number sits alone on its own line once marker has pulled
# it out of the margin. Heidegger's German pagination runs 1..437.
MARGIN_NUMBER = re.compile(r"^\s*\**(\d{1,3})\**\s*$")

# Footnote markers, in the several shapes marker may emit them.
SUP_MARKER = re.compile(r"<sup>([0-9ivxIVX]{1,4})</sup>")
MD_FOOTNOTE_REF = re.compile(r"\[\^([0-9ivxIVX]{1,4})\]")

# Hyphen at end of line, continued on the next. Keeps genuine em/en dashes.
# Both sides capture the whole hyphenated token ("Being-in" / "the-world"), not
# one letter: whether the hyphen stays is decided by what the word is.
_TOKEN = r"[^\W_]+(?:[-‐‑][^\W_]+)*"
HYPHEN_BREAK = re.compile(rf"({_TOKEN})([-‐‑])[ \t]*\n[ \t]*({_TOKEN})")
_VOCAB_TOKEN = re.compile(r"[^\W\d_]+(?:[-‐‑][^\W\d_]+)*")

# Characters a PDF hands back that are not the text. A text layer stores the
# font's glyphs, so "find" arrives as "ﬁnd" and no search matches it; accents
# arrive decomposed; and invisible characters split a word for every tool while
# rendering as nothing.
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_LIGATURE = re.compile("[" + "".join(LIGATURES) + "]")
# A soft hyphen is the typesetter marking a split word outright: at a line end
# it joins unconditionally, and anywhere else it is simply not text.
_SOFT_BREAK = re.compile("\u00ad[ \t]*\n[ \t]*")
_INVISIBLE = re.compile("[\u00ad\u200b\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# Reflow. marker emits a newline wherever pdftext/surya ended a line, and its
# markdownify keeps single newlines, so the Markdown carries the PDF's visual
# line breaks. Markdown *renders* a single newline as a space, which is why this
# looks fine in a viewer -- but as text it is wrong, and on some born-digital
# PDFs (pdfium breaks after every word) a page becomes a column of single words.
# Only plain prose is joined: a line Markdown reads as structure keeps its own.
_INLINE_TAGS = r"(?:sup|sub|i|b|em|strong|a|span|math)\b"
BLOCK_START = re.compile(
    r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||!\[|\[\^[^\]]+\]:|<(?!/?" + _INLINE_TAGS + r"))"
)
# Lines that nothing may be appended to: a heading, table row, figure or HTML
# block ends at its own newline.
BLOCK_WHOLE_LINE = re.compile(r"^(?:#{1,6}\s|\||!\[|```|~~~|\$\$|<(?!/?" + _INLINE_TAGS + r"))")
FENCE = re.compile(r"^(```|~~~|\$\$)")
# A footnote marker opening a line: in a notes paragraph it starts the next note.
NOTE_START = re.compile(r"^(?:<sup>[0-9ivxIVX]{1,4}</sup>|\[\^[0-9ivxIVX]{1,4}\])")
TRAILING_HYPHEN = re.compile(r"\w[-‐‑]$")


@dataclass
class Stats:
    pages: int = 0
    heads_removed: int = 0
    margins_converted: int = 0
    footnotes_namespaced: int = 0
    hyphens_joined: int = 0
    hyphens_kept: int = 0
    lines_joined: int = 0
    pages_bridged: int = 0
    chars_cleaned: int = 0
    margin_sequence: list[int] = field(default_factory=list)
    head_forms: list[str] = field(default_factory=list)

    def render(self) -> str:
        seq = self.margin_sequence
        # strict=False deliberately: pairing each item with its successor, so
        # the two sequences differ in length by one by construction.
        gaps = [(a, b) for a, b in zip(seq, seq[1:], strict=False) if b != a + 1]
        lines = [
            f"pages processed      : {self.pages}",
            f"running heads removed: {self.heads_removed}",
            f"[H. n] anchors made  : {self.margins_converted}",
            f"footnotes namespaced : {self.footnotes_namespaced}",
            f"characters cleaned   : {self.chars_cleaned}",
            f"hyphen joins         : {self.hyphens_joined}",
            f"compound hyphens kept: {self.hyphens_kept}",
            f"line breaks reflowed : {self.lines_joined}",
            f"page breaks bridged  : {self.pages_bridged}",
        ]
        if self.head_forms:
            shown = ", ".join(repr(f) for f in self.head_forms[:4])
            more = " ..." if len(self.head_forms) > 4 else ""
            lines.append(f"head patterns found  : {len(self.head_forms)} ({shown}{more})")
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
    marks = list(PAGE_MARK.finditer(text))
    if not marks:
        return [(0, text)]

    pages: list[tuple[int, str]] = []
    preamble = text[: marks[0].start()].strip()
    if preamble:
        pages.append((0, preamble))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        # Whichever alternative matched: marker's "{N}----" or our own anchor.
        number = m.group(1) or m.group(2)
        pages.append((int(number), text[m.end() : end]))
    return pages


def strip_running_head(
    page: str,
    heads: set[str],
    stats: Stats,
    local: set[str] | None = None,
    titles: set[str] | None = None,
) -> str:
    """Drop head lines at the page edges.

    ``heads`` are the whole-book forms, allowed in the first or last
    ``HEAD_SCAN_LINES``; ``local`` are this page's forms from
    ``find_local_heads``, allowed only as the very first or last line.

    ``titles`` collects the forms whose heading has been kept. A running head
    repeats a title, and the title itself is printed once as a heading -- the
    book's on its title page, a chapter's where the chapter opens. That line
    normalises to the same form as its heads, so the first occurrence Markdown
    marks as a heading is kept: losing a title is worse than leaving one head.
    """
    lines = page.split("\n")
    inked = [i for i, ln in enumerate(lines) if ln.strip()]
    # Only strip at the page edges: an identical string mid-paragraph is
    # body text, not a running head.
    edges = set(inked[:HEAD_SCAN_LINES] + inked[-HEAD_SCAN_LINES:])
    outer = {inked[0], inked[-1]} if inked else set()
    out = []
    for i, line in enumerate(lines):
        is_head = (i in edges and is_running_head(line, heads)) or bool(
            local
            and i in outer
            and _could_be_local_head(line)
            and normalise_head(line) in local
        )
        if is_head and titles is not None and line.lstrip().startswith("#"):
            form = normalise_head(line)
            is_head = form in titles
            titles.add(form)
        if is_head:
            stats.heads_removed += 1
            continue
        out.append(line)
    return "\n".join(out)


def clean_characters(text: str, stats: Stats) -> str:
    """Replace glyphs and invisible characters with the text they stand for."""
    changed = 0
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, n = _SOFT_BREAK.subn("", text)
    changed += n
    text, n = _INVISIBLE.subn("", text)
    changed += n
    text, n = _LIGATURE.subn(lambda m: LIGATURES[m.group(0)], text)
    changed += n
    changed += text.count("\u00a0")
    text = text.replace("\u00a0", " ")
    # NFC only: it composes "e" + combining acute into "é" and changes nothing
    # a reader could see. NFKC would also flatten superscripts and fractions,
    # which are content.
    composed = unicodedata.normalize("NFC", text)
    changed += len(text) - len(composed)
    stats.chars_cleaned += changed
    return composed


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


@dataclass
class Vocabulary:
    """How this document spells its words when no line break interferes.

    A hyphen at a line end is ambiguous on its own: "under-/standing" is a
    split word, "self-/evident" is a compound broken at its own hyphen. The
    document settles it -- whichever spelling it prints intact elsewhere is the
    spelling of the broken one. No dictionary could do this for a book whose
    terms ("Being-in-the-world") exist nowhere else.
    """

    words: Counter = field(default_factory=Counter)
    compounds: Counter = field(default_factory=Counter)

    @classmethod
    def of(cls, text: str) -> Vocabulary:
        vocab = cls()
        for token in _VOCAB_TOKEN.findall(text):
            token = re.sub("[‐‑]", "-", token.lower())
            (vocab.compounds if "-" in token else vocab.words)[token] += 1
        return vocab


def _keeps_hyphen(left: str, right: str, vocab: Vocabulary | None) -> bool:
    """Is the hyphen between ``left`` and ``right`` part of the word?"""
    if right[0].isupper() or right[0].isdigit() or left[-1].isdigit():
        return True  # "Anglo-/Saxon", "1914-/18": never a split word
    left, right = (re.sub("[‐‑]", "-", t.lower()) for t in (left, right))
    solid = left.rsplit("-", 1)[-1] + right.split("-", 1)[0]
    if vocab is not None:
        as_compound = vocab.compounds[f"{left}-{right}"]
        as_word = vocab.words[solid]
        if as_compound > as_word:
            return True
        if as_word:
            return False
    # No evidence either way. A token that is already hyphenated is a compound,
    # and compounds are broken at their hyphens; a plain word is a split word,
    # which is by far the commoner case in typeset prose.
    return "-" in left or "-" in right


def join_hyphens(text: str, stats: Stats, vocab: Vocabulary | None = None) -> str:
    """Rejoin words a line break split, keeping hyphens that are spelling."""

    def join(m: re.Match) -> str:
        left, hyphen, right = m.groups()
        if _keeps_hyphen(left, right, vocab):
            stats.hyphens_kept += 1
            return f"{left}{hyphen}{right}"
        stats.hyphens_joined += 1
        return f"{left}{right}"

    return HYPHEN_BREAK.sub(join, text)


def _accepts_continuation(line: str) -> bool:
    """Can the next prose line be appended to ``line``?"""
    if line.endswith(("  ", "\\")):
        return False  # an explicit Markdown hard break
    s = line.strip()
    if BLOCK_WHOLE_LINE.match(s):
        return False
    # A hyphen left at a line end is either a word split dehyphenation did not
    # join (``hyphens=False``) or a deliberate one; a space would wreck both.
    return not TRAILING_HYPHEN.search(s)


def reflow_lines(page: str, stats: Stats) -> str:
    """Join the soft line breaks inside each paragraph of one page.

    Blank lines (paragraph breaks) are kept, fenced code and block math are
    left verbatim, and a line that opens Markdown structure (heading, list item,
    table row, quote, figure, HTML block, footnote definition) keeps its own
    line. Inside a paragraph of footnotes -- one that *opens* with a marker --
    each marker starts a new line, so every note stays its own definition;
    anywhere else a marker opening a line is an inline reference that the PDF
    happened to wrap before, and is joined back into its sentence.
    """
    out: list[str] = []
    fence: str | None = None
    at_para_start = True
    notes_para = False
    for line in page.split("\n"):
        s = line.strip()
        if fence is not None:
            out.append(line)
            if s.startswith(fence) and (fence != "$$" or s == "$$"):
                fence = None
            continue
        m = FENCE.match(s)
        if m and (m.group(1) != "$$" or s == "$$"):
            fence = m.group(1)
            out.append(line)
            at_para_start = False
            continue
        if not s:
            out.append(line)
            at_para_start = True
            continue
        if at_para_start:
            notes_para = bool(NOTE_START.match(s))
            out.append(line)
            at_para_start = False
            continue
        starts_block = BLOCK_START.match(s) or (notes_para and NOTE_START.match(s))
        if not starts_block and _accepts_continuation(out[-1]):
            out[-1] = f"{out[-1].rstrip()} {s}"
            stats.lines_joined += 1
        else:
            out.append(line)
    return "\n".join(out)


# Trailing matter that can follow the end of a sentence without being it:
# footnote references, closing quotes and brackets, emphasis markers.
_SENTENCE_TAIL = re.compile(
    r"(?:\[\^[^\]]+\]|<sup>[^<]*</sup>|[\s*_\"'”’»)\]])+$"
)
NOTE_DEFINITION = re.compile(r"^(?:\[\^[^\]]+\]:|<sup>[0-9ivxIVX]{1,4}</sup>)")


def _is_prose(para: str) -> bool:
    lines = para.split("\n")
    return not (BLOCK_START.match(lines[0].strip()) or BLOCK_START.match(lines[-1].strip()))


def _bridge(
    before: str, after: str, stats: Stats, vocab: Vocabulary | None, hyphens: bool
) -> tuple[str, str] | None:
    """Move the continuation that opens ``after`` back onto ``before``.

    Returns the two new page bodies, or None when the text does not clearly
    run on. Both halves must agree -- no sentence end before the break and a
    lower-case letter after it -- because a wrong join fuses two paragraphs,
    which is worse than the split it was meant to repair.
    """
    paras = re.split(r"\n{2,}", before)
    # The page's notes sit under its text, so the paragraph the break cut is
    # the last one that is not a note.
    at = next(
        (i for i in range(len(paras) - 1, -1, -1) if not NOTE_DEFINITION.match(paras[i])),
        None,
    )
    if at is None:
        return None
    tail = paras[at].rstrip()
    rest = re.split(r"\n{2,}", after)
    head = rest[0]
    opening = head.lstrip("*_")
    if not opening or not opening[0].islower():
        return None
    if not _is_prose(tail) or not _is_prose(head) or NOTE_DEFINITION.match(head):
        return None
    if not _accepts_continuation(tail.split("\n")[-1]) and not TRAILING_HYPHEN.search(tail):
        return None

    if TRAILING_HYPHEN.search(tail):
        if not hyphens:
            return None
        merged = join_hyphens(f"{tail}\n{head}", stats, vocab)
        if merged == f"{tail}\n{head}":
            merged = tail + head
    else:
        ending = _SENTENCE_TAIL.sub("", tail)
        if not ending or ending[-1] in ".!?…:":
            return None
        merged = f"{tail} {head}"

    stats.pages_bridged += 1
    paras[at] = merged
    return "\n\n".join(paras), "\n\n".join(rest[1:])


def bridge_pages(
    pages: list[tuple[int, str]],
    stats: Stats,
    vocab: Vocabulary | None = None,
    hyphens: bool = True,
) -> list[tuple[int, str]]:
    """Rejoin the sentence or word each page break cut in two.

    The continuation moves *back* to the page its paragraph started on, so the
    anchor still stands alone on its line and every page keeps its number. A
    few words are thereby attributed to the previous page; that is the usual
    convention for a reflowed text and it leaves ``verify``'s page alignment
    intact, which an anchor placed mid-paragraph would not.
    """
    out = list(pages)
    last = None  # most recent page that still has a body
    for i, (number, body) in enumerate(out):
        if not body:
            continue
        if last is not None:
            joined = _bridge(out[last][1], body, stats, vocab, hyphens)
            if joined:
                out[last] = (out[last][0], joined[0])
                out[i] = (number, joined[1])
        if out[i][1]:
            last = i
    return out


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
    reflow: bool = True,
    bridge: bool = True,
    clean: bool = True,
) -> tuple[str, Stats]:
    stats = Stats()
    if clean:
        text = clean_characters(text, stats)
    pages = split_pages(text)
    stats.pages = len(pages)

    # First pass: learn which lines recur at page edges across the whole
    # document. Head removal needs the corpus, so it cannot be decided per page.
    head_forms = find_running_heads(pages) if heads else set()
    local_heads = find_local_heads(pages) if heads else {}
    stats.head_forms = sorted(head_forms | {f for forms in local_heads.values() for f in forms})
    # Likewise the vocabulary: how a broken word is spelled is answered by the
    # places the document prints it unbroken.
    vocab = Vocabulary.of(text) if hyphens else None

    done = []
    titles: set[str] = set()
    for idx, (page_no, page) in enumerate(pages):
        if heads:
            page = strip_running_head(
                page, head_forms, stats, local_heads.get(idx), titles
            )
        if margins:
            page = convert_margin_numbers(page, stats)
        # Per page; the break between two pages is bridge_pages' job. It must
        # precede reflow, which would otherwise put a space inside the split
        # word.
        if hyphens:
            page = join_hyphens(page, stats, vocab)
        # Before footnotes: namespacing reads a marker that opens a line as a
        # definition, so a reference the PDF wrapped onto a new line would
        # otherwise become a bogus note.
        if reflow:
            page = reflow_lines(page, stats)
        if footnotes:
            page = namespace_footnotes(page, page_no, stats)
        done.append((page_no, page.strip()))

    # After the per-page work: heads are gone, so a page's first line is really
    # its text, and notes are namespaced, so they can be told from the body.
    if bridge:
        done = bridge_pages(done, stats, vocab, hyphens)

    # Re-emit the page number as an invisible anchor. Dropping it entirely is
    # what broke verification on the finished document: without these, page
    # alignment has nothing to align by.
    out = "\n\n".join(
        f"{PAGE_ANCHOR_FMT.format(n=page_no)}\n\n{body}"
        for page_no, body in done
        if body
    )
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
    ap.add_argument(
        "--no-bridge",
        action="store_true",
        help="leave a sentence cut by a page break as two fragments",
    )
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="keep ligature glyphs, decomposed accents and invisible characters",
    )
    ap.add_argument(
        "--no-reflow",
        action="store_true",
        help="keep the PDF's line breaks instead of one line per paragraph",
    )
    args = ap.parse_args(argv)

    text = args.src.read_text(encoding="utf-8")
    cleaned, stats = process(
        text,
        heads=not args.no_heads,
        margins=args.margins,
        footnotes=not args.no_footnotes,
        hyphens=not args.no_hyphens,
        reflow=not args.no_reflow,
        bridge=not args.no_bridge,
        clean=not args.no_clean,
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
