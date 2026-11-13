"""Optional stage 5: one Markdown file per chapter, for a RAG vault.

``uv run python -m bt.chapters output/NAME/NAME.english.md``

The finished Markdown is one file per book. A retrieval tool that caps how
many passages one note may contribute -- Ask takes the 6 best passages, at most
3 from any note -- can then use 3 passages of a whole book, however relevant
the rest of it is. One note per chapter lifts that to all 6, from different
chapters. That cap is the reason for this stage; the rest follows from it:

**Chapters, not sections.** The retriever already cuts each note into passages
of about 1,200 characters under each heading, and every passage carries its
heading path. Sections as files would only multiply files. So the split level
is the *deepest* heading level whose pieces are still bigger than a section
(``choose_level``): ``##`` chapters under a single ``#`` book title, chapters
rather than the parts above them, chapters rather than the sections below.

**Page anchors are kept, never renumbered.** Each passage's page comes from the
``<!-- page N -->`` above it, so a chapter that opens mid-page starts with that
page's anchor again, and an anchor left at the very end of a chapter moves to
the next one, where its page's text is. Restarting at 1 per file would point
every citation at the wrong page.

**Nothing is lost or duplicated at a cut**, apart from those re-emitted
anchors. A footnote definition follows its reference into the other file when
a cut falls between them, because a footnote resolves only within its note.

**Books only.** A journal article is already note-sized; split, one argument
is scattered over files that each read as a fragment. Length is the signal --
articles run ~5-40 pages, books 100 or more -- so a document under
``BOOK_MIN_PAGES`` is left as the one note it is and nothing is written.

**Names are unique across books**: ``<Prefix> NN <Title>.md``. Links resolve
by file name across the whole vault, so ``Chapter 1.md`` in two books would
collide.

Runs on any finished Markdown, translated or not. ``bt.translate`` calls it when
a translation completes -- and only then: a part-done book split into chapters
looks finished, its last chapter simply ending early.
"""

from __future__ import annotations

import math
import re
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from bt.postprocess import HEADING, PAGE_ANCHOR, PAGE_ANCHOR_FMT

# A piece whose median is smaller than this is a section, not a chapter. Body
# text is ~2,000 characters a page, so ~10 pages: sections run a few pages,
# chapters a few dozen.
SECTION_CHARS = 20_000
# Above this a chapter is cut again, at page anchors. A book with no usable
# headings is otherwise one note again, and the vault opens a note above ~600k
# characters in plain view rather than the editor.
MAX_CHARS = 200_000
# Less text than this is a title page ("Part Two") or a stray label: it joins
# the chapter after it rather than becoming a note that can only answer its name.
MIN_CHARS = 500
# Shorter than this is an article, not a book, and is left as one note. Journal
# articles run ~5-40 pages and books 100+; 60 sits in the gap between them.
BOOK_MIN_PAGES = 60
# Body text per page, for a document without page anchors to count.
CHARS_PER_PAGE = 2_000
# Headings deeper than this are never chapters.
MAX_LEVEL = 3

TITLE_MAX = 80
FRONT_MATTER = "Front matter"
CONTENTS = "Contents"

FENCE = re.compile(r"^\s*(```|~~~)")
FOOTNOTE_DEF = re.compile(r"^\[\^([^\]\s]+)\]:")
FOOTNOTE_REF = re.compile(r"\[\^([^\]\s]+)\](?!:)")
IMAGE_TARGET = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")

# For titles: markup to drop whole, then markup to unwrap, then characters that
# break a file name (/ \ : * ? " < > |) or an Obsidian link (# ^ [ ] |).
_TITLE_DROP = re.compile(r"<sup>.*?</sup>|<sub>.*?</sub>|\[\^[^\]]*\]")
_TITLE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_TITLE_TAG = re.compile(r"<[^>]+>")
_TITLE_UNSAFE = re.compile(r'[\\/:*?"<>|#^\[\]`*_~]')


# --------------------------------------------------------------------------
# reading the document
# --------------------------------------------------------------------------
@dataclass
class Line:
    text: str
    page: int | None  # the page this line is on: the last anchor at or above it
    level: int = 0  # heading level, 0 if not a heading (or inside a fence)
    anchor: bool = False


def read_lines(text: str) -> list[Line]:
    lines: list[Line] = []
    page: int | None = None
    fenced = False
    for raw in text.splitlines():
        if FENCE.match(raw):
            fenced = not fenced
        m = None if fenced else PAGE_ANCHOR.match(raw)
        if m:
            page = int(m.group(1))
            lines.append(Line(raw, page, anchor=True))
            continue
        h = None if fenced else HEADING.match(raw)
        lines.append(Line(raw, page, level=len(h.group(1)) if h else 0))
    return lines


def text_chars(lines: list[Line]) -> int:
    """What a piece holds, excluding page anchors and blank lines."""
    return sum(len(ln.text.strip()) for ln in lines if not ln.anchor)


def _cut(lines: list[Line], level: int) -> list[list[Line]]:
    """Cut before every heading of ``level`` or shallower."""
    pieces: list[list[Line]] = [[]]
    for ln in lines:
        if 0 < ln.level <= level and any(not x.anchor and x.text.strip() for x in pieces[-1]):
            # An anchor directly above the heading is on the heading's page:
            # it goes with the heading, not at the end of the previous piece.
            tail: list[Line] = []
            while pieces[-1] and (pieces[-1][-1].anchor or not pieces[-1][-1].text.strip()):
                tail.insert(0, pieces[-1].pop())
            pieces.append(tail)
        pieces[-1].append(ln)
    return [p for p in pieces if p]


def _merge_small(pieces: list[list[Line]], min_chars: int) -> list[list[Line]]:
    """A part title page joins the chapter after it; a stray end, the one before."""
    merged: list[list[Line]] = []
    carry: list[Line] = []
    for piece in pieces:
        piece = carry + piece
        carry = []
        if text_chars(piece) < min_chars:
            carry = piece
        else:
            merged.append(piece)
    if carry:
        if merged:
            merged[-1].extend(carry)
        else:
            merged.append(carry)
    return merged


def levels(lines: list[Line]) -> list[int]:
    return sorted({ln.level for ln in lines if 0 < ln.level <= MAX_LEVEL})


def choose_level(
    text: str | list[Line],
    section_chars: int = SECTION_CHARS,
    min_chars: int = MIN_CHARS,
    **_: object,
) -> int | None:
    """The heading level that marks chapters, or None if no level splits the book.

    The deepest level whose median piece is still bigger than a section. Taking
    ``#`` regardless would give one file for a book whose only ``#`` is its
    title -- the cap this stage exists to escape -- and parts rather than
    chapters for a book in parts. Where every level is section-sized (a short
    paper) the shallowest that splits at all is used.
    """
    lines = read_lines(text) if isinstance(text, str) else text
    splitting = []
    for lvl in levels(lines):
        pieces = _merge_small(_cut(lines, lvl), min_chars)
        if len(pieces) >= 2:
            splitting.append((lvl, statistics.median(text_chars(p) for p in pieces)))
    if not splitting:
        return None
    chapter_sized = [lvl for lvl, median in splitting if median >= section_chars]
    return max(chapter_sized) if chapter_sized else splitting[0][0]


def _split_oversized(piece: list[Line], max_chars: int) -> list[list[Line]]:
    """Cut a piece over ``max_chars`` into near-equal runs of whole pages."""
    size = text_chars(piece)
    if size <= max_chars:
        return [piece]
    target = size / math.ceil(size / max_chars)
    parts: list[list[Line]] = [[]]
    for ln in piece:
        if ln.anchor and text_chars(parts[-1]) >= target:
            parts.append([])
        parts[-1].append(ln)
    return parts


# --------------------------------------------------------------------------
# the chapters
# --------------------------------------------------------------------------
@dataclass
class Chapter:
    title: str
    body: str
    first_page: int | None
    last_page: int | None

    @property
    def pages(self) -> str:
        if self.first_page is None:
            return ""
        if self.first_page == self.last_page:
            return f"p. {self.first_page}"
        return f"pp. {self.first_page}–{self.last_page}"


def clean_title(text: str) -> str:
    """A heading as a file name that is also a valid ``[[link]]`` target."""
    t = HEADING.match(text).group(2) if HEADING.match(text) else text
    t = _TITLE_DROP.sub("", t)
    t = _TITLE_LINK.sub(r"\1", t)
    t = _TITLE_TAG.sub("", t)
    t = _TITLE_UNSAFE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > TITLE_MAX:
        t = t[: TITLE_MAX + 1].rsplit(" ", 1)[0]
    # Windows will not keep a trailing dot or space on a file name.
    return t.rstrip(" .")


def _title(piece: list[Line], level: int | None) -> str:
    headings = [ln for ln in piece if ln.level]
    # The heading at the chapter level, not a part title merged in above it.
    pick = next((ln for ln in headings if ln.level == level), None) if level else None
    return clean_title(pick.text) if pick else FRONT_MATTER


def _render(piece: list[Line], anchored: bool) -> str:
    lines = list(piece)
    while lines and (lines[-1].anchor or not lines[-1].text.strip()):
        lines.pop()
    first = next((ln for ln in lines if ln.text.strip()), None)
    out = [ln.text for ln in lines]
    if anchored and first is not None and not first.anchor and first.page is not None:
        # Opens mid-page: say which page, or its first passage has none.
        out = [PAGE_ANCHOR_FMT.format(n=first.page), ""] + out
    return "\n".join(out).strip() + "\n"


def _footnote_blocks(body: str) -> dict[str, str]:
    """Definition id -> its full text (the line plus indented continuation)."""
    blocks: dict[str, str] = {}
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        m = FOOTNOTE_DEF.match(ln)
        if not m:
            continue
        block = [ln]
        for nxt in lines[i + 1 :]:
            if not nxt.startswith(("    ", "\t")):
                break
            block.append(nxt)
        blocks[m.group(1)] = "\n".join(block)
    return blocks


def _rehome_footnotes(chapters: list[Chapter]) -> int:
    """Move each definition to the file whose text refers to it."""
    defs = [_footnote_blocks(c.body) for c in chapters]
    refs = [set(FOOTNOTE_REF.findall(c.body)) for c in chapters]
    moved = 0
    for i, chapter in enumerate(chapters):
        missing = sorted(refs[i] - set(defs[i]))
        found = []
        for note in missing:
            owner = next((j for j, d in enumerate(defs) if note in d), None)
            if owner is None:
                continue
            block = defs[owner][note]
            found.append(block)
            if note not in refs[owner]:
                chapters[owner].body = chapters[owner].body.replace(block + "\n", "", 1)
            moved += 1
        if found:
            chapter.body = chapter.body.rstrip("\n") + "\n\n" + "\n".join(found) + "\n"
    return moved


def _split(
    text: str,
    level: int | None,
    section_chars: int,
    max_chars: int,
    min_chars: int,
) -> tuple[list[Chapter], int | None, int]:
    """Chapters, the level they were cut at, and footnotes moved."""
    lines = read_lines(text)
    if level is None:
        level = choose_level(lines, section_chars=section_chars, min_chars=min_chars)
    pieces = _merge_small(_cut(lines, level), min_chars) if level else [lines]
    anchored = any(ln.anchor for ln in lines)

    out: list[Chapter] = []
    for piece in pieces:
        title = _title(piece, level)
        parts = _split_oversized(piece, max_chars)
        for k, part in enumerate(parts, 1):
            pages = [ln.page for ln in part if ln.page is not None and ln.text.strip()]
            out.append(
                Chapter(
                    title=f"{title} ({k})" if len(parts) > 1 else title,
                    body=_render(part, anchored),
                    first_page=min(pages, default=None),
                    last_page=max(pages, default=None),
                )
            )
    return out, level, _rehome_footnotes(out)


def split_chapters(
    text: str,
    level: int | None = None,
    section_chars: int = SECTION_CHARS,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Chapter]:
    """The document as chapters, in order; text and page anchors preserved."""
    return _split(text, level, section_chars, max_chars, min_chars)[0]


# --------------------------------------------------------------------------
# the export
# --------------------------------------------------------------------------
def page_count(text: str) -> int:
    """Pages in the document: distinct page anchors, or an estimate from length.

    Distinct, because an anchor can repeat (a preamble shares page 0; a chapter
    opening mid-page re-emits its page).
    """
    numbers = {int(m.group(1)) for m in PAGE_ANCHOR.finditer(text)}
    return len(numbers) if numbers else max(1, math.ceil(len(text) / CHARS_PER_PAGE))


def book_prefix(name: str) -> str:
    """``2006_Dawkins_GodDelusion`` -> ``GodDelusion``: the book's short name.

    The output folders here are named year_author_title; the title part is what
    a reader recognises in a citation. A last part too short to name a book
    ("BT") keeps the whole name.
    """
    last = name.rsplit("_", 1)[-1].strip()
    return last if sum(ch.isalpha() for ch in last) >= 4 else name.strip()


def book_name(src_md: Path) -> str:
    """The book's own name: its output folder, when the file is that book's."""
    src_md = Path(src_md)
    folder = src_md.parent.name
    return folder if folder and src_md.stem.startswith(folder) else src_md.stem


def default_out_dir(src_md: Path) -> Path:
    """``output/NAME/chapters/<md stem>/``: the folder to copy into the vault.

    Named after the Markdown it came from, so an original and its translation
    each get their own folder rather than overwriting one another.
    """
    src_md = Path(src_md)
    return src_md.parent / "chapters" / src_md.stem


def file_name(prefix: str, number: int, title: str, width: int) -> str:
    return f"{prefix} {number:0{width}d} {title}.md"


@dataclass
class ExportResult:
    out_dir: Path
    level: int | None
    src: Path | None = None
    pages: int = 0
    skipped: bool = False  # an article: left as one note, nothing written
    files: list[Path] = field(default_factory=list)
    footnotes_moved: int = 0
    images_copied: int = 0
    images_missing: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.skipped:
            name = self.src.name if self.src else "the Markdown"
            return (
                f"{self.pages} pages: an article, not a book -- kept as one note, "
                f"no chapters written. Use {name} as it is."
            )
        lines = [
            f"{len(self.files)} files in {self.out_dir}",
            f"  chapter level: {'#' * self.level if self.level else 'none (no heading level splits it)'}",
        ]
        lines += [f"  {p.name}" for p in self.files]
        if self.footnotes_moved:
            lines.append(f"  footnotes moved to their reference: {self.footnotes_moved}")
        if self.images_copied:
            lines.append(f"  figures copied: {self.images_copied}")
        if self.images_missing:
            lines.append(f"  figures MISSING on disk: {', '.join(self.images_missing)}")
        if self.removed:
            lines.append(f"  removed stale: {', '.join(self.removed)}")
        return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    # Temp then rename, as everywhere here: a vault syncing the folder must
    # never pick up half a chapter.
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _copy_images(body: str, src_root: Path, out_dir: Path, res: ExportResult) -> None:
    """Copy each figure a chapter links, keeping the link exactly as written."""
    root = src_root.resolve()
    for target in IMAGE_TARGET.findall(body):
        if "://" in target or target.startswith(("/", "data:")):
            continue
        rel = unquote(target)
        src = (src_root / rel).resolve()
        if not src.is_relative_to(root):
            continue  # a link may not reach outside the book's folder
        if not src.is_file():
            if target not in res.images_missing:
                res.images_missing.append(target)
            continue
        dst = out_dir / rel
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        res.images_copied += 1


def export(
    src_md: Path,
    out_dir: Path | None = None,
    prefix: str | None = None,
    level: int | None = None,
    contents: bool = True,
    section_chars: int = SECTION_CHARS,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    min_pages: int = BOOK_MIN_PAGES,
) -> ExportResult:
    """Write ``src_md`` as one file per chapter into ``out_dir``.

    A document under ``min_pages`` is an article: nothing is written and the
    result says so (``skipped``).
    """
    src_md = Path(src_md)
    out_dir = Path(out_dir) if out_dir else default_out_dir(src_md)
    prefix = (prefix or book_prefix(book_name(src_md))).strip()
    text = src_md.read_text(encoding="utf-8")
    pages = page_count(text)
    if pages < min_pages:
        return ExportResult(out_dir=out_dir, level=None, src=src_md, pages=pages, skipped=True)

    parts, level, moved = _split(text, level, section_chars, max_chars, min_chars)
    res = ExportResult(
        out_dir=out_dir, level=level, src=src_md, pages=pages, footnotes_moved=moved
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(parts))))
    entries = []
    for n, chapter in enumerate(parts, 1):
        path = out_dir / file_name(prefix, n, chapter.title, width)
        _write(path, chapter.body)
        _copy_images(chapter.body, src_md.parent, out_dir, res)
        res.files.append(path)
        entries.append((path.stem, chapter.pages))

    if contents:
        toc = out_dir / file_name(prefix, 0, CONTENTS, width)
        body = [f"# {prefix}", ""] + [
            f"- [[{stem}]]" + (f" · {pages}" if pages else "") for stem, pages in entries
        ]
        _write(toc, "\n".join(body) + "\n")
        res.files.insert(0, toc)

    # Files this export wrote last time under names it no longer uses -- a
    # renamed or re-cut chapter -- would be indexed alongside their successors.
    # Only this book's prefix is touched; the user's own notes are not.
    keep = {p.name for p in res.files}
    for old in sorted(out_dir.glob(f"{glob_escape(prefix)} *.md")):
        if old.name not in keep and re.fullmatch(rf"{re.escape(prefix)} \d+ .+\.md", old.name):
            old.unlink()
            res.removed.append(old.name)

    return res


def glob_escape(s: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", s)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Split finished Markdown into one file per chapter, for a RAG vault."
    )
    ap.add_argument("src", type=Path, help="finished Markdown (stage 3 or 4)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <src folder>/chapters/<src stem>/")
    ap.add_argument("--prefix", default=None,
                    help="book prefix for every file name (default: the book's short name)")
    ap.add_argument("--level", type=int, default=None,
                    help="heading level that marks a chapter (default: chosen per book)")
    ap.add_argument("--no-contents", action="store_true", help="skip the 00 Contents note")
    ap.add_argument("--section-chars", type=int, default=SECTION_CHARS,
                    help="a level whose median piece is smaller than this is sections")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS,
                    help="cut a chapter larger than this at page anchors")
    ap.add_argument("--min-pages", type=int, default=BOOK_MIN_PAGES,
                    help="shorter documents are articles and stay one note; 0 splits anything")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS,
                    help="a smaller piece (a part's title page) joins the next chapter")
    args = ap.parse_args(argv)

    if not args.src.exists():
        print(f"error: {args.src} not found")
        return 2
    res = export(
        args.src, out_dir=args.out_dir, prefix=args.prefix, level=args.level,
        contents=not args.no_contents, section_chars=args.section_chars,
        max_chars=args.max_chars, min_chars=args.min_chars, min_pages=args.min_pages,
    )
    print(res.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
