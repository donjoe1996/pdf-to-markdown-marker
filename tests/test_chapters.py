"""Chapter export: one Markdown file per chapter, for a RAG vault.

Why it exists: the vault's Ask feature takes at most 3 passages from any one
note. A whole book as one note can therefore supply 3 of the 6 passages an
answer is built from, however relevant the rest of it is. Split into chapters,
the same book can supply all 6, from different chapters.

What must not break while splitting -- each of these is a silent failure, the
files would still read perfectly:

* text lost or duplicated at a cut;
* a page anchor renumbered, or missing at the top of a chapter that starts
  mid-page, so the page readout and citations point at the wrong page;
* a footnote reference left in one file with its definition in another;
* two books' files sharing a name, since links resolve by file name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bt import chapters
from bt.postprocess import PAGE_ANCHOR, PAGE_ANCHOR_FMT

# Small thresholds so a synthetic book of a few hundred characters behaves like
# a real one of a few hundred thousand. The ratios are what is being tested.
SMALL = dict(section_chars=200, max_chars=5000, min_chars=40)


def prose(tag: str, n: int = 3) -> str:
    """A paragraph long enough to count as text, unique per tag."""
    return " ".join(f"Sentence {k} of {tag} carries placeholder prose." for k in range(n))


def doc(pages: list[str], start: int = 0) -> str:
    return "\n\n".join(
        f"{PAGE_ANCHOR_FMT.format(n=i)}\n\n{body.strip()}"
        for i, body in enumerate(pages, start=start)
    )


def anchors(text: str) -> list[int]:
    return [int(m.group(1)) for m in PAGE_ANCHOR.finditer(text)]


def prose_lines(text: str) -> list[str]:
    return [
        ln for ln in text.splitlines()
        if ln.strip() and not PAGE_ANCHOR.match(ln) and not ln.startswith("#")
    ]


@pytest.fixture
def three_chapters() -> str:
    """Book title, then three chapters; chapter 2 starts mid-page."""
    return doc([
        "# The Placeholder Book\n\nBy Nobody.",
        f"## One beginning\n\n{prose('c1-a')}",
        f"{prose('c1-b')}\n\n## Two middle\n\n{prose('c2-a')}",
        f"{prose('c2-b')}",
        f"## Three end\n\n{prose('c3-a')}\n\n{prose('c3-b')}",
    ], start=10)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def test_splits_at_the_chapter_level_not_the_book_title(three_chapters):
    """One `#` (the book's title) over `##` chapters is the common shape.

    Splitting at `#` would give one file -- exactly the 3-passage cap this stage
    exists to escape -- so the level is chosen from the document, not taken as `#`.
    """
    parts = chapters.split_chapters(three_chapters, **SMALL)

    assert [c.title for c in parts] == ["One beginning", "Two middle", "Three end"]
    assert chapters.choose_level(three_chapters, **SMALL) == 2


def test_no_text_is_lost_or_duplicated(three_chapters):
    parts = chapters.split_chapters(three_chapters, **SMALL)
    joined = [ln for c in parts for ln in prose_lines(c.body)]

    assert sorted(joined) == sorted(prose_lines(three_chapters))
    assert len(joined) == len(set(joined))
    # the headings, too
    assert sum(c.body.count("## ") for c in parts) == 3


def test_page_anchors_keep_the_books_numbers(three_chapters):
    """Never restart at 1 per chapter: the readout and citations use these."""
    parts = chapters.split_chapters(three_chapters, **SMALL)

    assert [anchors(c.body) for c in parts] == [[10, 11, 12], [12, 13], [14]]
    assert [(c.first_page, c.last_page) for c in parts] == [(10, 12), (12, 13), (14, 14)]


def test_a_chapter_starting_mid_page_opens_with_that_pages_anchor(three_chapters):
    """Chapter 2 opens on page 12, below the end of chapter 1.

    Without a re-emitted anchor, the first passage of the file has no page at
    all -- or, read by a tool that carries the last anchor forward, the page of
    whatever note it read before.
    """
    two = chapters.split_chapters(three_chapters, **SMALL)[1]
    lines = [ln for ln in two.body.splitlines() if ln.strip()]

    assert lines[0] == PAGE_ANCHOR_FMT.format(n=12)
    assert lines[1] == "## Two middle"


def test_no_file_ends_on_an_orphan_anchor(three_chapters):
    """Page 14's anchor sits just above chapter 3's heading: it belongs to 3."""
    for c in chapters.split_chapters(three_chapters, **SMALL):
        last = [ln for ln in c.body.splitlines() if ln.strip()][-1]
        assert not PAGE_ANCHOR.match(last), c.title


def test_sections_are_not_split_out():
    """`##` sections under `#` chapters are below chapter size: more files, no gain."""
    text = doc([
        f"# Chapter {k}\n\n" + "\n\n".join(f"## Section {k}.{s}\n\n{prose(f'{k}{s}', 2)}"
                                           for s in (1, 2, 3))
        for k in (1, 2)
    ])
    assert chapters.choose_level(text, **SMALL) == 1


def test_plain_top_level_chapters_split_at_one_hash():
    text = doc([f"# Chapter {k}\n\n{prose(f'ch{k}')}" for k in range(1, 4)])
    assert chapters.choose_level(text, **SMALL) == 1
    assert len(chapters.split_chapters(text, **SMALL)) == 3


def test_parts_over_chapters_split_at_the_chapters():
    """`#` parts with `##` chapters under them: the deepest chapter-sized level wins.

    Parts are not wrong-sized for a file, but a part can still only give 3
    passages; its chapters can give 3 each.
    """
    pages = []
    for part in (1, 2):
        pages.append(f"# Part {part}")
        for ch in (1, 2, 3):
            pages.append(f"## Chapter {part}.{ch}\n\n{prose(f'p{part}c{ch}', 6)}")
    text = doc(pages)

    assert chapters.choose_level(text, **SMALL) == 2
    parts = chapters.split_chapters(text, **SMALL)
    assert [c.title for c in parts] == [f"Chapter {p}.{c}" for p in (1, 2) for c in (1, 2, 3)]


def test_a_bare_part_title_joins_the_chapter_after_it():
    """A part's title page is a heading and nothing else.

    As its own note it is a file Ask can only ever answer "Part 2" from; joined
    forward it becomes the top of the heading path of the chapter it opens.
    """
    pages = [
        f"# Part 1\n\n## Chapter A\n\n{prose('a', 6)}",
        "# Part 2",
        f"## Chapter B\n\n{prose('b', 6)}",
    ]
    parts = chapters.split_chapters(doc(pages), **SMALL)

    assert [c.title for c in parts] == ["Chapter A", "Chapter B"]
    assert "# Part 2" in parts[1].body
    assert anchors(parts[1].body) == [1, 2]


def test_text_before_the_first_heading_is_front_matter():
    text = doc([prose("preface", 4), f"# Chapter 1\n\n{prose('c1', 4)}"])
    parts = chapters.split_chapters(text, **SMALL)
    assert [c.title for c in parts] == ["Front matter", "Chapter 1"]


def test_a_heading_inside_a_code_fence_is_not_a_chapter():
    text = doc([
        f"# Real\n\n{prose('r', 4)}\n\n```\n# not a heading\n```\n\n{prose('s', 4)}",
        f"# Also real\n\n{prose('t', 4)}",
    ])
    assert [c.title for c in chapters.split_chapters(text, **SMALL)] == ["Real", "Also real"]


def test_an_oversized_chapter_is_cut_at_page_anchors():
    """A book with no usable headings is one 'chapter' -- back at the cap.

    Cutting at page anchors (never mid-page) keeps every page whole and every
    file under the size where the editor falls back to plain view.
    """
    text = doc([prose(f"pg{k}", 4) for k in range(12)])
    parts = chapters.split_chapters(text, section_chars=200, max_chars=600, min_chars=40)

    assert len(parts) > 2
    assert [c.title for c in parts] == [f"Front matter ({k})" for k in range(1, len(parts) + 1)]
    assert sum((anchors(c.body) for c in parts), []) == list(range(12))
    assert sorted(ln for c in parts for ln in prose_lines(c.body)) == sorted(prose_lines(text))


# --------------------------------------------------------------------------
# footnotes
# --------------------------------------------------------------------------
def test_a_footnote_follows_its_reference_across_a_cut():
    """Chapter 2 opens mid-page 5; the note for chapter 1's text sits below it.

    Footnotes resolve within a file, so left where it was the reference points
    at nothing in one note and the definition dangles in the other.
    """
    text = doc([
        f"# One\n\n{prose('one', 4)}",
        f"{prose('end of one')}[^p5-1]\n\n# Two\n\n{prose('two', 4)}[^p5-2]\n\n"
        "[^p5-1]: The note for chapter one.\n[^p5-2]: The note for chapter two.",
    ], start=4)
    one, two = chapters.split_chapters(text, **SMALL)

    assert "[^p5-1]: The note for chapter one." in one.body
    assert "[^p5-1]:" not in two.body
    assert "[^p5-2]: The note for chapter two." in two.body


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heading, title",
    [
        ("4. Why: *there* / is no God?", "4. Why there is no God"),
        ("[The Ultimate 747](#page-3-0)<sup>2</sup>", "The Ultimate 747"),
        ("**Being-in-the-World** [^p9-1]", "Being-in-the-World"),
        ("#hash | pipe ^caret [x]", "hash pipe caret x"),
        ("Trailing dots...", "Trailing dots"),
    ],
)
def test_titles_are_safe_as_file_names_and_link_targets(heading, title):
    """`/` makes a folder, `#`, `^`, `|` and `[]` break a [[link]] to the note."""
    assert chapters.clean_title(heading) == title


def test_a_long_title_is_cut_at_a_word():
    t = chapters.clean_title("word " * 40)
    assert len(t) <= chapters.TITLE_MAX and not t.endswith(" ") and t.endswith("word")


@pytest.mark.parametrize(
    "name, prefix",
    [
        ("2006_Dawkins_GodDelusion", "GodDelusion"),
        ("2022_Esteban_TheVegetation CalakmulCampecheMexico",
         "TheVegetation CalakmulCampecheMexico"),
        ("meditationsofmar00marc", "meditationsofmar00marc"),
        ("1962_Heidegger_BT", "1962_Heidegger_BT"),  # last part too short to name it
    ],
)
def test_the_book_prefix_is_its_short_name(name, prefix):
    assert chapters.book_prefix(name) == prefix


# --------------------------------------------------------------------------
# the export
# --------------------------------------------------------------------------
@pytest.fixture
def book(tmp_path, three_chapters) -> Path:
    folder = tmp_path / "output" / "2006_Dawkins_GodDelusion"
    folder.mkdir(parents=True)
    md = folder / "2006_Dawkins_GodDelusion.english.md"
    md.write_text(three_chapters, encoding="utf-8")
    return md


def test_export_writes_one_uniquely_named_file_per_chapter(book):
    res = chapters.export(book, **SMALL)

    assert res.out_dir == book.parent / "chapters" / "2006_Dawkins_GodDelusion.english"
    names = sorted(p.name for p in res.out_dir.glob("*.md"))
    assert names == [
        "GodDelusion 00 Contents.md",
        "GodDelusion 01 One beginning.md",
        "GodDelusion 02 Two middle.md",
        "GodDelusion 03 Three end.md",
    ]
    assert res.level == 2


def test_contents_links_every_chapter_with_its_pages(book):
    res = chapters.export(book, **SMALL)
    toc = (res.out_dir / "GodDelusion 00 Contents.md").read_text(encoding="utf-8")

    links = re.findall(r"\[\[([^\]]+)\]\]", toc)
    assert links == [p.stem for p in sorted(res.out_dir.glob("GodDelusion 0[1-9]*.md"))]
    assert "pp. 12–13" in toc
    # no page anchors: the contents note has no pages of its own
    assert not anchors(toc)


def test_contents_can_be_left_out(book):
    res = chapters.export(book, contents=False, **SMALL)
    assert not list(res.out_dir.glob("* 00 Contents.md"))


def test_a_rerun_replaces_its_own_files_and_nothing_else(book):
    """Renamed chapters must not leave their old files to be indexed twice."""
    res = chapters.export(book, **SMALL)
    stale = res.out_dir / "GodDelusion 07 Old name.md"
    stale.write_text("old", encoding="utf-8")
    mine = res.out_dir / "My own note.md"
    mine.write_text("keep", encoding="utf-8")

    chapters.export(book, **SMALL)

    assert not stale.exists()
    assert mine.read_text(encoding="utf-8") == "keep"


def test_figures_travel_with_the_folder(book):
    """The chapter folder is what gets copied into the vault; links stay as written."""
    images = book.parent / "2006_Dawkins_GodDelusion_images"
    images.mkdir()
    (images / "0000-0009_page_3_Figure_2.jpeg").write_bytes(b"jpeg")
    link = "![](2006_Dawkins_GodDelusion_images/0000-0009_page_3_Figure_2.jpeg)"
    gone = "![](2006_Dawkins_GodDelusion_images/missing.jpeg)"
    book.write_text(book.read_text(encoding="utf-8") + f"\n\n{link}\n\n{gone}", encoding="utf-8")

    res = chapters.export(book, **SMALL)

    assert (res.out_dir / "2006_Dawkins_GodDelusion_images"
            / "0000-0009_page_3_Figure_2.jpeg").read_bytes() == b"jpeg"
    assert link in (res.out_dir / "GodDelusion 03 Three end.md").read_text(encoding="utf-8")
    assert res.images_copied == 1
    assert res.images_missing == ["2006_Dawkins_GodDelusion_images/missing.jpeg"]


def test_a_percent_encoded_figure_link_is_found_on_disk(tmp_path):
    """Book names contain spaces; transcribe.link_target() escapes them."""
    folder = tmp_path / "A Book"
    (folder / "A Book_images").mkdir(parents=True)
    (folder / "A Book_images" / "f.png").write_bytes(b"png")
    md = folder / "A Book.md"
    md.write_text(doc([f"# One\n\n{prose('x', 4)}\n\n![](A%20Book_images/f.png)"]),
                  encoding="utf-8")

    res = chapters.export(md, **SMALL)

    assert (res.out_dir / "A Book_images" / "f.png").exists()
    assert res.images_missing == []


def test_a_link_cannot_copy_from_outside_the_book(tmp_path, book):
    (tmp_path / "secret.png").write_bytes(b"x")
    book.write_text(book.read_text(encoding="utf-8") + "\n\n![](../../secret.png)",
                    encoding="utf-8")
    res = chapters.export(book, **SMALL)
    assert res.images_copied == 0
    assert not list(res.out_dir.rglob("secret.png"))


def test_main_prints_where_the_files_went(book, capsys):
    assert chapters.main([str(book), "--prefix", "GD", "--section-chars", "200",
                          "--min-chars", "40"]) == 0
    out = capsys.readouterr().out
    assert "GD 01 One beginning.md" in out


# --------------------------------------------------------------------------
# wiring: a finished translation is exported; a part-done one is not
# --------------------------------------------------------------------------
class _Upper:
    input_tokens = output_tokens = 0

    def __init__(self, *a, **kw):
        pass

    def __call__(self, text, target):
        return text


class _Broken(_Upper):
    def __call__(self, text, target):
        raise RuntimeError("rate limited")


@pytest.mark.parametrize("backend, exported", [(_Upper, True), (_Broken, False)])
def test_translate_exports_chapters_only_when_the_book_is_complete(
    book, monkeypatch, backend, exported
):
    """A part-translated book split into chapters looks finished: its last
    chapter simply ends early, and nothing in the vault says so."""
    from bt import translate

    monkeypatch.setattr(translate, "OpenAICompatTranslator", backend)
    src = book.with_name("2006_Dawkins_GodDelusion.md")
    book.rename(src)
    out = src.with_name("2006_Dawkins_GodDelusion.english.md")

    translate.main([str(src), str(out), "--provider", "local"])

    folder = src.parent / "chapters" / out.stem
    assert folder.exists() is exported
