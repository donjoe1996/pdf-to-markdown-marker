"""Text transforms: unit tests plus a golden end-to-end case.

Most of these pin bugs that actually happened. Where that is so, the test says
which one -- a test whose purpose is remembered is far likelier to survive a
future refactor than one that merely asserts something true.
"""

from __future__ import annotations

import pytest

from bt.postprocess import (
    PAGE_SEPARATOR,
    Stats,
    _could_be_head,
    find_running_heads,
    join_hyphens,
    namespace_footnotes,
    normalise_head,
    process,
    reflow_lines,
    split_pages,
)
from tests.conftest import marker_markdown


# --------------------------------------------------------------------------
# page separators
# --------------------------------------------------------------------------
def test_separator_matches_markers_braced_form():
    """REGRESSION: the pattern once required a bare rule of dashes.

    marker writes "{3}------", so a dashes-only pattern matched nothing, every
    page merged into one, and per-page footnote namespacing silently stopped
    working.
    """
    assert PAGE_SEPARATOR.search("{3}" + "-" * 48)
    assert not PAGE_SEPARATOR.search("-" * 48), "a bare rule is not a separator"


def test_split_pages_recovers_marker_page_numbers():
    text = marker_markdown(["first", "second", "third"], start=5)
    assert [n for n, _ in split_pages(text)] == [5, 6, 7]


def test_split_pages_without_separators_is_one_page():
    assert [n for n, _ in split_pages("no separators here")] == [0]


# --------------------------------------------------------------------------
# running heads
# --------------------------------------------------------------------------
def test_normalise_head_erases_page_numbers_and_emphasis():
    a = normalise_head("**82  A Title  II. 3**")
    b = normalise_head("7  A Title  II. 1")
    assert a == b, "heads differing only in numbers must normalise identically"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("82  A Title  II. 3", True),
        ("INT. I  A Title  27", True),
        # REGRESSION: templated body text repeats across pages too, and an
        # earlier version removed it. A head is a label, not a sentence.
        ("Body text unique to page 7, which should survive.", False),
        ("<sup>1</sup> A footnote on page 3.", False),
        ("", False),
    ],
)
def test_could_be_head_shape(line, expected):
    assert _could_be_head(line) is expected


def test_find_running_heads_needs_repetition_across_pages():
    pages = [
        (i, f"12  A Title  II. {i}\n\nUnique body {i} goes here, as a sentence.")
        for i in range(10)
    ]
    assert find_running_heads(pages) == {"# a title ii. #"}


def test_find_running_heads_known_limit_unpunctuated_repeated_line():
    """A documented boundary of the heuristic, not an aspiration.

    A short line that recurs at a page edge, carries no sentence-ending
    punctuation and differs only in digits is indistinguishable from a running
    head by these rules, so it is treated as one. Real body prose ends in
    punctuation, which is what keeps this from firing in practice -- but if a
    document ever loses content this is the first thing to suspect.
    """
    pages = [(i, f"12  A Title  II. {i}\n\nCaption {i} without a full stop") for i in range(10)]
    assert "caption # without a full stop" in find_running_heads(pages)


def test_find_running_heads_ignores_short_documents():
    """Three pages is not evidence; anything can repeat twice."""
    pages = [(i, f"12  A Title  II. {i}\n\nbody") for i in range(3)]
    assert find_running_heads(pages) == set()


def test_process_removes_heads_but_keeps_body_and_footnotes(book_markdown):
    cleaned, stats = process(book_markdown(8))
    assert "A Placeholder Title" not in cleaned
    assert stats.heads_removed == 8
    assert "Body text unique to page 5" in cleaned
    assert "A note attached to page 5" in cleaned


# --------------------------------------------------------------------------
# footnotes
# --------------------------------------------------------------------------
def test_footnote_ids_are_namespaced_per_page():
    """REGRESSION: footnote "1" recurs on nearly every page.

    Without a per-page namespace the same id is defined hundreds of times in one
    document and no reference resolves.
    """
    stats = Stats()
    first = namespace_footnotes("text<sup>1</sup>", 4, stats)
    second = namespace_footnotes("text<sup>1</sup>", 9, stats)
    assert "[^p4-1]" in first
    assert "[^p9-1]" in second
    assert first != second


def test_footnote_definition_gets_a_colon():
    """A marker opening a line defines the note; Markdown needs the colon."""
    out = namespace_footnotes("<sup>2</sup> The note body.", 0, Stats())
    assert out.startswith("[^p0-2]: ")


def test_process_produces_unique_footnote_ids(book_markdown):
    cleaned, _ = process(book_markdown(6))
    refs = {line for line in cleaned.splitlines() if line.startswith("[^")}
    assert len(refs) == 6, "one distinct footnote definition per page"


# --------------------------------------------------------------------------
# hyphenation
# --------------------------------------------------------------------------
def test_join_hyphens_rejoins_split_words():
    stats = Stats()
    assert "hyphenbreak" in join_hyphens("hyphen-\nbreak", stats)
    assert stats.hyphens_joined == 1


def test_join_hyphens_leaves_dashes_alone():
    stats = Stats()
    text = "an em-dash — stays\nand a normal line break"
    assert join_hyphens(text, stats) == text
    assert stats.hyphens_joined == 0


# --------------------------------------------------------------------------
# golden: the whole transform in one assertion
# --------------------------------------------------------------------------
def test_golden_document(book_markdown, request):
    """Pin the full output. Any unintended change to any transform fails here.

    Regenerate deliberately with: pytest --golden-update
    """
    cleaned, _ = process(book_markdown(4))
    golden = request.path.parent / "golden" / "book.md"

    if request.config.getoption("--golden-update"):
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(cleaned, encoding="utf-8")
        pytest.skip("golden file rewritten")

    assert golden.exists(), "missing golden file; run with --golden-update"
    assert cleaned == golden.read_text(encoding="utf-8")


def test_page_anchors_survive_post_processing(book_markdown):
    """REGRESSION: post-processing deleted the page markers entirely.

    Found on the first completed book. `raw.md` had 256 `{N}` separators and the
    final `.md` had none, because process() split on them and rejoined without
    putting anything back. That silently disabled the most important check in
    verify.py, which aligns output pages against PDF pages by those numbers --
    it reported "ok / no meaningful embedded text" for a whole book, a pass that
    proved nothing.
    """
    cleaned, _ = process(book_markdown(6))
    assert [n for n, _ in split_pages(cleaned)] == [0, 1, 2, 3, 4, 5]


def test_margin_numbers_off_by_default(book_markdown):
    """marker discards marginal numbers, so the transform is a no-op here."""
    _, stats = process(book_markdown(4))
    assert stats.margins_converted == 0


def test_image_links_are_not_stripped_as_running_heads():
    """A figure on most pages normalises to the same line on every page.

    Running-head removal works by repetition with digits normalised away, and
    `images/0000-0009_page_3_Figure_2.jpeg` collapses to the same form as
    `images/0010-0019_page_7_Figure_1.jpeg`. A book with a chart on most pages
    would have every one of its figures deleted -- the text would still read
    perfectly, with the pictures silently gone.
    """
    pages = [
        f"![](images/{i:04d}-{i:04d}_page_{i}_Figure_{i}.jpeg)\n\n"
        f"Body text unique to page {i}, long enough to read as a sentence.\n"
        for i in range(8)
    ]
    out, stats = process(marker_markdown(pages))

    assert out.count("![](images/") == 8
    assert stats.heads_removed == 0


# --------------------------------------------------------------------------
# reflow: soft line breaks inside a paragraph
# --------------------------------------------------------------------------
WORDS = "On return, they told Drona the entire story. O king! Kounteya Arjuna".split()


def test_reflow_joins_one_word_per_line():
    """REGRESSION: a born-digital book came out one word per line.

    With ``--no-ocr`` marker keeps every line break pdftext reports, and
    markdownify preserves single newlines. On that PDF pdfium broke after every
    word, so `0160-0169.md` was a column of single words. It still *rendered*
    as prose -- Markdown folds a single newline into a space -- which is why it
    looked fine in a viewer and was unusable as text (translation, grep, diff).
    """
    stats = Stats()
    out = reflow_lines("\n".join(WORDS), stats)
    assert out == " ".join(WORDS)
    assert stats.lines_joined == len(WORDS) - 1


def test_reflow_keeps_paragraph_breaks():
    text = "first line\nwrapped here\n\nsecond para\nwrapped too"
    assert reflow_lines(text, Stats()) == "first line wrapped here\n\nsecond para wrapped too"


@pytest.mark.parametrize(
    "block",
    [
        "# A heading\nBody right under it",
        "- item one\n- item two",
        "1. first\n2. second",
        "| a | b |\n|---|---|\n| 1 | 2 |",
        "> quoted\n> still quoted",
        "![](book_images/0000-0009_page_3_Figure_2.jpeg)\nCaption under it",
        "hard break  \nnext line",
        "$$\nx = 1\n$$",
        "<table>\n<tr><td>a</td></tr>\n</table>",
    ],
)
def test_reflow_leaves_markdown_structure_alone(block):
    """Only plain prose lines are joined; anything Markdown reads as structure
    keeps its own line, or it would stop being that structure."""
    assert reflow_lines(block, Stats()).split("\n")[0] == block.split("\n")[0]


def test_reflow_does_not_touch_fenced_code():
    code = "```\nline one\nline two\n\nline four\n```"
    assert reflow_lines(code, Stats()) == code


def test_reflow_joins_list_item_continuations():
    assert reflow_lines("- an item that\nwraps", Stats()) == "- an item that wraps"


def test_inline_footnote_ref_at_line_start_stays_a_reference():
    """A wrapped line can begin with a reference (`on?<sup>366</sup> He ...`
    broken before the marker). Left on its own line, namespacing reads any
    marker that opens a line as a *definition* and adds a colon -- a reference
    in the middle of a sentence silently becomes a bogus footnote."""
    out, _ = process(marker_markdown(["while he looked on?\n<sup>366</sup>\nHe has a power"]))
    assert "on?[^p0-366] He has a power" in out or "on? [^p0-366] He has a power" in out
    assert "[^p0-366]:" not in out


def test_footnote_block_keeps_one_note_per_line():
    """A footnote paragraph lists several notes, each its own definition."""
    text = "<sup>1</sup> First note\nwraps here.\n<sup>2</sup> Second note."
    out = reflow_lines(text, Stats())
    assert out == "<sup>1</sup> First note wraps here.\n<sup>2</sup> Second note."


def test_reflow_without_dehyphenation_does_not_split_words_with_a_space():
    out, _ = process(marker_markdown(["a hyphen-\nbreak"]), hyphens=False)
    assert "hyphen- break" not in out


def test_process_reflows_word_per_line_page():
    pages = ["\n".join(WORDS) + "\n\nNext\nparagraph."]
    out, stats = process(marker_markdown(pages))
    assert " ".join(WORDS) + "\n\nNext paragraph." in out
    assert stats.lines_joined == len(WORDS)


def test_reflow_can_be_turned_off():
    out, _ = process(marker_markdown(["\n".join(WORDS)]), reflow=False)
    assert "\n".join(WORDS) in out
