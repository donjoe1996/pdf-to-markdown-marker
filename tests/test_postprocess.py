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


def test_margin_numbers_off_by_default(book_markdown):
    """marker discards marginal numbers, so the transform is a no-op here."""
    _, stats = process(book_markdown(4))
    assert stats.margins_converted == 0
