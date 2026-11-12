"""Per-page audit: the checks that give a bad page a number.

Whole-document checks pass on a book with one half-empty page in fifty, because
the damage is local and the measure is global. Everything here is about making
that page visible -- and, as with verify, each detector has a positive control,
since a detector never seen firing is not known to work.
"""

from __future__ import annotations

import pymupdf

from bt.audit import audit_pages, find_repetition, summarise, text_amount
from bt.postprocess import process
from bt.verify import check_pages, run_all
from tests.conftest import marker_markdown
from tests.test_verify import extracted, page_body, pdf_with_text


def bodies(n: int = 6) -> list[str]:
    return [page_body(i) * 3 for i in range(n)]


def test_complete_transcription_has_no_issues(tmp_path):
    pdf = pdf_with_text(tmp_path / "ok.pdf", bodies())
    assert audit_pages(marker_markdown(extracted(pdf)), pdf) == []


def test_page_that_lost_text_is_named(tmp_path):
    """marker drops an OCR block whose output loops, and logs it. The Markdown
    just has less text on that page -- fluent, plausible, and short."""
    pdf = pdf_with_text(tmp_path / "short.pdf", bodies())
    pages = extracted(pdf)
    pages[3] = pages[3][: len(pages[3]) // 3]
    issues = audit_pages(marker_markdown(pages), pdf)
    assert [(i.page, i.kind, i.strong) for i in issues] == [(3, "missing", True)]


def test_page_with_invented_text_is_named(tmp_path):
    pdf = pdf_with_text(tmp_path / "long.pdf", bodies())
    pages = extracted(pdf)
    pages[2] = pages[2] + pages[4] + pages[5]
    issues = audit_pages(marker_markdown(pages), pdf)
    assert [(i.page, i.kind) for i in issues] == [(2, "extra")]


def test_audit_still_works_on_the_finished_markdown(tmp_path):
    """Post-processing replaces the separators with anchors and moves a little
    text between pages; the audit must read both forms."""
    pdf = pdf_with_text(tmp_path / "final.pdf", bodies())
    pages = extracted(pdf)
    pages[1] = pages[1][:80]
    cleaned, _ = process(marker_markdown(pages))
    assert [i.page for i in audit_pages(cleaned, pdf)] == [1]


def test_markup_is_not_counted_as_text():
    """Footnote ids and image paths are letters and digits too. Counted, a page
    of figures would look longer than its text layer."""
    plain = "Some words here."
    marked = (
        "<!-- page 3 -->\nSome words[^p3-1] here.\n"
        "![](book_images/0000-0009_page_3_Figure_2.jpeg)"
    )
    assert text_amount(marked) == text_amount(plain)


def test_mid_page_loop_is_detected():
    """marker only catches a loop that runs to the end of the output."""
    line = "The same sentence over and over again."
    body = "A real opening sentence.\n" + "\n".join([line] * 9) + "\nAnd it goes on."
    assert find_repetition(body) == (line.lower(), 9)
    issues = audit_pages(marker_markdown([body]))
    assert [(i.kind, i.strong) for i in issues] == [("repeated", True)]


def test_ordinary_prose_is_not_a_loop():
    assert find_repetition(page_body(1) * 2) is None


def test_scanned_page_that_came_out_empty_is_reported_not_failed(make_scan_pdf):
    """With --no-ocr a scanned page has no text to extract and comes out
    empty. It is reported, but weakly: a page of ink and no text is usually a
    plate, and failing every illustrated book would teach people to ignore
    the check."""
    pdf = make_scan_pdf("scan.pdf", pages=3)
    text = marker_markdown(["", "Only this page produced text, some of it.", ""])
    issues = audit_pages(text, pdf)
    assert [(i.page, i.kind, i.strong) for i in issues] == [
        (0, "empty", False),
        (2, "empty", False),
    ]
    assert check_pages(text, pdf).ok


def test_blank_page_that_came_out_blank_is_fine(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    doc.close()
    assert audit_pages(marker_markdown([""]), tmp_path / "blank.pdf") == []


def test_unpaginated_text_is_not_measured_against_page_zero(tmp_path):
    """Without separators the whole document is one "page 0"; measured against
    the PDF's first page it would always read as invented text."""
    pdf = pdf_with_text(tmp_path / "flat.pdf", bodies())
    assert audit_pages("\n\n".join(extracted(pdf)), pdf) == []


def test_split_book_is_measured_against_the_visible_half(make_pdf, tmp_path):
    """A split page is a crop box over a spread, and the other half's text is
    still in the file. Counting it would double every expectation and flag
    the whole book as missing text."""
    from bt.split_spreads import split_document

    spread = make_pdf("spread.pdf", pages=3, spread=True)
    split = tmp_path / "pages.pdf"
    split_document(spread, split, tmp_path / "pagemap.json")
    with pymupdf.open(split) as doc:
        halves = [p.get_text("text", clip=p.rect) for p in doc]
    assert len(halves) == 6
    assert audit_pages(marker_markdown(halves), split) == []


def test_verification_fails_on_a_page_that_lost_text(tmp_path):
    pdf = pdf_with_text(tmp_path / "v.pdf", bodies())
    pages = extracted(pdf)
    pages[4] = "Almost nothing."
    finding = next(
        f for f in run_all(marker_markdown(pages), pdf_path=pdf, fresh_ocr=False)
        if f.name == "page audit"
    )
    assert not finding.ok
    assert "page 4" in finding.detail


def test_summary_names_pages_and_caps_the_list():
    from bt.audit import PageIssue

    issues = [PageIssue(n, "missing", "short", True) for n in range(10)]
    line = summarise(issues, limit=3)
    assert "10 page(s) lost or repeated text" in line
    assert "page 2: short" in line and "page 3: short" not in line
    assert "and 7 more" in line
    assert summarise([]) == "no page looks incomplete or repeated"
