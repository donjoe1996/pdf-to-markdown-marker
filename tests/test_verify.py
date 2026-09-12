"""Quality checks on transcribed Markdown.

The check that matters most is that fresh OCR actually happened. Its failure
mode is silent: marker reads the PDF's existing bad text instead of the page and
returns fluent Markdown made of the wrong characters. Nothing about the output
looks broken, so the check has to be provably able to catch it -- hence the
positive control below.
"""

from __future__ import annotations

import pymupdf

from bt.verify import (
    check_footnotes,
    check_greek,
    check_italics,
    check_not_embedded_layer,
    check_run_ons,
    check_text_layer_replaced,
    run_all,
)
from tests.conftest import marker_markdown


def page_body(i: int) -> str:
    """A page of placeholder prose.

    Deliberately over 200 characters: check_not_embedded_layer skips any page
    with less embedded text than that, so a shorter fixture would make the test
    pass by doing nothing.
    """
    return (
        f"Page {i} opens with a sentence that belongs to page {i} alone. "
        f"It continues at some length so the page carries enough text to "
        f"compare against, and mentions {i} again to keep pages distinct. "
        f"A third sentence closes page {i} without adding anything new."
    )


def pdf_with_text(path, bodies: list[str]):
    doc = pymupdf.open()
    for body in bodies:
        page = doc.new_page(width=500, height=700)
        page.insert_textbox(pymupdf.Rect(30, 30, 470, 670), body, fontsize=10)
    doc.save(path)
    doc.close()
    return path


def extracted(path) -> list[str]:
    """What the checker will read back, so a control can match it exactly."""
    with pymupdf.open(path) as doc:
        return [page.get_text("text") for page in doc]


# --------------------------------------------------------------------------
# content checks are informational unless asked for
# --------------------------------------------------------------------------
def test_absent_content_does_not_fail_by_default():
    """Most books have no Greek. Failing on that would make the report noise."""
    assert check_greek("plain english text").ok
    assert check_italics("no emphasis here").ok
    assert check_footnotes("no notes here").ok


def test_absent_content_fails_when_required():
    assert not check_greek("plain english", required=True).ok
    assert not check_italics("plain english", required=True).ok
    assert not check_footnotes("plain english", required=True).ok


def test_present_content_is_reported():
    assert check_greek("δῆλον γὰρ ὡς ὑμεῖς").ok
    assert check_italics("this is *emphasised* text").ok
    assert check_footnotes("a marker<sup>1</sup>").ok


# --------------------------------------------------------------------------
# damaged-text detection
# --------------------------------------------------------------------------
def test_run_ons_flag_lost_word_spacing():
    """Collapsed spacing is the signature failure of old OCR."""
    bad = " ".join("thisisaverylongrunontogetherword" for _ in range(10))
    assert not check_run_ons(bad).ok
    assert check_run_ons("ordinary words with spaces between them").ok


def test_stale_marker_list_is_overridable():
    """The default list is specific to one scan; other documents differ."""
    assert not check_text_layer_replaced("contains itse1f somewhere").ok
    assert check_text_layer_replaced("clean text").ok
    assert not check_text_layer_replaced("has wibble", markers=["wibble"]).ok


# --------------------------------------------------------------------------
# the general fresh-OCR check
# --------------------------------------------------------------------------
def test_reused_text_layer_is_caught(tmp_path):
    """POSITIVE CONTROL: feed it the PDF's own text and it must fail.

    A check that has never been seen to fail is not known to work.
    """
    pdf = pdf_with_text(tmp_path / "doc.pdf", [page_body(i) for i in range(6)])
    identical = marker_markdown(extracted(pdf))

    finding = check_not_embedded_layer(identical, pdf)
    assert not finding.ok
    assert "identical" in finding.detail


def test_genuinely_different_output_passes(tmp_path):
    pdf = pdf_with_text(tmp_path / "doc.pdf", [page_body(i) for i in range(6)])
    fresh = marker_markdown(
        [
            f"Quite different wording for page {i}. The transcription shares "
            f"almost no phrasing with what the file already contained, as a "
            f"genuine re-reading of the image would not. Page {i} ends here."
            for i in range(6)
        ]
    )
    assert check_not_embedded_layer(fresh, pdf).ok


def test_comparison_is_page_aligned(tmp_path):
    """REGRESSION: an earlier version compared whole-document blobs.

    It sampled pages the output did not even cover, so it passed because
    different pages always look different -- the right verdict for the wrong
    reason. Here the output covers only pages 0-2 of a 9-page PDF and is
    identical to them, so an aligned comparison must still fail.
    """
    pdf = pdf_with_text(tmp_path / "doc.pdf", [page_body(i) for i in range(9)])
    partial = marker_markdown(extracted(pdf)[:3])

    assert not check_not_embedded_layer(partial, pdf).ok


def test_run_all_skips_ocr_checks_for_extraction(tmp_path):
    """For a born-digital extract, matching the text layer is correct."""
    pdf = pdf_with_text(tmp_path / "doc.pdf", [page_body(i) for i in range(6)])
    text = marker_markdown(extracted(pdf))

    names = {f.name for f in run_all(text, pdf_path=pdf, fresh_ocr=False)}
    assert "fresh OCR" not in names
    assert all(f.ok for f in run_all(text, pdf_path=pdf, fresh_ocr=False))
