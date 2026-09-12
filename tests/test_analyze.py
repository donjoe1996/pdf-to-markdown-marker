"""Document inspection: the two decisions that cost hours when wrong.

``analyze`` answers *split or not* and *OCR or not* before a run starts. Getting
either wrong is expensive and quiet: splitting a single-page PDF cuts every page
in half, and OCRing a born-digital file burns hours reproducing text that could
be extracted in seconds.

These are characterization tests — they describe what the code does today, so a
later change has something to be checked against.
"""

from __future__ import annotations

import pymupdf

from bt.analyze import (
    EXTRACT_SECONDS_PER_PAGE,
    OCR_SECONDS_PER_PAGE,
    Analysis,
    _page_image_coverage,
    analyze,
)


# --------------------------------------------------------------------------
# the decisive signal: is the page a photograph of paper?
# --------------------------------------------------------------------------
def test_page_image_coverage_detects_a_full_page_scan(make_scan_pdf, make_pdf):
    scan = make_scan_pdf("scan.pdf", pages=1)
    typeset = make_pdf("typeset.pdf", pages=1)

    with pymupdf.open(scan) as doc:
        assert _page_image_coverage(doc[0]) >= 0.9
    with pymupdf.open(typeset) as doc:
        assert _page_image_coverage(doc[0]) == 0.0


def test_text_over_a_full_page_image_is_never_trusted(make_scan_pdf):
    """The key insight, and the bug that prompted it.

    A scan of Marcus Aurelius reports producer "Recoded by LuraDocument", has
    zero run-on words and 1225 clean-looking chars per page — and was
    misclassified as typeset until coverage became the deciding test. Text
    sitting on a photograph of paper can only have come from someone's OCR.
    """
    info = analyze(make_scan_pdf("scan.pdf", pages=4, text_layer=True))

    assert info.text_verdict == "bad_ocr"
    assert info.recommend_ocr is True
    assert info.scanned_ratio >= 0.6
    assert "full-page image" in info.text_reason


def test_scan_without_a_text_layer_needs_ocr(make_scan_pdf):
    info = analyze(make_scan_pdf("scan.pdf", pages=4))

    assert info.text_verdict == "no_text"
    assert info.recommend_ocr is True


def test_typeset_document_skips_ocr(make_pdf):
    """Extraction takes seconds; OCRing it would waste hours for nothing."""
    info = analyze(make_pdf("typeset.pdf", pages=4))

    assert info.text_verdict == "born_digital"
    assert info.recommend_ocr is False
    assert info.scanned_ratio == 0.0


def test_ocr_producer_alone_is_enough_to_distrust_the_text(make_pdf):
    """A weaker signal than coverage, but still worth acting on."""
    doc = pymupdf.open(make_pdf("typeset.pdf", pages=3))
    doc.set_metadata({"producer": "ABBYY FineReader 12"})
    path = doc.name.replace(".pdf", "-ocr.pdf")
    doc.save(path)
    doc.close()

    info = analyze(path)
    assert info.text_verdict == "bad_ocr"
    assert info.recommend_ocr is True
    assert "OCR software" in info.text_reason


def test_no_text_is_decided_before_coverage(make_scan_pdf):
    """Order matters: an image-only page reports no_text, not bad_ocr.

    Both are "needs OCR", so the recommendation is the same either way — but the
    reason shown to the user should say there is no text, not that its text is
    untrustworthy.
    """
    info = analyze(make_scan_pdf("scan.pdf", pages=3))
    assert info.text_verdict == "no_text"
    assert "chars/page" in info.text_reason


# --------------------------------------------------------------------------
# spreads
# --------------------------------------------------------------------------
def test_spread_document_is_recommended_for_splitting(make_pdf):
    info = analyze(make_pdf("spread.pdf", pages=4, spread=True))

    assert info.is_spread is True
    assert info.recommend_split is True
    assert info.output_pages == info.pages * 2


def test_portrait_document_is_not_split(make_pdf):
    """Splitting this would cut every page in half."""
    info = analyze(make_pdf("single.pdf", pages=4))

    assert info.is_spread is False
    assert info.recommend_split is False
    assert info.output_pages == info.pages
    assert "portrait" in info.spread_reason


# --------------------------------------------------------------------------
# the estimate the GUI shows
# --------------------------------------------------------------------------
def test_estimate_uses_the_page_count_after_splitting():
    ocr = Analysis(path=None, pages=10, recommend_split=True, recommend_ocr=True)
    assert ocr.output_pages == 20
    assert ocr.estimated_seconds == 20 * OCR_SECONDS_PER_PAGE


def test_extraction_is_estimated_in_seconds_not_hours():
    extract = Analysis(path=None, pages=100, recommend_ocr=False)
    assert extract.estimated_seconds == 100 * EXTRACT_SECONDS_PER_PAGE
    assert "second" in extract.estimate_human()


def test_estimate_is_phrased_by_magnitude():
    assert "second" in Analysis(path=None, pages=10, recommend_ocr=False).estimate_human()
    assert "minute" in Analysis(path=None, pages=20, recommend_ocr=True).estimate_human()
    assert "hour" in Analysis(path=None, pages=500, recommend_ocr=True).estimate_human()


def test_empty_document_guard():
    """The zero-page guard in detect_spreads, tested where it can be reached.

    A zero-page file cannot arrive through analyze(): PyMuPDF refuses to save
    one ("cannot save with zero pages"), so the guard is purely defensive. It is
    still worth pinning, because the division it protects against
    (spreads / len(idxs)) would otherwise be a ZeroDivisionError.
    """
    from bt.split_spreads import detect_spreads

    with pymupdf.open() as empty:  # in memory; never saved
        result = detect_spreads(empty)

    assert result["is_spread"] is False
    assert result["checked"] == 0
