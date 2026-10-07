"""Gutter detection and spread splitting.

``find_gutter`` is pure -- it takes an ink profile as a numpy array -- so the
trickiest logic in the pipeline is also the cheapest to test.
"""

from __future__ import annotations

import numpy as np
import pymupdf

from bt.split_spreads import (
    SPREAD_MIN_BAND,
    _ink_profile,
    detect_spreads,
    find_gutter,
    split_document,
)


def profile(*runs: tuple[int, int]) -> np.ndarray:
    """Build an ink profile from (value, length) pairs."""
    return np.concatenate([np.full(n, v) for v, n in runs])


def test_find_gutter_picks_the_widest_blank_run():
    """REGRESSION: an early version used argmin.

    The whole gutter reads zero, so argmin returns whichever zero it meets
    first and drifts to the edge of the search band. Here a narrow 4-column gap
    sits before a wide 40-column one; the wide one is the gutter.
    """
    ink = profile((5, 100), (0, 4), (5, 40), (0, 40), (5, 116))
    position, band = find_gutter(ink)
    centre = (144 + 40 / 2) / 300
    assert abs(position - centre) < 0.01
    assert band > SPREAD_MIN_BAND


def test_find_gutter_falls_back_to_midpoint_without_a_gap():
    position, band = find_gutter(np.full(300, 7))
    assert position == 0.5
    assert band == 0.0, "no blank run means no confidence in a gutter"


def test_find_gutter_ignores_wide_outer_margins():
    """Only the central band is searched, so blank margins cannot win."""
    ink = profile((0, 120), (9, 40), (0, 20), (9, 40), (0, 80))
    position, _ = find_gutter(ink)
    assert 0.35 < position < 0.65


def test_find_gutter_tolerates_a_speck_in_the_gutter():
    """A scan's gutter is rarely *exactly* empty: dust and toner specks land
    there. Requiring zero ink let one faint speck split the gutter in two and
    halve its measured width. Blank means negligible next to the text.
    """
    ink = profile((900, 105), (0, 20), (10, 1), (0, 20), (900, 154))
    position, band = find_gutter(ink)
    assert abs(position - (105 + 41 / 2) / 300) < 0.01
    assert abs(band - 41 / 300) < 0.005


def test_find_gutter_bridges_a_binding_shadow():
    """A flatbed scan leaves a dark line at the fold (visible in the Boyce
    preview). It split the gutter into two half-width runs, so the cut sat
    against one page's text and the band could fall below SPREAD_MIN_BAND,
    making ``detect_spreads`` reject a genuine spread.
    """
    ink = profile((900, 110), (0, 15), (5000, 2), (0, 15), (900, 158))
    position, band = find_gutter(ink)
    assert abs(position - (110 + 32 / 2) / 300) < 0.01
    assert abs(band - 32 / 300) < 0.005


def test_find_gutter_does_not_bridge_a_text_column():
    """Only a *narrow* dark strip is a fold; a block of text is a page."""
    ink = profile((900, 100), (0, 6), (900, 20), (0, 30), (900, 144))
    position, band = find_gutter(ink)
    assert abs(position - (126 + 30 / 2) / 300) < 0.01
    assert abs(band - 30 / 300) < 0.005


def test_bitonal_scan_is_cut_in_the_gutter(make_bitonal_spread):
    """REGRESSION: Boyce, *A History of Zoroastrianism* (1975), page 90.

    The GUI preview cut that spread at 60.6% of width, reporting a "blank band"
    5.8% wide -- in the middle of the right-hand page's text. The probe counted
    pixels darker than 128 at 36 DPI, but a 300 DPI bitonal scan's thin strokes
    average out to light grey at that resolution, so body text registered as no
    ink at all and the "widest blank run" could fall anywhere. Ink must be
    measured as summed darkness, which downsampling preserves.
    """
    path, (left_end, right_start) = make_bitonal_spread()
    with pymupdf.open(path) as doc:
        ink = _ink_profile(doc[0])
    position, band = find_gutter(ink)
    assert left_end <= position <= right_start
    assert band >= SPREAD_MIN_BAND, "the binding shadow must not halve the gutter"


def test_blank_half_beside_a_binding_shadow_is_dropped(make_bitonal_spread, tmp_path):
    """Summed darkness counts the fold shadow as ink -- about a tenth of a
    text page's, far above BLANK_INK_FRACTION. It must still land with the
    text half: on a spread with one blank page the widest blank run is that
    page, so the cut falls beside the shadow, not through it. If gutter
    selection ever changes, this is what keeps an empty page out of OCR.
    """
    path, _ = make_bitonal_spread(blank_left=True)
    records = split_document(path, tmp_path / "pages.pdf", None)
    assert [r.half for r in records] == ["right"]


def test_bitonal_scan_body_text_registers_as_ink(make_bitonal_spread):
    """The root cause, pinned directly: every column under body text has ink."""
    path, (left_end, _) = make_bitonal_spread()
    with pymupdf.open(path) as doc:
        ink = _ink_profile(doc[0])
    text = ink[int(len(ink) * 0.1) : int(len(ink) * (left_end - 0.02))]
    assert (text > 0).all()


def test_detect_spreads_true_for_landscape_with_a_gutter(make_pdf):
    with pymupdf.open(make_pdf("spread.pdf", pages=4, spread=True)) as doc:
        result = detect_spreads(doc)
    assert result["is_spread"] is True
    assert result["median_band"] >= SPREAD_MIN_BAND


def test_detect_spreads_false_for_portrait_pages(make_pdf):
    """Splitting a single-page PDF would cut every page in half."""
    with pymupdf.open(make_pdf("single.pdf", pages=4)) as doc:
        result = detect_spreads(doc)
    assert result["is_spread"] is False
    assert "portrait" in result["reason"]


def test_split_document_doubles_the_page_count(make_pdf, tmp_path):
    src = make_pdf("spread.pdf", pages=3, spread=True)
    out = tmp_path / "pages.pdf"
    records = split_document(src, out, tmp_path / "pagemap.json")

    assert len(records) == 6
    assert [r.half for r in records[:2]] == ["left", "right"]
    with pymupdf.open(out) as doc:
        assert doc.page_count == 6


def test_split_document_drops_blank_halves(make_pdf, tmp_path):
    """REGRESSION: six halves in the source book are blank.

    Without this they become empty pages that still cost a full OCR pass.
    """
    src = make_pdf("spread.pdf", pages=3, spread=True, blank_left_on=(1,))
    records = split_document(src, tmp_path / "pages.pdf", None)

    assert len(records) == 5, "the blank left half should be dropped"
    from_page_1 = [r.half for r in records if r.src_page == 1]
    assert from_page_1 == ["right"]


def test_split_document_writes_a_traceable_pagemap(make_pdf, tmp_path):
    import json

    src = make_pdf("spread.pdf", pages=2, spread=True)
    pagemap = tmp_path / "pagemap.json"
    split_document(src, tmp_path / "pages.pdf", pagemap)

    entries = json.loads(pagemap.read_text())
    assert [e["out_page"] for e in entries] == [0, 1, 2, 3]
    assert {e["src_page"] for e in entries} == {0, 1}


def test_split_output_pages_are_narrower_than_the_spread(make_pdf, tmp_path):
    """Cropping must actually change the visible box, not just copy pages."""
    src = make_pdf("spread.pdf", pages=1, spread=True)
    out = tmp_path / "pages.pdf"
    split_document(src, out, None)

    with pymupdf.open(src) as s, pymupdf.open(out) as o:
        assert o[0].rect.width < s[0].rect.width * 0.75
        assert abs(o[0].rect.height - s[0].rect.height) < 1
