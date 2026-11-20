"""The Docling engine's plumbing.

Docling's models cannot run in the suite (they are hundreds of MB and need a
download), so these build a real ``DoclingDocument`` by hand -- the object the
models would have produced -- and test everything this repo adds around it:
pagination, anchors, image links and the wiring that selects the engine.
Whether Docling *reads a journal well* is checked on real output with
``bt.verify`` / ``bt.audit``, not here.
"""

from __future__ import annotations

import pytest
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)
from docling_core.types.doc.document import TableCell, TableData

from bt import docling_engine as de
from bt.jobs import JobSpec
from bt.postprocess import split_pages


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1)
    )


@pytest.fixture
def doc() -> DoclingDocument:
    d = DoclingDocument(name="article")
    d.add_heading("Introduction", level=1, prov=_prov(1))
    d.add_text(label=DocItemLabel.TEXT, text="first page body", prov=_prov(1))
    cells = [
        TableCell(
            text=t,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate([["plot", "tC/ha"], ["A", "120"]])
        for c, t in enumerate(row)
    ]
    d.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=2), prov=_prov(2))
    # page 3 is deliberately empty of text
    d.add_text(label=DocItemLabel.TEXT, text="fourth page body", prov=_prov(4))
    return d


def test_pages_use_markers_separator_and_absolute_zero_based_index(doc):
    """Downstream aligns pages by ``{N}`` -- 0-based, absolute -- so a chunk
    for source pages 1-3 must say 1, 2, 3, not 0, 1, 2 (Docling counts from 1)."""
    text, _ = de.render_pages(doc, 0, 3)
    pages = dict(split_pages(text))
    assert set(pages) == {0, 1, 2, 3}
    assert "first page body" in pages[0]
    assert "tC/ha" in pages[1]  # Docling page 2 -> index 1
    assert "fourth page body" in pages[3]


def test_a_page_without_text_still_gets_its_anchor(doc):
    """An omitted empty page shifts every later page onto the wrong PDF page,
    which is how verify's stale-layer check once became a no-op."""
    text, _ = de.render_pages(doc, 0, 3)
    assert de.page_separator(2) in text
    assert dict(split_pages(text))[2].strip() == ""


def test_tables_come_out_as_markdown_tables(doc):
    text, _ = de.render_pages(doc, 1, 1)
    assert "| plot" in text and "120 |" in text and "|---" in text


def test_the_separator_matches_what_postprocess_parses():
    from bt.postprocess import PAGE_SEPARATOR

    assert PAGE_SEPARATOR.match(de.page_separator(17)).group(1) == "17"


def test_image_tokens_become_links_in_order_and_extras_vanish():
    out = de.fill_image_links(
        f"a {de.IMAGE_TOKEN} b {de.IMAGE_TOKEN} c {de.IMAGE_TOKEN}", ["x.jpeg", "y.jpeg"]
    )
    assert out == "a ![](x.jpeg) b ![](y.jpeg) c "
    assert de.IMAGE_TOKEN not in out


def test_figures_are_dropped_when_not_requested():
    d = DoclingDocument(name="f")
    d.add_picture(prov=_prov(1))
    text, images = de.render_pages(d, 0, 0)
    assert images == {} and de.IMAGE_TOKEN not in text and "![" not in text


def test_picture_names_follow_markers_convention():
    """save_images/rewrite_image_links key on this name; it is shared on purpose."""
    assert de.picture_name(3, 2) == "_page_3_Figure_2.jpeg"
    d = DoclingDocument(name="f")
    d.add_picture(prov=_prov(1))
    text, images = de.render_pages(d, 0, 0, pictures_on=lambda i: ["IMG"])
    assert images == {"_page_0_Figure_1.jpeg": "IMG"}
    assert "![](_page_0_Figure_1.jpeg)" in text


def test_jobspec_passes_engine_only_when_not_default():
    base = JobSpec(pdf="a.pdf", out_dir="o")
    assert "--engine" not in base.command()
    cmd = JobSpec(pdf="a.pdf", out_dir="o", engine="docling").command()
    assert cmd[cmd.index("--engine") + 1] == "docling"


def test_preflight_docling_needs_no_llama_server():
    """Docling runs in-process; demanding llama-server would block a run that
    could proceed, and the check is the one thing that can fail it."""
    from bt import preflight

    names = [c.name for c in preflight.run(require_marker=True, engine="docling")]
    assert "docling" in names and "marker-pdf" not in names
    assert not any("llama" in n for n in names)
    marker_names = [c.name for c in preflight.run(require_marker=True)]
    assert "marker-pdf" in marker_names
