"""Stage 2, alternative engine: Docling instead of marker.

Why it exists. marker is a page-image reader: on a born-digital journal article
it re-reads pixels the PDF already holds as exact text, and it takes its table
cells from the same guess. Docling instead runs a dedicated *layout* model
(what is a heading, a table, a figure, a caption, a footnote, a running head)
and a dedicated *table-structure* model (TableFormer), and takes the text from
the PDF's own layer wherever that layer is good. For journals that is the right
division of labour: structure from models, characters from the source.

The rest of the pipeline is unchanged because this module returns what marker
returned: Markdown with one ``{N}----`` separator per page, ``N`` the absolute
0-based page index, plus a ``{filename: image}`` dict. ``postprocess``,
``verify``, ``audit`` and ``translate`` therefore cannot tell the engines apart.

Docling is imported inside functions only -- the same rule as marker/torch in
``transcribe.py`` -- so ``bt.analyze`` and ``--split-only`` stay light.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

# marker's separator is "{N}" + 48 dashes; postprocess accepts any 10 or more.
SEPARATOR_DASHES = 48

# Docling leaves this where a picture was. It is swapped for a real link (or
# nothing) page by page, so the token must be one prose cannot contain.
IMAGE_TOKEN = "<!--bt-image-->"


def page_separator(index: int) -> str:
    return f"{{{index}}}" + "-" * SEPARATOR_DASHES


def join_pages(pages: Iterable[tuple[int, str]]) -> str:
    """Markdown in marker's paginated shape: separator, then the page's text.

    Pages are emitted even when empty. A page with no text must still carry its
    anchor, or ``audit`` and ``verify`` align every later page to the wrong PDF
    page -- the same silent shift that once made the stale-layer check a no-op.
    """
    parts = []
    for index, text in pages:
        parts.append(page_separator(index) + "\n\n" + text.strip())
    return "\n\n".join(parts) + "\n"


def picture_name(page_index: int, n: int) -> str:
    """marker's naming (``_page_3_Figure_2.jpeg``), so ``save_images`` and
    ``rewrite_image_links`` need no knowledge of which engine ran."""
    return f"_page_{page_index}_Figure_{n}.jpeg"


def fill_image_links(text: str, names: list[str]) -> str:
    """Replace each image token, in order, with a link to the next picture.

    A token with no picture left (Docling found a figure it could not crop) is
    dropped rather than left as a literal marker in the prose.
    """
    out: list[str] = []
    names_iter = iter(names)
    pieces = text.split(IMAGE_TOKEN)
    for i, piece in enumerate(pieces):
        out.append(piece)
        if i < len(pieces) - 1:
            name = next(names_iter, None)
            if name:
                out.append(f"![]({name})")
    return "".join(out)


def render_pages(
    doc,
    first: int,
    last: int,
    pictures_on: Callable[[int], list] | None = None,
) -> tuple[str, dict]:
    """Paginated Markdown for source pages ``first``..``last`` (0-based).

    ``doc`` is a ``DoclingDocument``. Docling numbers pages from 1, so every
    index crosses that boundary exactly once, here.

    ``pictures_on(page_index)`` returns that page's picture images in reading
    order; ``None`` means figures are not wanted, and their tokens vanish.
    """
    pages: list[tuple[int, str]] = []
    images: dict = {}
    for index in range(first, last + 1):
        text = doc.export_to_markdown(
            page_no=index + 1,
            image_placeholder=IMAGE_TOKEN,
            # Docling's default wraps long lines at no width; keep that, and keep
            # underscores unescaped so snake_case identifiers survive intact.
            escape_underscores=False,
        )
        names: list[str] = []
        if pictures_on is not None:
            for n, image in enumerate(pictures_on(index), 1):
                if image is None:
                    continue
                name = picture_name(index, n)
                images[name] = image
                names.append(name)
        pages.append((index, fill_image_links(text, names)))
    return join_pages(pages), images


def make_converter(ocr: bool = False, extract_images: bool = False, dpi: int = 300):
    """A reusable ``DocumentConverter``.

    ``ocr=False`` (the default for journals) reads the PDF's own text layer.
    ``ocr=True`` forces a full-page OCR, for scans -- whose embedded layer is
    someone else's OCR and is not to be trusted, as everywhere in this project.
    Table-structure recognition is always on: it is the reason to use Docling.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    opts.do_ocr = ocr
    if ocr:
        # A scan's embedded layer is bad OCR; reading it would reproduce its
        # mistakes. Replace it outright, not just where Docling sees no text.
        opts.ocr_options.force_full_page_ocr = True
    if extract_images:
        opts.generate_picture_images = True
        # Docling renders at 72 dpi * scale; ~2 is sharp without huge files.
        opts.images_scale = max(1.0, min(dpi / 150, 3.0))
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_range(
    converter, pdf_path: Path, first: int, last: int, extract_images: bool = False
) -> tuple[str, dict]:
    """Markdown and images for source pages ``first``..``last`` (0-based)."""
    # page_range is 1-based and inclusive.
    result = converter.convert(str(pdf_path), page_range=(first + 1, last + 1))
    doc = result.document

    def pictures_on(index: int) -> list:
        found = []
        for pic in doc.pictures:
            if any(p.page_no == index + 1 for p in pic.prov[:1]):
                found.append(pic.get_image(doc))
        return found

    return render_pages(
        doc, first, last, pictures_on if extract_images else None
    )


def warm() -> None:
    """Download Docling's layout and table models before the first chunk.

    Same reason as ``bt.warmup``: a first conversion that stalls on a download
    looks like a hung job.
    """
    from docling.utils.model_downloader import download_models

    download_models(progress=False)
