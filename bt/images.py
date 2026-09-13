"""Figures marker pulls out of the page, and the links that point at them.

marker returns extracted pictures as ``{name: PIL image}`` alongside the
Markdown, which refers to them as ``![](_page_3_Picture_0.jpeg)``. Two things
have to be true for those links to still work in the finished book:

**The files must not collide.** A run converts the document in chunks, and the
names are invented per conversion. Whether marker numbers pages from the start
of the document or from the start of the chunk is an implementation detail of
its page ids -- and if it is ever the latter, chunk two's
``_page_0_Picture_0.jpeg`` silently overwrites chunk one's, leaving the right
caption above the wrong plate. Rather than depend on the answer, every chunk's
images are namespaced with the chunk's own page range, which is collision-proof
either way.

**The path must be right from the finished file.** Images live in
``<out_dir>/images/`` and are linked as ``images/...``, which resolves from
``raw.md`` and the final ``.md``. Chunk files sit one level down in
``chunks/``, so the link is one level off when a chunk is read on its own --
they are intermediates, and correctness of the deliverable wins.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

IMAGE_DIR_NAME = "images"

# ![alt](path) -- alt text may be empty, and often is.
MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

# Everything else is replaced. This is what makes a name from marker unable to
# reach outside the image directory, whatever it contains.
UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

JPEG_SUFFIXES = {".jpg", ".jpeg"}


def safe_name(name: str, prefix: str) -> str:
    """``_page_3_Picture_0.jpeg`` -> ``0000-0019_page_3_Picture_0.jpeg``.

    Only the basename is kept and only ``[A-Za-z0-9._-]`` survives it, so a
    separator or a ``..`` in marker's key cannot escape the directory. The name
    is mangled rather than rejected on purpose: a chunk is an hour of OCR, and
    losing it to a surprising filename would cost far more than an ugly one.
    """
    base = Path(name).name.lstrip("._") or "image"
    return f"{prefix}_{UNSAFE.sub('_', base)}"


def save_chunk_images(
    images: dict[str, Any], image_dir: Path, prefix: str
) -> dict[str, str]:
    """Write one chunk's images; return ``{marker name: link path}``.

    Written temp-then-rename for the same reason chunk files are: the chunk
    ``.md`` is the record that its pages are done, so a half-written figure
    that survived a kill would never be rewritten -- resume skips the chunk.
    """
    if not images:
        return {}

    image_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for name, image in images.items():
        target = image_dir / safe_name(name, prefix)
        # The temp name keeps the real extension last: PIL picks the format
        # from it, and a bare ".partial" would leave it with nothing to go on.
        tmp = target.with_name(f"{target.stem}.partial{target.suffix}")
        _write(image, tmp)
        tmp.replace(target)
        mapping[name] = f"{image_dir.name}/{target.name}"
    return mapping


def _write(image: Any, path: Path) -> None:
    """Save a PIL image, or bytes, without importing PIL here.

    JPEG cannot hold an alpha channel, and PIL raises rather than dropping it.
    A figure with transparency is not worth failing a chunk over, so it is
    flattened to RGB instead.
    """
    if isinstance(image, (bytes, bytearray)):
        path.write_bytes(image)
        return
    if path.suffix.lower() in JPEG_SUFFIXES:
        mode = getattr(image, "mode", "RGB")
        if mode not in ("RGB", "L"):
            image = image.convert("RGB")
    image.save(path)


def rewrite_references(text: str, mapping: dict[str, str]) -> str:
    """Repoint ``![](name)`` at the saved file.

    A reference with no entry in the mapping is left exactly as it was. Making
    one up would turn a link marker never backed with an image into a link that
    merely *looks* right, which is precisely what ``verify.check_images`` is
    there to notice.
    """
    if not mapping:
        return text

    def repoint(m: re.Match) -> str:
        target = mapping.get(m.group(2))
        return m.group(0) if target is None else f"![{m.group(1)}]({target})"

    return MD_IMAGE.sub(repoint, text)


def stage_chunk(text: str, images: dict[str, Any], image_dir: Path, prefix: str) -> str:
    """Save a chunk's images and return its Markdown pointing at them."""
    return rewrite_references(text, save_chunk_images(images, image_dir, prefix))


def count_references(text: str) -> int:
    return len(MD_IMAGE.findall(text))
