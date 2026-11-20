"""Seed a demo library into an app copy: a 3-page journal-like PDF with a
ruled table, and a finished Markdown so the Result tab has something to show.
Synthetic text only -- real output is book text and is never used here."""

import sys
from pathlib import Path

import pymupdf

root = Path(sys.argv[1])
doc = pymupdf.open()
for p in range(3):
    page = doc.new_page()
    page.insert_text((72, 72), f"Journal of Tests  vol {p}", fontsize=8)
    page.insert_text((72, 110), f"{p + 1}. Introduction", fontsize=16)
    page.insert_textbox((72, 130, 520, 250), "Placeholder prose for layout. " * 12, fontsize=10)
    if p == 1:
        for r in range(4):
            for c in range(3):
                page.insert_text((80 + c * 120, 300 + r * 24), f"r{r}c{c}", fontsize=10)
doc.save(root / "journal.pdf")

out = root / "output" / "journal"
(out / "chunks").mkdir(parents=True)
(out / "journal.md").write_text(
    "<!-- page 0 -->\n\n# Introduction\n\nPlaceholder prose.\n\n<!-- page 1 -->\n\nMore.\n",
    encoding="utf-8",
)
