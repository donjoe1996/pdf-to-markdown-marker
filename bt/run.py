"""CLI entry point: split -> transcribe -> postprocess -> verify.

    uv run bt-transcribe --test          # 3 spreads, for eyeballing quality
    uv run bt-transcribe                 # the whole book
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Source page 8 is book page 1 of the Being and Time scan: dense polytonic
# Greek plus a half-page footnote block. Pages 40-41 are ordinary body spreads.
# Only meaningful for that document; --pages covers any other.
TEST_PAGES = "8,40-41"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe a PDF to Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--pdf", type=Path, required=True, help="source PDF")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument(
        "--test",
        action="store_true",
        help=f"only process source pages {TEST_PAGES} (validation slice)",
    )
    ap.add_argument("--pages", default=None, help="source pages, e.g. '8,40-41'")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--mode", choices=["fast", "balanced"], default="fast")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--no-split",
        action="store_true",
        help="treat each PDF page as one page. Use for any document that is NOT "
        "stored as two-page spreads -- splitting those cuts every page in half",
    )
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="read the existing text layer instead of OCRing (born-digital PDFs "
        "only -- on a scan this reproduces the old OCR's mistakes)",
    )
    ap.add_argument(
        "--split-only", action="store_true", help="stop after stage 1 (no models needed)"
    )
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args(argv)

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-test" if args.test else ""
    split_pdf = out_dir / f"pages{suffix}.pdf"
    raw_md = out_dir / f"raw{suffix}.md"
    final_md = out_dir / f"{args.pdf.stem}{suffix}.md"
    chunk_dir = out_dir / f"chunks{suffix}"

    if not args.skip_preflight:
        from bt import preflight

        print("== Preflight ==")
        checks = preflight.run(require_marker=not args.split_only)
        if not preflight.report(checks):
            print("\nPreflight failed; fix the items above and re-run.")
            return 1
        print()

    # -- Stage 1: split spreads ------------------------------------------
    from bt.split_spreads import parse_range, split_document

    pages = args.pages or (TEST_PAGES if args.test else None)
    page_range = parse_range(pages) if pages else None

    if args.no_split:
        # The document is already one page per page. Hand the source PDF
        # straight to marker rather than rewriting it.
        import pymupdf

        print("== Stage 1: skipped (--no-split) ==")
        with pymupdf.open(args.pdf) as doc:
            total_pages = len(page_range) if page_range else doc.page_count
        ocr_pdf = args.pdf
        print(f"  {total_pages} pages, using the source PDF as-is")
        if args.split_only:
            return 0
    else:
        print("== Stage 1: splitting spreads ==")
        records = split_document(
            args.pdf, split_pdf, out_dir / f"pagemap{suffix}.json", page_range
        )
        total_pages = len(records)
        ocr_pdf = split_pdf
        print(f"  {total_pages} book pages -> {split_pdf}")
        if args.split_only:
            return 0

    # -- Stage 2: marker --------------------------------------------------
    # Models are fetched first so marker's timed server spawns start from cache.
    print("\n== Stage 2a: model warmup ==")
    from bt.warmup import warm_all

    warm_all()

    print("\n== Stage 2b: %s via marker ==" % ("text extraction" if args.no_ocr else "OCR"))
    from bt.transcribe import transcribe

    transcribe(
        ocr_pdf,
        raw_md,
        chunk_dir,
        total_pages=total_pages,
        chunk_size=args.chunk_size,
        mode=args.mode,
        dpi=args.dpi,
        use_llm=args.use_llm,
        resume=not args.no_resume,
        ocr=not args.no_ocr,
    )

    # -- Stage 3: cleanup -------------------------------------------------
    print("\n== Stage 3: post-processing ==")
    from bt.postprocess import process

    cleaned, stats = process(raw_md.read_text(encoding="utf-8"))
    final_md.write_text(cleaned, encoding="utf-8")
    print(stats.render())
    print(f"\nWrote {final_md}")

    # -- Verify -----------------------------------------------------------
    print("\n== Verification ==")
    from bt.verify import run_all

    findings = run_all(cleaned)
    for f in findings:
        print(f"  [{'ok  ' if f.ok else 'FAIL'}] {f.name:<13} {f.detail}")
    return 0 if all(f.ok for f in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
