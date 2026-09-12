"""Which documents exist, how far each has got, and which is next.

State is **derived from disk** wherever possible rather than tracked in a file.
Chunk files already record exactly what has been transcribed, so a separate
progress record could only drift away from the truth. The queue file holds just
the things disk cannot answer: whether to skip a document, how many attempts
have failed to make progress, and a cached analysis so a survey does not re-read
every PDF.

Completion is judged by chunk count, never by output existing:
``bt/run.py`` writes ``raw.md`` and a final ``.md`` even when the disk guard cut
a run short, so a finished-looking file says nothing about whether the book is
actually finished.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bt import jobs

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output"
QUEUE_FILE = OUTPUT_ROOT / "queue.json"

DEFAULT_CHUNK_SIZE = 10
# Attempts that complete without advancing a single chunk. Three consecutive
# no-progress runs means something is wrong with this document specifically, so
# move on rather than blocking the queue forever.
MAX_ATTEMPTS = 3


@dataclass
class DocState:
    pdf: Path
    out_dir: Path
    chunks_done: int = 0
    chunks_total: int = 0
    pages: int = 0
    split: bool = True
    ocr: bool = True
    skip: bool = False
    attempts: int = 0
    last_error: str = ""

    @property
    def key(self) -> str:
        """Stable identifier: path relative to the project."""
        try:
            return str(self.pdf.relative_to(ROOT))
        except ValueError:
            return str(self.pdf)

    @property
    def done(self) -> bool:
        return self.chunks_total > 0 and self.chunks_done >= self.chunks_total

    @property
    def started(self) -> bool:
        return self.chunks_done > 0

    @property
    def status(self) -> str:
        if self.skip:
            return "skipped"
        if self.done:
            return "done"
        if self.attempts >= MAX_ATTEMPTS:
            return "stalled"
        return "in progress" if self.started else "pending"

    @property
    def fraction(self) -> float:
        if self.chunks_total <= 0:
            return 0.0
        return min(1.0, self.chunks_done / self.chunks_total)


def discover() -> list[Path]:
    """PDFs in the project root and uploads/ -- the same list the GUI shows."""
    found = list(ROOT.glob("*.pdf")) + list((ROOT / "uploads").glob("*.pdf"))
    return sorted(
        (p for p in found if not p.name.startswith(".")), key=lambda p: p.name.lower()
    )


def chunks_in(folder: Path) -> int:
    return len(list((folder / "chunks").glob("[0-9]*-[0-9]*.md")))


def folder_belongs_to(folder: Path, pdf: Path) -> bool:
    """Whether a folder's chunks came from this PDF.

    Chunk files are named by page number alone, so they cannot say which
    document produced them. Claiming a folder because it merely *has* chunks
    would let one book resume onto another's output. The lock file records the
    source PDF, so that is what decides ownership.
    """
    spec = jobs.status(folder).spec
    if not spec or not spec.pdf:
        return False
    try:
        return Path(spec.pdf).resolve() == pdf.resolve()
    except OSError:
        return False


def resolve_out_dir(pdf: Path) -> Path:
    """Where this document's work lives.

    New documents get their own folder. Earlier CLI runs wrote straight into
    output/, so that is adopted too -- but only when the lock proves it belongs
    to this PDF.
    """
    per_doc = OUTPUT_ROOT / pdf.stem
    if chunks_in(per_doc):
        return per_doc
    if chunks_in(OUTPUT_ROOT) and folder_belongs_to(OUTPUT_ROOT, pdf):
        return OUTPUT_ROOT
    return per_doc


# --------------------------------------------------------------------------
# persisted overlay
# --------------------------------------------------------------------------
def load_overlay() -> dict:
    if not QUEUE_FILE.exists():
        return {}
    try:
        return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def save_overlay(data: dict) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(QUEUE_FILE)


def update_doc(key: str, **fields) -> dict:
    data = load_overlay()
    entry = data.setdefault(key, {})
    entry.update(fields)
    save_overlay(data)
    return data


def set_skip(key: str, skip: bool) -> None:
    update_doc(key, skip=skip)


def record_attempt(key: str, progressed: bool, error: str = "") -> None:
    """A run that advanced any chunk resets the counter.

    Progress is measured in chunks, not exit status: the disk guard stops runs
    routinely here, and a run that transcribed two chunks before stopping is a
    success, not a failure.
    """
    data = load_overlay()
    entry = data.setdefault(key, {})
    entry["attempts"] = 0 if progressed else int(entry.get("attempts", 0)) + 1
    entry["last_error"] = "" if progressed else error
    save_overlay(data)


# --------------------------------------------------------------------------
# survey
# --------------------------------------------------------------------------
def _cached_analysis(pdf: Path, overlay: dict) -> dict:
    """analyze() samples pages, so cache it against the file's identity."""
    key = str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf)
    entry = overlay.get(key, {})
    cached = entry.get("analysis")
    stat = pdf.stat()
    if cached and cached.get("mtime") == stat.st_mtime and cached.get("size") == stat.st_size:
        return cached

    from bt.analyze import analyze

    info = analyze(pdf)
    fresh = {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "pages": info.pages,
        "output_pages": info.output_pages,
        "split": info.recommend_split,
        "ocr": info.recommend_ocr,
    }
    update_doc(key, analysis=fresh)
    return fresh


def survey(chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[DocState]:
    """Current state of every known document."""
    overlay = load_overlay()
    states: list[DocState] = []
    for pdf in discover():
        info = _cached_analysis(pdf, overlay)
        out_dir = resolve_out_dir(pdf)
        key = str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf)
        entry = overlay.get(key, {})
        total_pages = int(info.get("output_pages") or 0)
        states.append(
            DocState(
                pdf=pdf,
                out_dir=out_dir,
                chunks_done=chunks_in(out_dir),
                chunks_total=-(-total_pages // chunk_size) if total_pages else 0,
                pages=total_pages,
                split=bool(info.get("split", True)),
                ocr=bool(info.get("ocr", True)),
                skip=bool(entry.get("skip", False)),
                attempts=int(entry.get("attempts", 0)),
                last_error=str(entry.get("last_error", "")),
            )
        )
    return states


def next_pending(chunk_size: int = DEFAULT_CHUNK_SIZE) -> DocState | None:
    """The document the worker should pick up next.

    Documents already part-done come first: finishing a book releases its disk
    and gives a usable result sooner than spreading effort across five.
    """
    candidates = [
        s
        for s in survey(chunk_size)
        if not s.skip and not s.done and s.attempts < MAX_ATTEMPTS
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (not s.started, s.pdf.name.lower()))
    return candidates[0]


def spec_for(state: DocState, chunk_size: int = DEFAULT_CHUNK_SIZE) -> jobs.JobSpec:
    return jobs.JobSpec(
        pdf=str(state.pdf),
        out_dir=str(state.out_dir),
        split=state.split,
        ocr=state.ocr,
        chunk_size=chunk_size,
        total_pages=state.pages,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Show the transcription queue.")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = ap.parse_args(argv)

    states = survey(args.chunk_size)
    width = max((len(s.key) for s in states), default=10)
    for s in states:
        bar = f"{s.chunks_done}/{s.chunks_total}" if s.chunks_total else "?"
        print(f"  {s.key:<{width}}  {s.status:<12} {bar:>9}  {s.fraction:5.0%}  -> {s.out_dir.name}")
    nxt = next_pending(args.chunk_size)
    print(f"\nnext: {nxt.key if nxt else '(nothing pending)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
