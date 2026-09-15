"""Optional stage 4: translate finished Markdown into another language.

Not part of ``bt.run``. A book is translated on request, after the OCR is done
and has been looked at -- translating a bad transcription only multiplies the
damage.

Two decisions shape this module:

**The page anchors never reach the model.** ``postprocess`` writes
``<!-- page N -->`` between pages, and ``verify.check_not_embedded_layer``
aligns output to the PDF by those numbers. A model told to "preserve" them
would drop one eventually, and nothing would look wrong: the translation would
still read perfectly while the page alignment quietly lost a page. So the
anchors are stripped before the request and re-emitted here afterwards, exactly
as ``postprocess.process()`` does. One page in, one page out, by construction.

**Chunks are the progress record**, as in ``bt.transcribe``. A book is hundreds
of requests over hours; each chunk is written atomically and skipped on resume,
so a rate limit at page 300 costs the current chunk and nothing else.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from bt.postprocess import PAGE_ANCHOR_FMT, PAGE_MARK, split_pages
from bt.transcribe import chunk_path, concatenate

DEFAULT_TARGET = "English"
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_PAGES_PER_CHUNK = 10

# One page of dense body text is ~1.5k output tokens; 16k leaves room for a
# heavy footnote page without risking the SDK's non-streaming HTTP timeout.
# Hitting this ceiling is treated as an error, not a result -- see _check().
MAX_TOKENS = 16000

# Translation is high-volume, low-judgement work: the same instruction applied
# to page after page. Effort buys nothing here and is charged per page.
DEFAULT_EFFORT = "low"

# Footnote ids ([^p13-1]) and image targets are cross-references, not prose. If
# the model rewrites one the document still renders -- it just points at the
# wrong thing, which is the kind of damage that surfaces months later.
FOOTNOTE_ID = re.compile(r"\[\^([^\]]{1,32})\]")
IMAGE_TARGET = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# A model asked for Markdown tends to hand back Markdown in a fence.
FENCE = re.compile(r"\A\s*```[a-zA-Z]*\n(.*)\n```\s*\Z", re.DOTALL)


class Translator(Protocol):
    """Anything that turns one page of Markdown into another language.

    Deliberately this small: the risky part of this stage is the document
    surgery around the call, not the call, so the tests substitute a plain
    function and never touch the network.
    """

    def __call__(self, text: str, target: str) -> str: ...


class TranslationError(RuntimeError):
    """A response that must not be written to disk."""


def system_prompt(target: str) -> str:
    return f"""You are translating one page of a scanned book into {target}.

The input is Markdown produced by OCR. Return only the translated Markdown for
that page: no preamble, no commentary, no code fence.

Rules:
- Preserve the Markdown structure exactly -- headings, emphasis, lists, tables,
  block quotes, and the blank lines between paragraphs.
- Copy footnote markers and their definitions through unchanged, ids included
  (for example [^p13-1]). They are cross-references; renumbering one breaks it.
- Copy image links through unchanged, path and all: ![](images/whatever.jpeg).
- Leave numbers, units, dates, measurements, bibliographic citations, author
  names, place names and scientific binomials (*Manilkara zapota*) as they are.
- Translate the running prose, headings, table cells, figure captions and
  footnote text.
- If a passage is already in {target}, copy it through unchanged.
- Do not summarise, expand, correct or comment. Where the OCR is garbled,
  translate what is there rather than guessing at what was meant.
"""


@dataclass
class Stats:
    pages: int = 0
    pages_translated: int = 0
    pages_blank: int = 0
    chunks_done: int = 0
    chunks_pending: int = 0
    chars_in: int = 0
    chars_out: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    markup_drift: list[int] = field(default_factory=list)
    stopped_early: bool = False
    error: str = ""

    def render(self) -> str:
        lines = [
            f"pages in document   : {self.pages}",
            f"pages translated    : {self.pages_translated}",
            f"blank pages skipped : {self.pages_blank}",
            f"characters          : {self.chars_in:,} -> {self.chars_out:,}",
        ]
        if self.input_tokens or self.output_tokens:
            lines.append(
                f"tokens              : {self.input_tokens:,} in, "
                f"{self.output_tokens:,} out"
            )
        if self.markup_drift:
            shown = ", ".join(str(n) for n in self.markup_drift[:10])
            more = " ..." if len(self.markup_drift) > 10 else ""
            lines.append(
                f"markup drift        : {len(self.markup_drift)} pages "
                f"(footnote id or image link changed: {shown}{more})"
            )
        else:
            lines.append("markup drift        : none")
        if self.stopped_early:
            lines.append(f"STOPPED EARLY       : {self.error}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the response, before it is trusted
# --------------------------------------------------------------------------
def strip_fence(text: str) -> str:
    m = FENCE.match(text)
    return m.group(1) if m else text.strip()


def markup_signature(text: str) -> tuple[list[str], list[str]]:
    """The cross-references a page carries, in a form that survives translation."""
    return sorted(FOOTNOTE_ID.findall(text)), sorted(IMAGE_TARGET.findall(text))


def translate_page(
    body: str, translator: Translator, target: str, page_no: int, stats: Stats
) -> str:
    """One page through the model, with its markup checked on the way back."""
    if not body.strip():
        stats.pages_blank += 1
        return ""

    out = strip_fence(translator(body, target))
    if not out.strip():
        raise TranslationError(f"page {page_no}: empty translation")

    if markup_signature(body) != markup_signature(out):
        # Not fatal: a page may legitimately be renumbered by a model that
        # decided a footnote belonged elsewhere. But it is never *intended*,
        # so it is counted and reported rather than silently accepted.
        stats.markup_drift.append(page_no)

    stats.pages_translated += 1
    stats.chars_in += len(body)
    stats.chars_out += len(out)
    return out


def render_pages(pages: list[tuple[int, str]], anchors: bool) -> str:
    """Re-emit the page anchors the model never saw."""
    if not anchors:
        return "\n\n".join(body for _, body in pages if body.strip())
    return "\n\n".join(
        f"{PAGE_ANCHOR_FMT.format(n=n)}\n\n{body}".rstrip()
        for n, body in pages
        if body.strip()
    )


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def translate_document(
    src_md: Path,
    out_md: Path,
    chunk_dir: Path,
    translator: Translator,
    target: str = DEFAULT_TARGET,
    pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
    resume: bool = True,
) -> tuple[Path, Stats]:
    """Translate ``src_md`` page by page, resumably, into ``out_md``."""
    text = Path(src_md).read_text(encoding="utf-8")
    anchors = bool(PAGE_MARK.search(text))
    pages = split_pages(text)

    stats = Stats(pages=len(pages))
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Bounds are indices into the page list, not page numbers: a page number can
    # repeat (a preamble shares page 0 with the first page proper), and an
    # overlapping bound would splice the same text in twice.
    bounds = [
        (i, min(i + pages_per_chunk - 1, len(pages) - 1))
        for i in range(0, len(pages), pages_per_chunk)
    ]
    pending = [b for b in bounds if not (resume and chunk_path(chunk_dir, *b).exists())]
    stats.chunks_done = len(bounds) - len(pending)
    stats.chunks_pending = len(pending)
    print(
        f"{len(pages)} pages in {len(bounds)} chunks; "
        f"{stats.chunks_done} already done, {len(pending)} to run."
    )

    done_bounds = [b for b in bounds if b not in pending]
    for n, (first, last) in enumerate(pending, 1):
        started = time.time()
        try:
            translated = [
                (page_no, translate_page(body, translator, target, page_no, stats))
                for page_no, body in pages[first : last + 1]
            ]
        except Exception as exc:  # noqa: BLE001 -- any failure stops the run
            # Stop rather than skip. A skipped chunk would leave a hole that
            # resume cannot see, and the book would look finished with a
            # missing stretch in the middle.
            stats.stopped_early = True
            stats.error = f"{type(exc).__name__}: {exc}"
            print(
                f"\n  STOPPING at chunk {first}-{last}: {stats.error}\n"
                f"  {n - 1} of {len(pending)} chunks done and saved. "
                "Re-run the same command to resume."
            )
            break

        # Temp file then rename, as the OCR chunks do: a chunk killed mid-write
        # would otherwise look complete on resume and silently drop pages.
        target_path = chunk_path(chunk_dir, first, last)
        tmp = target_path.with_suffix(".partial")
        tmp.write_text(render_pages(translated, anchors), encoding="utf-8")
        tmp.replace(target_path)
        done_bounds.append((first, last))
        stats.chunks_done += 1

        elapsed = time.time() - started
        count = last - first + 1
        print(
            f"  [{n}/{len(pending)}] pages {first}-{last} -> {target_path.name} "
            f"({elapsed:.0f}s, {elapsed / count:.1f}s/page)"
        )

    out_md = Path(out_md)
    concatenate(chunk_dir, out_md, sorted(done_bounds))
    return out_md, stats


# --------------------------------------------------------------------------
# the Claude backend
# --------------------------------------------------------------------------
class ClaudeTranslator:
    """Translate a page with the Claude API.

    Credentials are resolved by the SDK (``ANTHROPIC_API_KEY``, or a profile
    from ``ant auth login``) -- nothing is read or stored here.

    Retries are the SDK's: it already backs off on 429 and 5xx, and a long
    unattended run wants more of them than the default two.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = 8,
    ) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, text: str, target: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt(target),
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": text}],
        )

        # Check why it stopped before reading what it said. A truncated page
        # reads as a complete page -- it just ends early, mid-sentence, and
        # nothing downstream can tell.
        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", "") or ""
            raise TranslationError(f"model declined this page: {detail}")
        if response.stop_reason == "max_tokens":
            raise TranslationError(
                f"page exceeded max_tokens ({self.max_tokens}); the translation "
                "would be truncated mid-sentence. Raise --max-tokens."
            )

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        return "".join(b.text for b in response.content if b.type == "text")


def default_out_path(src_md: Path, target: str) -> Path:
    """``book.md`` + English -> ``book.english.md``, beside the original."""
    slug = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-") or "translated"
    # with_name(stem), not with_suffix: with_suffix strips the last dot segment,
    # so `Calakmul v1.2.md` came back as `Calakmul v1.english.md`.
    src_md = Path(src_md)
    return src_md.with_name(f"{src_md.stem}.{slug}.md")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Translate finished Markdown, page by page, resumably."
    )
    ap.add_argument("src", type=Path, help="Markdown from stage 3")
    ap.add_argument("out", type=Path, nargs="?", help="output (default: SRC.<lang>.md)")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="target language")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT,
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--pages-per-chunk", type=int, default=DEFAULT_PAGES_PER_CHUNK)
    ap.add_argument("--chunk-dir", type=Path, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    if not args.src.exists():
        print(f"error: {args.src} not found")
        return 2

    out = args.out or default_out_path(args.src, args.target)
    chunk_dir = args.chunk_dir or args.src.parent / "translation" / "chunks"

    translator = ClaudeTranslator(
        model=args.model, effort=args.effort, max_tokens=args.max_tokens
    )
    _, stats = translate_document(
        args.src,
        out,
        chunk_dir,
        translator=translator,
        target=args.target,
        pages_per_chunk=args.pages_per_chunk,
        resume=not args.no_resume,
    )
    stats.input_tokens = translator.input_tokens
    stats.output_tokens = translator.output_tokens
    print()
    print(stats.render())
    print(f"\nWrote {out}")
    return 1 if stats.stopped_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
