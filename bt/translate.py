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

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from bt.postprocess import PAGE_ANCHOR_FMT, PAGE_MARK, split_pages
from bt.transcribe import chunk_path, concatenate

DEFAULT_TARGET = "English"
DEFAULT_PAGES_PER_CHUNK = 10

# One page of dense body text is ~1.5k output tokens; 8k leaves room for a heavy
# footnote page while staying inside the output cap of the smaller free models.
# Hitting this ceiling is an error, not a result -- see _read().
MAX_TOKENS = 8000

# Sent on every request, because urllib's default is not neutral. Left unset it
# sends ``Python-urllib/3.x``, which Cloudflare's managed rules ban outright --
# and Groq, OpenRouter and Gemini all sit behind a CDN. The refusal arrives as
# ``HTTP 403: error code: 1010`` from the edge, before the key or the model id
# is ever looked at, so it reads like a retired model. An ordinary agent string
# is all that is being asked for here; nothing is being disguised.
USER_AGENT = "bt-translate/1.0"

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
# the backend: any OpenAI-compatible /chat/completions endpoint
# --------------------------------------------------------------------------
# One request shape covers every free option worth having, which is why this is
# a single backend rather than one per vendor: a llama.cpp server on this
# machine, Ollama, and the free tiers of the hosted providers all speak it.
#
# Written against urllib rather than a vendor SDK on purpose. It adds no
# dependency, and the thing that actually needs care here is not the HTTP -- it
# is the rate limiting, which every free tier does differently and no SDK
# handles the way a 582-page unattended run needs.
@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    key_env: str | None = None
    # Requests per minute to hold to. 0 means unpaced -- correct for a server
    # running on this machine, where the only cost is the machine's own time.
    rpm: int = 0
    note: str = ""


PROVIDERS: dict[str, Provider] = {
    # No account, no key, no network. llama.cpp is already a hard requirement
    # of this project (marker 2.0 spawns llama-server itself), so the binary is
    # present on any machine that can run the OCR at all -- but it serves
    # surya's OCR model, not a translator, so this needs its own server:
    #   llama-server -hf <a GGUF instruct model> --port 8080
    "local": Provider(
        name="local",
        base_url="http://127.0.0.1:8080/v1",
        model="local-model",  # llama-server ignores the name and serves what it loaded
        rpm=0,
        note="llama-server on this machine; free and offline, costs disk and hours",
    ),
    "ollama": Provider(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:7b-instruct",
        rpm=0,
        note="Ollama on this machine; free and offline",
    ),
    # Free tiers. No money, but a request budget -- hence the pacing.
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct:free",
        key_env="OPENROUTER_API_KEY",
        rpm=20,
        note="free models (the ':free' suffix); a free account, no card",
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        key_env="GROQ_API_KEY",
        rpm=25,
        note="free tier, fastest of these by a distance",
    ),
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        key_env="GEMINI_API_KEY",
        rpm=12,
        note="free tier via Google's OpenAI-compatible endpoint; strongest here on Spanish",
    ),
}

DEFAULT_PROVIDER = "openrouter"


def resolve_provider(
    name: str, model: str = "", base_url: str = "", key_env: str | None = None
) -> Provider:
    """A preset, with anything the caller named taking precedence.

    Model ids on free tiers come and go, so the preset is a starting point
    rather than a constraint -- ``--model`` overrides it without needing a code
    change when one is retired.
    """
    try:
        provider = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; known: {', '.join(sorted(PROVIDERS))}"
        ) from None
    return replace(
        provider,
        model=model or provider.model,
        base_url=base_url or provider.base_url,
        key_env=key_env if key_env is not None else provider.key_env,
    )


def _urllib_transport(url: str, headers: dict, body: bytes, timeout: float):
    """POST, returning ``(status, payload)`` for any status rather than raising.

    A 429 is data here, not an exception: it carries the Retry-After the caller
    needs. Injected in tests, which is how the request is inspected without a
    key and without spending a free tier's budget.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        # Retry-After rides on the error, so hand it back rather than losing it.
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        return (exc.code, payload, {"Retry-After": retry_after} if retry_after else {})
    except urllib.error.URLError as exc:
        raise TranslationError(
            f"cannot reach {url}: {exc.reason}. For a local provider, is the "
            "server running? (llama-server ... --port 8080)"
        ) from exc


# Cloudflare's own error pages, which are not the provider's. The body is bare
# text -- ``error code: 1010`` -- where every real refusal from these APIs is
# JSON with a ``code`` naming the cause. Distinguishing them matters because
# the two point at opposite fixes: one is a wrong model id or key, the other
# means the request never reached the API at all.
EDGE_ERROR = re.compile(rb"\A\s*error code:\s*(\d{3,4})\s*\Z")

EDGE_CAUSES = {
    "1010": "the client's user agent is banned (urllib's default is)",
    "1015": "the edge is rate limiting this IP, ahead of the API's own limit",
    "1020": "a firewall rule rejected the request",
}


def edge_block_message(status: int, body: bytes) -> str:
    """Name a CDN refusal, or return "" if this is the provider's own error.

    Reported as ``HTTP 403: error code: 1010`` the failure sends the reader to
    check their key and their model id, neither of which is wrong -- that body
    never came from the API.
    """
    match = EDGE_ERROR.match(body or b"")
    if not match:
        return ""
    code = match.group(1).decode()
    cause = EDGE_CAUSES.get(code, "the request was rejected at the edge")
    return (
        f"HTTP {status}: Cloudflare error {code} -- {cause}. The request never "
        "reached the API, so neither the API key nor the model id is at fault. "
        "Check the network path (VPN, proxy, corporate DNS) and retry; a "
        "different provider (--provider gemini) avoids this edge entirely."
    )


class OpenAICompatTranslator:
    """Translate a page through an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        provider: Provider,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.2,
        rpm: int | None = None,
        max_retries: int = 6,
        timeout: float = 300.0,
        transport=None,
    ) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rpm = provider.rpm if rpm is None else rpm
        self.max_retries = max_retries
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self._last_call = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

        self.api_key = ""
        if provider.key_env:
            self.api_key = os.environ.get(provider.key_env, "")
            if not self.api_key:
                # Fail here rather than on page 1 of 582: the failure is the
                # same either way, but one of them costs a launch, a wait and a
                # look at the log to understand.
                raise TranslationError(
                    f"{provider.name} needs an API key in ${provider.key_env}. "
                    "It is free to obtain; export it and re-run."
                )

    # -- pacing ---------------------------------------------------------
    def _wait_for_slot(self) -> None:
        """Hold to the provider's request budget by choice.

        Better to wait three seconds than to be refused and wait sixty: a 429
        costs the request *and* the backoff, and free tiers count refusals.
        """
        if self.rpm <= 0:
            return
        interval = 60.0 / self.rpm
        due = self._last_call + interval
        now = time.monotonic()
        if now < due:
            time.sleep(due - now)

    # -- the request ----------------------------------------------------
    def _post(self, payload: dict) -> tuple[int, bytes, dict]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.provider.base_url.rstrip("/") + "/chat/completions"
        result = self._transport(
            url, headers, json.dumps(payload).encode(), self.timeout
        )
        status, body, *rest = result
        return status, body, (rest[0] if rest else {})

    def __call__(self, text: str, target: str) -> str:
        payload = {
            "model": self.provider.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt(target)},
                {"role": "user", "content": text},
            ],
        }

        last = ""
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            status, body, headers = self._post(payload)
            self._last_call = time.monotonic()

            if status == 200:
                return self._read(body)

            last = edge_block_message(status, body) or (
                f"HTTP {status}: {body[:200].decode('utf-8', 'replace')}"
            )
            # 429 and 5xx are worth waiting out; a 400 (bad model id, bad key)
            # will fail identically 582 times, so raise it now.
            if status != 429 and status < 500:
                raise TranslationError(last)
            if attempt == self.max_retries:
                break
            time.sleep(self._backoff(attempt, headers))

        raise TranslationError(f"giving up after {self.max_retries} retries -- {last}")

    def _backoff(self, attempt: int, headers: dict) -> float:
        """The server's own number if it gave one, else exponential."""
        retry_after = (headers or {}).get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 300.0)
            except (TypeError, ValueError):
                pass
        return min(2.0**attempt, 120.0)

    def _read(self, body: bytes) -> str:
        try:
            data = json.loads(body)
        except ValueError as exc:
            # A proxy, a captive portal or an HTML error page. Saying "not JSON"
            # points at the right problem; a KeyError would not.
            raise TranslationError(
                f"response was not JSON: {body[:200].decode('utf-8', 'replace')}"
            ) from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError(f"unexpected response shape: {data}") from exc

        # Truncation reads exactly like a page that ended there, so nothing
        # downstream can tell. Free-tier models often have small output caps,
        # which makes this the likeliest silent failure of the whole stage.
        if choice.get("finish_reason") == "length":
            raise TranslationError(
                f"the answer was truncated at max_tokens ({self.max_tokens}); the "
                "page would end mid-sentence. Raise --max-tokens, or use a "
                "smaller --pages-per-chunk on a model with a short output cap."
            )
        if not content.strip():
            raise TranslationError("the model returned an empty translation")

        usage = data.get("usage") or {}
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        return content


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
    ap.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=sorted(PROVIDERS),
        help="; ".join(f"{n}: {p.note}" for n, p in PROVIDERS.items()),
    )
    ap.add_argument(
        "--model",
        default="",
        help="override the provider's default model. Free-tier model ids are "
        "retired regularly, so this is the escape hatch when one stops working",
    )
    ap.add_argument("--base-url", default="", help="override the provider's endpoint")
    ap.add_argument(
        "--api-key-env",
        default=None,
        help="environment variable holding the key (default: the provider's)",
    )
    ap.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="requests per minute to hold to; 0 disables pacing. Defaults to "
        "the provider's free-tier budget",
    )
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

    try:
        provider = resolve_provider(
            args.provider,
            model=args.model,
            base_url=args.base_url,
            key_env=args.api_key_env,
        )
        translator = OpenAICompatTranslator(
            provider, max_tokens=args.max_tokens, rpm=args.rpm
        )
    except (ValueError, TranslationError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"{provider.name}: {provider.model} at {provider.base_url}")
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
