# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A pipeline that transcribes a scanned PDF of *Being and Time* (Macquarrie &
Robinson, 1962) into Markdown, using [marker](https://github.com/datalab-to/marker)
as the OCR engine. Single-purpose tooling around one specific document, not a
general library.

Not a git repository. Source PDF: `42700894-Martin-Heidegger-Being-and-Time.pdf`
(26 MB, 294 pages). Outputs go to `output/`.

## Commands

```bash
uv sync                                   # install (Python 3.12/3.13; torch has no 3.14 wheels)
brew install llama.cpp                    # REQUIRED: marker 2.0 spawns llama-server itself
uv run python -m bt.preflight             # disk / llama-server / marker checks
uv run python -m bt.warmup                # pre-download ~1.8 GB of models

uv run bt-transcribe --test               # validation slice (source pages 8,40-41)
uv run bt-transcribe                      # full book, ~14.5 h, resumable
uv run bt-transcribe --split-only         # stage 1 only; needs no models at all
uv run python -m bt.verify output/being-and-time.md
uv run python -m bt.postprocess out/raw.md --report   # tune transforms, write nothing
```

`--split-only` is the fast feedback loop: it exercises the trickiest logic
(gutter detection) in about a minute with no models loaded.

## Architecture

`bt/run.py` orchestrates four stages. Each module is also runnable standalone.

| Stage | Module | Role |
|---|---|---|
| 0 | `preflight.py` | Gate the run: Python version, disk, `llama-server`, marker, torch |
| 1 | `split_spreads.py` | Cut 2-page spreads into single book pages (PyMuPDF) |
| 2a | `warmup.py` | Pre-download every model before any timed spawn |
| 2b | `transcribe.py` | Chunked, resumable OCR via marker |
| 3 | `postprocess.py` | Strip heads, namespace footnotes, dehyphenate |
| — | `verify.py` | Quality checks on the finished Markdown |

`transcribe.py` and `warmup.py` set env vars **at import time, before torch is
imported**. Keep marker/torch imports function-local in those modules — hoisting
them to module scope silently breaks the env setup.

## Document facts that drive the design

Verified by inspection; do not re-derive:

- **Every PDF page is a 2-page spread.** 294 landscape pages ≈ 588 book pages.
  Splitting yields **582** pages (6 halves are blank: source pages 0, 2, 8, 9,
  32, 250).
- **The embedded text layer is bad OCR** (Acrobat "Paper Capture"), with
  artefacts like `itse1f`, `sorne`, and collapsed word spacing. `force_ocr` +
  `strip_existing_ocr` are mandatory, not tuning.
- **The scan is 300 DPI bitonal.**
- **The gutter is not at 50%** — it lands at 0.497–0.514 across the book.
- Index/back matter pages are genuinely **two-column within a book page**.
- Footnotes use **two series**: arabic (translators') and roman (Heidegger's
  marginalia).

## Traps

Each of these cost real debugging time. Read before changing related code.

**Gutter detection must use the widest zero-ink *run*, not `argmin`.** The whole
gutter reads zero, so `argmin` returns the first zero it meets and drifts to the
band edge.

**Splitting uses `set_cropbox`, deliberately.** It is lossless, so marker
resamples the original scan exactly once. Rasterising here would resample twice.
A `--rasterize` escape hatch exists only if the text layer ever leaks through.

**`HF_HUB_DISABLE_XET=1` is load-bearing.** The Hub routes large files through
its Xet CDN bridge, which on some networks accepts the connection and then
delivers nothing — downloads hang at **0 bytes forever**. An auth token does not
fix it; `hf_transfer` does not fix it. Without this var a first run just hangs.

**Models must be downloaded before marker runs.** marker 2.0 starts each model
in its own server subprocess with a 300s health check. On a cold cache the
download outlives the check and marker force-kills the server, producing a
`SpawnError` that reads like a crash. That is what `warmup.py` prevents; the
`*_SERVER_STARTUP_TIMEOUT=1800` vars are a second layer.

**DPI is not a speed lever.** 192 DPI measured 92.1 s/page vs 82.5–95.2 s/page at
300, with 99.71% identical output — surya resizes to a fixed internal
resolution. Lowering DPI costs fidelity and buys nothing. Keep 300.

**`mode="fast"` is the default on MPS and reads the PDF text layer** via pdftext.
For this document that is exactly wrong, which is why `force_ocr` is forced on.
Benchmarks put `fast` at 76.0 vs `balanced` 76.6 on multi-column, so `balanced`
is rarely worth its cost here.

**marker discards the marginal Niemeyer numbers.** Its MarginaliaProcessor drops
them before the Markdown is written, so `[H. n]` anchors cannot be recovered
from marker output. `--margins` still enables the transform, but it is a no-op
unless a number leaks through. Recovering them properly needs a separate
margin-strip OCR pass.

**marker's page separator is `{N}------...`, not a bare rule.** A dashes-only
pattern matches nothing, collapsing the book into one "page" and breaking
per-page footnote namespacing.

**Footnote ids must be namespaced per page.** Footnote "1" recurs on nearly every
page; un-namespaced ids would collide hundreds of times in one document.

**Chunk writes are atomic (temp file + rename).** A chunk killed mid-write would
otherwise look complete on resume and silently truncate the book.

**`concatenate()` must be passed the current run's bounds.** Globbing the chunk
directory splices overlapping page ranges into a duplicated book when
`--chunk-size` changes between runs.

## Running on a GPU (Colab)

`colab/Being_and_Time_Colab.ipynb` + `bt/gpu_setup.py`. Two facts shape it:

- **marker's NVIDIA default needs Docker.** The `vllm` backend spawns the
  `vllm/vllm-openai` image, and Colab has no Docker daemon. `gpu_setup` forces
  `SURYA_INFERENCE_BACKEND=llamacpp` instead, whose spawn already passes
  `-ngl 99` (`LLAMA_CPP_NGL`) — all layers on CUDA on Linux, exactly as it uses
  Metal on a Mac. Same validated code path, no second one to maintain.
  (`SURYA_INFERENCE_URL` would also let surya attach to a hand-run vLLM server,
  if that is ever wanted.)
- **llama.cpp ships CUDA binaries for Windows only.** Linux release assets are
  x64/arm64/vulkan/rocm/sycl. So `gpu_setup.build_llama_server()` compiles it
  with `-DGGML_CUDA=ON` for the detected arch (one arch, for build speed) and
  caches the binary — point `cache_dir` at Drive so sessions reuse it.

A GPU also allows `SURYA_INFERENCE_PARALLEL > 1` (several pages at once), which
the memory-starved Mac could not. Each slot costs ~12k ctx of KV cache against a
~1.5 GB model, so a 16 GB T4 comfortably takes 4.

`preflight.check_llama_server()` honours `LLAMA_CPP_BINARY` before `PATH`,
because on Colab the binary is never on `PATH`.

## Resource constraints

This machine runs near-full, and the OCR run is what pushes it over.
`llama-server` holds a ~2.4 GB working set and drives **multi-GB swap growth on
the boot volume**, which macOS does not reclaim promptly after the process
exits.

`transcribe.py` stops cleanly below `MIN_FREE_GB = 2.0`, but **only checks at
chunk boundaries** — a 20-page chunk is ~30 minutes, during which the disk can
still fill. Use a smaller `--chunk-size` when headroom is tight so the guard
fires sooner.

Check free space before starting a long run. Reclaimable candidates (delete
nothing without asking): `~/anaconda3/pkgs`, `~/Library/Caches`, `~/.cache/uv`.

## Verification

`verify.py`'s most important check is `check_text_layer_replaced`: it greps for
the embedded layer's distinctive damage. **Any hit means marker reused the bad
Acrobat text instead of OCRing** — readable Markdown made of the wrong
characters. Always run it after changing OCR config.

Expected on a healthy run: no stale-layer artefacts, 0 run-on words, Greek runs
found, italic spans found, footnote markers found.
