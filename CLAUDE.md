# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A pipeline that turns scanned PDFs into Markdown using
[marker](https://github.com/datalab-to/marker) as the OCR engine. It was built
for one document — a 1962 scan of *Being and Time* stored as two-page spreads —
and then generalised, so the Heidegger-specific facts below still explain most
of the design decisions.

Git repository; `master` is the main branch. Outputs go to `output/<pdf-stem>/`.

## Commands

```bash
uv sync                                   # install (Python 3.12/3.13; torch has no 3.14 wheels)
brew install llama.cpp                    # REQUIRED: marker 2.0 spawns llama-server itself
uv run python -m bt.preflight             # disk / llama-server / marker checks
uv run python -m bt.warmup                # pre-download ~1.8 GB of models

uv run streamlit run app.py               # GUI -- the normal entry point

uv run python -m bt.analyze FILE.pdf      # spreads? OCR needed? time estimate?
uv run bt-transcribe --pdf FILE.pdf                 # full run, resumable
uv run bt-transcribe --pdf FILE.pdf --split-only    # stage 1 only; no models needed
uv run bt-transcribe --pdf FILE.pdf --no-split --no-ocr   # single pages, text-layer extract
uv run python -m bt.verify output/NAME/NAME.md
uv run python -m bt.postprocess output/NAME/raw.md --report  # tune transforms, write nothing

uv run bt-transcribe --pdf FILE.pdf --images            # keep figures/charts too
uv run python -m bt.translate output/NAME/NAME.md --target English   # optional stage 4
```

**`--pdf` is required** — there is no default document any more.

`--split-only` is the fast feedback loop: it exercises the trickiest logic
(gutter detection) in about a minute with no models loaded. `bt.analyze` is
faster still and answers the two questions that matter before a long run.

## Skills — load these before working

| Skill | Load it before |
|---|---|
| `adding-a-feature` | **any** new feature, module, behaviour change, refactor or bug fix |
| `developing-with-streamlit` | editing `app.py` or anything Streamlit |

`.claude/skills/adding-a-feature/SKILL.md` is the blueprint for how work is
added here: check what is running first, find the seam, pin current behaviour,
write the failing test before the fix, keep the suite fast and hermetic, and
verify with `ruff` before `pytest`. Each step is there because skipping it has
already cost time on this project.

Load it **before** writing code, not after — several of its steps (checking for
a running worker, pinning behaviour before changing it) are worthless once the
edit is made.

## Tests

```bash
uv run ruff check .          # layer 0: undefined names, unused imports (~1s)
uv run pytest                # fast suite, ~2s, 127 tests
uv run pytest -m slow        # Streamlit smoke tests (boot the app; ~1 min locally)
uv run pytest -m ""          # everything, as CI runs it
uv run pytest --golden-update   # rewrite golden files -- review the diff
```

**Run `ruff` before `pytest`.** Two real bugs here were names used but never
imported (`re`, `os`); they compile cleanly and fail only when that line finally
executes, which for one of them was once per chunk, hours in.

**Tests must never touch live state.** Several functions have global side
effects — `stop_pids()` runs `pkill -f llama-server`, `clear_stale_sentinels()`
deletes from the real `~/.cache/datalab/surya`, and the worker lock lives in the
repo's `output/`. A worker is often processing books while tests run, so the
autouse `isolate` fixture in `tests/conftest.py` redirects all of it at
`tmp_path`. Rules when adding tests:

- Never signal a pid the test did not spawn itself.
- Call `stop_pids(..., kill_inference=False)`; the default runs a global `pkill`.
- Do not write under the repo's `output/`.

**Fixtures are synthetic and generated.** Real pipeline output is book text, so
it can be neither committed nor used as a golden file. `tests/conftest.py`
builds marker-shaped markdown (`{N}----` separators, `<sup>` markers, running
heads) with placeholder prose, and draws test PDFs with PyMuPDF — including
synthetic two-page spreads with a blank gutter. There are no binary fixtures.

**Most tests pin a bug that actually happened** and say so in the docstring. A
test whose purpose is recorded survives a refactor; one that merely asserts
something true gets deleted when it becomes inconvenient.

The suite is verified to work by reintroducing two real bugs (the `concatenate`
glob and the bare-rule page separator) and confirming it goes red — a suite
never seen failing is not known to work.

## Working on the GUI

**Load the `developing-with-streamlit` skill before editing `app.py`.** It is
installed in this repo (`.claude/skills/developing-with-streamlit`, symlinked
from the `streamlit` package) and routes to references for layout, caching,
fragments, session state, theming and testing.

```bash
streamlit docs st.<command>     # exact signature/params from the installed version
```

Conventions it enforces that this app follows:

- `use_container_width` is deprecated — use `width="stretch"` / `width="content"`.
- Prefer native elements over custom HTML/CSS; `st.container(border=True)` for
  grouping and `st.container(horizontal=True)` for responsive rows.
- Material Symbols icons (`:material/name:`) over emoji; sentence case for labels.
- Cache expensive work (`st.cache_data`), and isolate self-refreshing sections in
  `st.fragment` so the whole script does not rerun.

Test the app **without a browser** using `AppTest`, which executes the script
in-process and surfaces exceptions a 200 response would hide:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=300); at.run()
assert not at.exception
at.selectbox[0].set_value("meditationsofmar00marc.pdf").run()
```

## Architecture

`bt/run.py` orchestrates four stages. Each module is also runnable standalone.

| Stage | Module | Role |
|---|---|---|
| 0 | `preflight.py` | Gate the run: Python version, disk, `llama-server`, marker, torch |
| 1 | `split_spreads.py` | Cut 2-page spreads into single book pages (PyMuPDF) |
| 2a | `warmup.py` | Pre-download every model before any timed spawn |
| 2b | `transcribe.py` | Chunked, resumable OCR via marker |
| 3 | `postprocess.py` | Strip heads, namespace footnotes, dehyphenate |
| 4 | `translate.py` | **Optional.** Translate the finished Markdown via the Claude API |
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

**Post-processing must re-emit page anchors.** `process()` writes
`<!-- page N -->` where marker had `{N}----`. It renders as nothing, so a reader
never sees it, but `verify.check_not_embedded_layer` aligns output pages to PDF
pages by those numbers. An earlier version deleted them, which silently turned
the most important check into a no-op: it reported "ok / no meaningful embedded
text" for a whole 256-page book — a pass that proved nothing. Anchors also cost
cross-page hyphen joins (6 in that book), since the anchor now sits between the
split halves.

**Image links must survive running-head removal.** `_could_be_head()` rejects any
line containing `](`. Extracted figures are named by page and figure number, so
`images/0000-0009_page_3_Figure_2.jpeg` normalises to exactly the same form as
every other figure's link — a book with a chart on most pages had every one of
them deleted as a running head, leaving text that still read perfectly with the
pictures silently gone.

**Extracted images are namespaced by chunk.** marker numbers pages *within the
range it was given*, so two chunks each hand back a `_page_3_Figure_2.jpeg`.
Saved under marker's own name the second overwrites the first and both chunks'
Markdown then points at the same picture. `save_images()` prefixes the chunk
bounds and rewrites the links; writes are atomic (temp name, same extension,
then rename) for the same reason chunk writes are.

**Footnote ids must be namespaced per page.** Footnote "1" recurs on nearly every
page; un-namespaced ids would collide hundreds of times in one document.

**Chunk writes are atomic (temp file + rename).** A chunk killed mid-write would
otherwise look complete on resume and silently truncate the book.

**`concatenate()` must be passed the current run's bounds.** Globbing the chunk
directory splices overlapping page ranges into a duplicated book when
`--chunk-size` changes between runs.

## The GUI (`app.py`)

`uv run streamlit run app.py`. A launcher and monitor, not the pipeline.

**Why it runs a subprocess.** Streamlit re-executes its script on every
interaction, so a long job cannot live inside the app. `bt/jobs.py` launches
`bt.run` detached and reconstructs status **entirely from disk** — chunk files
are already written atomically and skipped on resume, so they *are* the
progress record. The GUI holds no state; close the browser or restart the app
and the run is untouched.

**Concurrency guard.** `jobs.find_pipeline_processes()` scans `ps` for any
`bt.run`/`bt.transcribe` process, not just ones the GUI started — a run
launched from a terminal has no lock file, and starting a second alongside it
is the worst thing possible on this hardware (two llama-servers → swap → see
`QnA.md` Q2).

**Generalisation for other PDFs.** `bt/analyze.py` decides split/OCR per
document. The decisive OCR signal is **full-page image coverage**: if a raster
covers the whole page, any text on it came from someone else's OCR, whatever
the producer string says. Producer names and run-on counts are too weak — a
scan of Marcus Aurelius reports "LuraDocument", has zero run-ons, and would
otherwise be misread as typeset.

Running-head removal is now **frequency-based** (`postprocess.find_running_heads`):
lines recurring at page edges across >30% of pages, with digits normalised
away. Repetition alone is not sufficient — `_could_be_head()` also requires the
line to look like a *label* (short, no sentence-ending punctuation, no footnote
markup), because templated body text repeats too and was being eaten.

`verify.check_not_embedded_layer()` is the general form of the stale-layer
check: it compares output against the PDF's own text **page by page**, using
marker's `{N}` separators to align them. Comparing whole-document blobs passes
for the wrong reason — different pages always look different.

## The queue worker (`bt/worker.py`, `bt/queue.py`)

`uv run python -m bt.worker` — processes every discovered PDF, one at a time,
indefinitely. It exists so the machine does not idle between books.

**Standalone process, not part of the GUI.** Streamlit runs its script only
while a browser session is connected, so a queue driven from `app.py` would stop
the moment the tab closed.

**State is derived, not tracked.** `queue.survey()` reads progress from chunk
files; `output/queue.json` persists only `skip`, `attempts`, `last_error` and a
cached analysis. Do not add a progress field there — it could only drift from
the chunk files, which are the truth.

**Completion is `chunks_done >= chunks_total`, never "output exists".**
`run.py` writes `raw.md` and a final `.md` even when the disk guard cut the run
short, and its exit code reports *verification*, not completion. `stopped_early`
never leaves `transcribe()`.

**An early stop is the normal case, not an error.** Runs here stop after a few
chunks by design (see Resource constraints). The worker cools down so swap can
drain, then resumes the same book. Success is measured as *chunks advanced*, so
a run that stopped early after real work does not burn an attempt.
`START_FREE_GB = MIN_FREE_GB + 1.5` keeps it from starting a run with only
enough room for a chunk or two.

`jobs.PIPELINE_CMD` matches `-m bt.run` / `-m bt.transcribe` specifically. It was
a loose substring test, which matched a transient process belonging to its own
inspection command — a false positive makes the worker wait for a job that does
not exist. `bt.worker` and `bt.queue` deliberately do not match.

`queue.resolve_out_dir()` is the single source of truth for where a document's
chunks live; `app.py` uses it too, so the GUI and worker cannot disagree and
resume one book onto another's output.

## Translation (`bt/translate.py`) — optional stage 4

`uv run python -m bt.translate output/NAME/NAME.md --target English`, or the
**Translate** panel under Result in the GUI. Not part of `bt.run` and not
something the worker does: translating a transcription you have not looked at
only multiplies whatever the OCR got wrong.

**The page anchors never reach the model.** They are stripped before the request
and re-emitted here, exactly as `postprocess.process()` does. A model told to
"preserve" `<!-- page N -->` would drop one eventually and nothing would look
wrong — the translation would still read perfectly while page alignment quietly
lost a page, which is the same failure that once turned
`verify.check_not_embedded_layer` into a no-op. One page in, one page out, by
construction.

**Footnote ids and image links are checked, not trusted.** They *are* sent (they
sit inside the prose), so `markup_signature()` compares them before and after
and `Stats.markup_drift` names every page where they changed. A renumbered
footnote still renders; it just points at the wrong note.

**Chunks are the progress record**, as in `transcribe.py` — same atomic write,
same resume, same `concatenate()` bounds guard. A failure stops the run rather
than skipping the chunk: a skipped chunk leaves a hole resume cannot see, and
the book would look finished with a missing stretch in the middle.

**It is a separate job from the pipeline.** Its own lock (`.translate.lock`),
log, and chunk directory (`translation/chunks/`, never `chunks/` — `queue.survey()`
counts files there to decide a book is done). `PIPELINE_CMD` deliberately does
not match `bt.translate`: translation waits on the network, not on memory, so it
neither needs nor should hold the one-OCR-at-a-time lock.

Defaults: `claude-opus-5` at `effort="low"` (high-volume, low-judgement work —
effort is charged per page and buys nothing here), one request per page, blank
pages skipped. `stop_reason` is checked before the text is read: a `max_tokens`
truncation is an error, because a page cut off mid-sentence reads exactly like a
page that ended there.

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
