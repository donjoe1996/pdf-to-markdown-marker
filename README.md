# Being and Time → Markdown

Transcribes the scanned PDF of *Being and Time* (Macquarrie & Robinson) to
Markdown using [datalab-to/marker](https://github.com/datalab-to/marker).

## Why this isn't just `marker_single file.pdf`

The source scan has four properties that break a naive run:

| Property | Why it matters |
|---|---|
| Each PDF page is a **2-page spread** (294 landscape pages ≈ 588 book pages) | Layout models merge lines across the gutter and can't tell the two pages' headers and footnotes apart |
| It carries a **bad embedded OCR layer** (Acrobat "Paper Capture") | marker's `fast` mode reads the text layer instead of OCRing, silently producing wrong characters |
| 300 DPI bitonal scan | marker's default `highres_image_dpi` of 192 loses the small footnote type |
| Marginal **Niemeyer pagination** in the outer margins | These are the standard citation anchors for Heidegger; naive output scatters bare numbers through the prose |

So the pipeline splits the spreads first, forces real OCR, and cleans up after.

## Setup

```bash
# 1. The surya VLM backend. marker 2.0 spawns this itself but does NOT install it.
brew install llama.cpp

# 2. Python deps. Python 3.12 or 3.13 — torch has no 3.14 wheels yet.
uv sync
```

Needs roughly **4 GB free**: ~2 GB for the venv (torch) and ~1.6 GB of surya
weights downloaded on first run. `preflight` checks this and refuses to start
otherwise; it never deletes anything.

```bash
uv run python -m bt.preflight
```

```bash
# 3. Pre-download the models (~1.8 GB). Optional but strongly recommended.
uv run python -m bt.warmup
```

**Why warmup matters.** marker 2.0 runs each model in its own server
subprocess and waits 300s for it to report healthy. On a first run those
subprocesses still have to *download* their weights — the ocr-error model alone
is 258 MB — so on a slow link the download outlives the health check, marker
force-kills the server, and the run dies with a misleading error:

```
SpawnError: ocr_error server failed to become healthy at
http://127.0.0.1:65071 within 300.0s.
```

`bt.warmup` fetches everything up front so later spawns start from cache. It is
resumable and idempotent — re-running after an interruption costs nothing.
`bt-transcribe` runs it automatically; the standalone command just lets you get
the download out of the way first. As a second layer of defence, `transcribe.py`
raises the `*_SERVER_STARTUP_TIMEOUT` env vars to 1800s.

### If the download sits at 0 bytes

The Hub resolves large files through its **Xet** CDN bridge
(`us.aws.cdn.hf.co/xet-bridge-us`). On some networks that endpoint accepts the
connection and then delivers nothing — the download hangs at 0 bytes forever.
Observed here: it is *not* rate limiting, and an auth token does not fix it;
`hf_transfer` does not fix it either. Forcing the classic CDN path does:

```bash
export HF_HUB_DISABLE_XET=1
```

`warmup.py` and `transcribe.py` both set this by default, so you should not hit
it — but it is worth knowing, because the symptom looks like a dead network
rather than a CDN problem. As a last resort `warmup.py` also carries a
resumable `curl` fallback that writes straight into the HF cache layout.

An `HF_TOKEN` (via `huggingface-cli login`) is still worth setting for higher
rate limits, but it is not what unblocks a stalled download.

## Use

```bash
# Validation slice: book page 1 (dense Greek + big footnote block) + a body spread
uv run bt-transcribe --test

# The whole book
uv run bt-transcribe

# Stage 1 only — needs no models, no llama.cpp, no GPU
uv run bt-transcribe --split-only
```

Output lands in `output/`:

| File | Contents |
|---|---|
| `pages.pdf` | the split single-page PDF |
| `pagemap.json` | `output page → (source page, half)`, for tracing problems back |
| `chunks/` | per-chunk Markdown; **re-running skips completed chunks** |
| `raw.md` | concatenated marker output |
| `being-and-time.md` | final cleaned Markdown |

A full run takes hours. It is chunked and resumable — interrupt it and re-run
the same command to pick up where it stopped.

### Useful flags

| Flag | Effect |
|---|---|
| `--mode balanced` | VLM layout instead of rf-detr. Slower; marker's own benchmark puts multi-column at 76.6 vs 76.0, so rarely worth it here |
| `--dpi N` | `highres_image_dpi` (default 300, matching the scan) |
| `--pages '8,40-41'` | process specific source pages |
| `--chunk-size N` | pages per resumable chunk (default 20) |
| `--use-llm` | LLM hybrid mode — mainly helps the polytonic Greek. Needs a key; see marker's docs |
| `--no-resume` | redo chunks that already exist |

## Running on a GPU instead (Google Colab)

On an M2 the full run takes ~7 hours. `colab/Being_and_Time_Colab.ipynb` runs
the same pipeline on a free Colab T4, which has roughly 3x the memory bandwidth
and enough spare VRAM to OCR several pages at once.

Open `colab/Being_and_Time_Colab.ipynb` in Colab and run the cells. It clones
this repo, so there is nothing to copy but the PDF: put the scan in a Drive
folder and point cell 4 at it. Output is written back to Drive, so a
disconnected session resumes rather than restarting.

The PDF stays out of git deliberately — it is large, and the scan is not ours
to redistribute.

The notebook handles the two Colab-specific obstacles: marker's NVIDIA backend
wants Docker (unavailable there, so the llama.cpp CUDA path is forced), and
llama.cpp publishes CUDA binaries for Windows only (so `llama-server` is built
once and cached to Drive).

## How it works

**Stage 1 — `split_spreads.py`.** Finds the gutter per page by projecting ink
vertically at 36 DPI and taking the *widest run of zero-ink columns* in the
central 35–65% band. (Plain `argmin` fails: the whole gutter reads zero, so it
drifts to the band edge.) Measured gutters on this book land at 0.497–0.514, so
a hard 50% split would clip text. Halves are emitted with `set_cropbox`, which
is lossless — marker then resamples the original scan exactly once. Halves
holding under 2% of the spread's ink are dropped (6 exist: source pages 0, 2,
8, 9, 32, 250).

**Stage 2 — `transcribe.py`.** Runs marker with `force_ocr` +
`strip_existing_ocr` so the bad text layer is never used, at 300 DPI, paginated,
with image extraction off. Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (marker's CLI
sets it; the library doesn't) and `SURYA_INFERENCE_KEEP_ALIVE=1` so the
inference server is started once rather than per chunk.

**Stage 3 — `postprocess.py`.** Drops running heads, converts marginal numbers
to inline `[H. 56]` anchors and checks the sequence is consecutive, namespaces
footnote markers per page (footnote "1" recurs on nearly every page, so
un-namespaced ids would collide hundreds of times), and rejoins words
hyphenated across line breaks. Each transform has a `--no-*` switch, and
`--report` shows what would change without writing.

**`verify.py`.** The important one is `check_text_layer_replaced`: the embedded
Acrobat layer has distinctive damage (`itse1f`, `sorne`, collapsed word
spacing), so finding any of it in the output proves marker fell back to the old
text layer instead of OCRing. Also checks word spacing, Greek, italics, and
footnote markers.

## Known limitations

- **Polytonic Greek** is the weakest area without `--use-llm`.
- The two footnote series (arabic = translators', roman = Heidegger's
  marginalia) are namespaced per page but not separated by series.
- PyMuPDF is AGPL-licensed — fine for local use, relevant if redistributed.
