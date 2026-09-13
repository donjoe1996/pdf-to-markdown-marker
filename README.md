---
title: PDF to Markdown OCR
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Being and Time → Markdown

Transcribes the scanned PDF of *Being and Time* (Macquarrie & Robinson) to
Markdown using [datalab-to/marker](https://github.com/datalab-to/marker). The
same pipeline generalises to other scanned or born-digital PDFs -- see
"Deploy your own copy" below for a live, upload-a-PDF web version.

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

The **web GUI is the easier entry point** — see below. From the command line:

```bash
# What am I dealing with? Spreads? Does it need OCR? How long will it take?
uv run python -m bt.analyze FILE.pdf

# The whole document, resumable
uv run bt-transcribe --pdf FILE.pdf

# Splitting only — needs no models, no llama.cpp, no GPU
uv run bt-transcribe --pdf FILE.pdf --split-only

# A single-page document whose text layer is trustworthy: seconds, not hours
uv run bt-transcribe --pdf FILE.pdf --no-split --no-ocr
```

Output lands in `output/<pdf-stem>/`:

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
| `--no-split` | treat each PDF page as one page (any document not stored as spreads) |
| `--no-ocr` | read the existing text layer instead of OCRing |
| `--use-llm` | LLM hybrid mode — mainly helps the polytonic Greek. Needs a key; see marker's docs |
| `--no-resume` | redo chunks that already exist |

## Web GUI

```bash
uv run streamlit run app.py
```

A local page for pointing the pipeline at any PDF without remembering flags.
It inspects the file first and pre-fills the settings, because the two
expensive choices are easy to get wrong and costly to undo:

- **Split spreads?** Correct for a scanned two-page spread; on a normal PDF it
  would cut every page in half. Detected from landscape shape plus a blank
  central band, and shown as a before/after preview so a wrong guess is caught
  before a multi-hour run rather than after.
- **OCR, or read the existing text?** A born-digital PDF extracts in seconds.
  A scan needs OCR — and the giveaway is that its pages are *full-page images*,
  so any text on them came from someone else's OCR, however clean it looks.

Long runs happen in a detached subprocess, with progress read back from the
chunk files on disk. You can close the browser, restart the app, or reboot the
GUI and the run is still there; **Resume** continues from the last finished
chunk. The app also refuses to start while another pipeline is running
anywhere on the machine — two at once push it into swap and slow both down.

## Deploy your own copy (free, no server to manage)

The GUI above is normally a local page. `Dockerfile` packages the same app
plus a CPU build of `llama-server` (see the Traps section on why marker
needs it) so it can run as a public web app on [Hugging Face
Spaces](https://huggingface.co/spaces)' free CPU tier -- 16 GB RAM, enough for
torch + surya + llama-server's ~2.4 GB working set, with no card required.

**The real tradeoff of "free":** this repo was tuned around Apple Silicon's
Metal-accelerated llama.cpp (82-95 s/page). The free tier has no GPU, so OCR
runs noticeably slower there -- budget minutes per page, not seconds. A free
Space also sleeps after a period of inactivity and cold-starts on the next
visit. The models (~1.8 GB) are baked into the Docker image at build time
specifically to keep that cold start to "container restart," not "download
1.8 GB again."

**A naive login, so one visitor's pages don't become everyone's problem.**
`BT_PUBLIC_MODE=1` (set by the Dockerfile, never by a local run) turns on a
lightweight account gate in `app.py`, backed by `bt/auth.py`:

- Sign up with just a username and an email -- **the email is never
  verified.** A one-time code is shown once, right there in the browser, and
  that's the login: no inbox is ever involved, including for "forgot your
  code," which just reissues a new code to anyone who can name a matching
  username + email pair. This is intentionally not real security -- it
  exists to give each visitor a private upload folder and a place to hang
  the page cap below, not to prove who anyone is.
- Each document is capped at `bt.auth.MAX_PAGES_PER_DOCUMENT` (5) pages after
  splitting; a longer upload is rejected outright rather than truncated,
  since the free CPU tier is shared by everyone using the link at once.
- Accounts live in `output/accounts.json` -- **not persistent** on the free
  Spaces tier. Only the Docker image itself survives a sleep/restart cycle;
  anything written at runtime, accounts and uploads included, resets with
  it. Fine for a casual demo link; if real persistence matters, look at
  Spaces' paid persistent storage, or syncing that file to a free private HF
  Dataset repo with the same token this deployment already uses.
- The unattended queue worker (below) is hidden in this mode: it would
  process every account's uploads at once with no way to see the per-account
  cap, on hardware sized for one job at a time.

One-time setup (a Hugging Face account and a repository setting need a human
with access to click them -- nothing here can do that for you):

1. Create a free Space at <https://huggingface.co/new-space>: any name,
   **Docker** as the SDK, **CPU basic** as hardware.
2. Create a Hugging Face access token with **write** access
   (<https://huggingface.co/settings/tokens>).
3. In this GitHub repo's settings, add:
   - **Secret** `HF_TOKEN` — the token from step 2.
   - **Variable** `HF_SPACE_REPO` — `your-username/your-space-name` from step 1.

From then on, `.github/workflows/deploy-hf-spaces.yml` pushes every update on
`master` straight to the Space, which rebuilds the Docker image and
redeploys it automatically -- no further manual steps. Trigger it by hand
from the Actions tab (`workflow_dispatch`) to deploy before merging to
`master`, or after adding the secrets for the first time.

## Keeping the machine busy — the queue worker

```bash
uv run python -m bt.worker
```

Transcription takes hours, so the expensive waste is idle time: a book that
finishes at midnight does nothing until someone notices in the morning. The
worker picks up the next document the moment the previous one stops, and keeps
watching for new ones, so overnight hours become progress.

It processes **every PDF** in the project root and `uploads/` — drop a file in
and it gets transcribed eventually. The GUI's **Queue** panel shows the order,
each book's progress, and a per-book *Skip* toggle.

Two design points, both consequences of how this machine behaves:

- **A run stopping early is normal, not a failure.** The disk guard halts a run
  once free space runs low, which here happens after a few chunks. The worker
  waits for the disk to recover and resumes the *same* book, so a book finishes
  as many short cycles rather than one long run. It judges success by whether
  chunks advanced, not by exit status — a run that transcribed two chunks before
  stopping made progress.
- **Only one job runs at a time.** Two would push the machine into swap and slow
  both down, so the worker waits for anything already running, including a job
  you started by hand.

A document that completes several attempts without advancing a single chunk is
marked *stalled* and the worker moves on, rather than blocking the queue.

The worker runs as its own process rather than inside the GUI because Streamlit
only executes its script while a browser is connected — with the tab closed,
nothing would tick.

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

A run passes through six stages. The short version: **look before you leap,
cut the pages apart, fetch the models, read the pages, tidy the text, then
prove it actually worked.**

### 0 · Look at the document first — `analyze.py`

Two decisions dominate everything else, and both are expensive to get wrong, so
the pipeline inspects the file before touching it.

*Is this one page per page, or two?* A scanned book is often photographed as
facing pairs. Splitting those is essential; splitting a normal PDF cuts every
page in half. The giveaway is a page that is both **landscape** and has a
**blank stripe down the middle**, sampled across ~20 pages.

*Does it need OCR at all?* If the PDF was exported from a word processor its
text is already perfect and extracting it takes seconds. The reliable test is
not the metadata but whether **a single image covers the whole page** — if it
does, the page is a photograph of paper, so any text sitting on it was produced
by somebody else's OCR, however tidy it looks. That text is then treated as
untrustworthy rather than reused.

Out of this comes a recommendation, a page count and a time estimate.

### 1 · Check the machine can finish — `preflight.py`

A transcription runs for hours, so the avoidable failures are checked up front:
a Python version that torch has wheels for, enough free disk, the `llama-server`
binary marker will try to spawn, and marker itself. Failing in ten seconds beats
failing in four hours.

### 2 · Cut the spreads apart — `split_spreads.py`

Each scanned page is examined at low resolution and its ink projected into a
column profile — effectively asking "how much darkness is in each vertical
slice?" The gutter is the **widest run of completely blank columns** near the
middle.

Two details matter. Asking merely for the *emptiest* column does not work,
because the whole gutter is empty and the answer drifts to wherever it first
looks. And the fold is not at the midpoint — on this book it wanders between
49.7% and 51.4%, enough that a fixed half-and-half cut would shave text off the
edges.

The halves are then written as new pages by **changing what part of the original
is visible**, rather than re-rendering them. Nothing is re-photographed, so the
OCR stage still sees the original scan at full quality. Halves that are
essentially blank — the backs of title pages — are dropped rather than sent
through OCR to produce nothing.

### 3 · Fetch the models before they are needed — `warmup.py`

marker starts each model as its own little server and waits a fixed time for it
to answer. On a first run that server is still downloading its weights when the
clock runs out, so marker kills it and reports what looks like a crash.
Downloading everything in advance turns that into a non-event. It is resumable,
so an interrupted download costs nothing.

### 4 · Read the pages — `transcribe.py`

This is the slow part, and it is not scanning: a vision model **looks at each
page and writes the text out**, which is what lets it keep italics, recognise
Greek, and tell a footnote marker from a number in the prose.

The work is done in **chunks of a few pages**, each written to disk the moment
it finishes, and a re-run skips whatever is already done. That single decision
is what makes an hours-long job survivable — interruptions, crashes, closing the
laptop and a deliberate stop all cost at most one chunk.

Two guards run alongside. Each chunk is written to a temporary file and renamed
into place, so a job killed mid-write can never leave a half-finished chunk that
later looks complete. And free disk is checked between chunks, because the model
pushes this machine into swap, swap grows on the same disk, and a run that fills
the disk takes the rest of the system with it.

### 5 · Tidy the text — `postprocess.py`

Raw OCR output is accurate but not yet comfortable to read:

- **Running heads** — the title and page number repeated at the top of every
  page — are found by looking for lines that *recur* at page edges once their
  numbers are ignored, then removed. Repetition alone is not enough to convict a
  line, so it must also look like a label rather than a sentence.
- **Footnote markers are renumbered per page.** Almost every page has a footnote
  "1", and in one long document those would all collide.
- **Words broken across a line break** are rejoined.

Each step can be switched off, and a report mode shows what *would* change
without writing anything.

### 6 · Prove it worked — `verify.py`

The failure worth fearing is silent: marker quietly reads the PDF's existing bad
text instead of looking at the page, and returns fluent Markdown made of the
wrong characters. Nothing about it looks broken.

So the output is compared against the PDF's own text layer, **page by page**, and
near-identical output is treated as a failure rather than a success. Alongside
that it counts run-on words (the signature of old OCR losing its spaces), plus
Greek, italics and footnote markers — reported for information, since plenty of
documents legitimately have none.

## Known limitations

- **Polytonic Greek** is the weakest area without `--use-llm`.
- The two footnote series (arabic = translators', roman = Heidegger's
  marginalia) are namespaced per page but not separated by series.
- PyMuPDF is AGPL-licensed — fine for local use, relevant if redistributed.
