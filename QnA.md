# Q&A — learning notes

A running log of questions about this codebase, with answers. Newest questions
are appended at the bottom.

---

## Q1 — Why must the model be downloaded and loaded locally before any page can be transcribed?

**As asked:** *"Why does the LLM model need to be unpacked first in my local
computer, then after that can be used to perform this task?"*

**Sharpened:** *Why does the pipeline have to (a) download ~1.8 GB of model
weights to disk and then (b) load them into memory before it can transcribe even
a single page — instead of just starting work immediately?*

### Short answer

Because the "model" is not a program that knows how to read. It is a large table
of learned numbers, and the arithmetic that turns a page image into text is
performed **on this machine**, using those numbers. Numbers you do not have on
hand cannot be multiplied. So they must first be fetched over the network, then
placed in memory fast enough to be read millions of times per page.

### What the model actually is

`surya-2.gguf` is ~1.27 GB of **weights** — roughly 650 million numbers learned
during training. There is no separate "program" inside that file that
understands Greek or italics. The behaviour *is* the numbers: reading a page
means multiplying the image and the text-so-far through those 650 million values
in a fixed pattern.

This pipeline pulls four separate things (see `bt/warmup.py`):

| File | Size | Job |
|---|---|---|
| `surya-2.gguf` | 1.27 GB | the vision-language model that reads and writes text |
| `surya-2-mmproj.gguf` | 205 MB | the "projector" that turns image patches into something the model can attend to |
| ocr-error detection | 258 MB | judges whether existing page text is garbage |
| text detection + layout | ~210 MB | finds text regions and reading order |

### Why "unpacked" is the right instinct — there are two distinct steps

Your word *unpacked* captures something real. Two different things happen, and
they have different costs:

**Step 1 — Download (network → disk).** Paid **once, ever**. The files land in
`~/.cache/huggingface` and `~/Library/Caches/datalab`. Every later run finds them
there. This is what `bt/warmup.py` does deliberately up front.

**Step 2 — Load (disk → RAM/VRAM).** Paid **once per run**. `llama-server`
starts, reads the GGUF, and lays the weights out in memory ready for arithmetic
— this is why you saw it holding ~2.4 GB resident. A GGUF is a container
(quantised weights plus metadata describing the shapes), so there is genuine
unpacking involved, not just copying.

### Why the weights must be in *memory*, not left on disk

This is the part that explains the runtime.

Generating one token requires a pass over **all** the weights. A dense page of
this book produces on the order of 1,500 tokens. So per page the machine streams
roughly 1.27 GB × 1,500 ≈ **2 terabytes** through memory.

RAM delivers that at ~100 GB/s on an M2. An SSD is one to two orders of
magnitude slower. Reading the weights from disk per token would turn a
40-second page into hours. So the one-time load is what makes the per-page work
possible at all.

It also explains a result that surprised us: lowering DPI from 300 to 192 did
**not** speed anything up (measured 92.1 vs 82.5–95.2 s/page, 99.71% identical
output). The cost is dominated by weights-read-per-token-generated, not by how
many pixels go in.

### Why not just call a cloud API instead?

That is a real alternative — marker supports Gemini, Claude and OpenAI via
`--use-llm`. Then no weights live here at all; you send each page image to a
server that already has them loaded.

The trade-offs, and why local won for this job:

| | Local weights | Cloud API |
|---|---|---|
| One-time setup | ~1.8 GB download | none |
| Per page | free, offline | a network round-trip and a fee |
| 582 pages | costs nothing extra | costs real money |
| Privacy | nothing leaves the machine | every page is uploaded |
| Works offline | yes | no |

The download is a **fixed** cost amortised across 582 pages; API calls are a
**per-page** cost. For a whole book, local wins.

### How this shaped the code

Two consequences worth knowing, both in `bt/warmup.py`:

1. **marker does not wait for the download.** It starts each model in its own
   server subprocess and gives it 300 seconds to report healthy. On a first run
   that subprocess is still downloading when the clock runs out, so marker
   force-kills its own server and raises a `SpawnError` that reads like a crash
   but is really a timeout. Downloading everything *before* marker runs is the
   fix.
2. **Loading is why chunking pays.** Because the load happens per run rather
   than per page, the pipeline creates the model handles once and reuses them
   across all chunks (`create_model_dict()` is called once in
   `bt/transcribe.py`), rather than paying the startup cost 59 times.

### One-line summary

The model is data, the arithmetic happens here, and memory is the only place
fast enough to hold data that gets re-read 1,500 times per page — so it must be
downloaded once, then loaded once, before page one.
