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

---

## Q2 — Why does the transcription process end up using swap memory?

**As asked:** *"Why does this transcribe process involve swap memory?"*

**Sharpened:** *Why does running this pipeline make macOS allocate gigabytes of
swap — growing to ~10 GB on this machine — when `llama-server`'s own working set
is only about 2 GB? And why does that make the job progressively slower rather
than just a bit slower?*

### Short answer

Swap appears because everything the run needs at once does not fit in physical
RAM, so macOS starts moving pages out to disk. That would normally be a mild
slowdown. Here it is catastrophic, because an LLM re-reads **all** of its weights
for **every token** — so pages the OS evicts are needed again milliseconds later.
The usual assumption that makes swap work is false for this workload.

### What swap actually is

RAM is finite. When programs collectively want more than exists, the OS picks
memory pages that look least recently used, writes them to disk, and hands the
freed RAM to whoever asked. If the original owner touches an evicted page, the OS
must fetch it back from disk — a **page fault**.

Swap is disk pretending to be RAM. It works well when the bet behind it holds:
*that the paged-out memory won't be needed again soon.*

### Why this pipeline pushes RAM over the edge

Several things are resident simultaneously:

| Consumer | Roughly |
|---|---|
| `llama-server` holding the VLM weights | ~1.5–2.4 GB |
| KV cache — 12,288 tokens per slot, min 16,384 total context | hundreds of MB |
| The Python process: torch, marker, plus three smaller models (ocr-error, text detection, layout) | ~1–2 GB |
| Page images rendered at 300 DPI by PyMuPDF | tens of MB, churning |
| macOS, browser, editor | the rest |

None of these is outrageous alone. Together, on a machine that was already near
capacity with a nearly full disk, they exceed what is available.

### Why it degrades instead of just being slower — the important part

Most programs have **locality**: they touch a small working set repeatedly, so
evicting the rest is nearly free. The OS's least-recently-used heuristic is built
on that assumption.

A transformer forward pass has **no locality at all**. Generating a single token
sweeps the entire weight array start to finish. There is no "cold" region — every
page is touched ~1,500 times per page of output. So whatever the OS evicts gets
demanded back almost immediately, at disk speed instead of memory speed.

That creates a feedback loop:

1. Memory pressure → OS pages out part of the weights
2. Next token needs those pages → page fault → disk read
3. Tokens now take far longer → the process stays resident longer
4. Pressure persists → more eviction → back to 1

### The evidence from this run

The numbers recorded during the local attempt show exactly this:

| Observation | Value |
|---|---|
| Per-page rate, early chunks | 39.0 → 44.9 → 47.4 s/page |
| Per-page rate, later | **164.4 s/page** |
| Swap at rest | ~5 GB |
| Swap during the run | **11.2 GB allocated, 9.9 GB used** |
| Disk free, worst point | **909 MB** (from 6.9 GB) |
| Disk after the process exited | recovered to 7.2 GB |

The clearest single clue: `llama-server`'s resident size **fell** from 2,427 MB
to 1,578 MB while it was still working. Its memory need had not shrunk — the OS
had paged ~850 MB of it out to disk, memory it still required for every token.
That gap is the slowdown.

### Why swap eats disk too

macOS grows swapfiles dynamically **on the boot volume**. So memory pressure
consumes free disk: swap climbed to 11.2 GB while free space fell to 909 MB.
When the process exits, the swapfiles are released and the space returns — which
is why the disk recovered to 7.2 GB on its own each time the run was stopped.

This is why `bt/transcribe.py` carries a disk guard (`MIN_FREE_GB = 2.0`) rather
than a memory check: on this platform, running out of memory shows up first as
running out of *disk*.

### Why a GPU sidesteps the whole problem

On a T4 the weights live in **16 GB of dedicated VRAM**. VRAM is not swapped —
there is no "page out to disk" path for it. The model stays resident, every token
reads it at full bandwidth, and the rate does not decay over a long run. That,
more than raw speed, is why the Colab path exists: not just faster, but *stable*.

### One-line summary

Swap shows up because RAM is oversubscribed — and it is ruinous here rather than
merely slow because an LLM has no memory locality, so every page the OS evicts is
needed again within milliseconds.
