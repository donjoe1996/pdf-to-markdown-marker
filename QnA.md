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

---

## Q3 — What did we build for testing, and why does it matter?

**As asked:** *"Summary in plain English on what we have done for the test side
and why it matters."*

**Sharpened:** *What does the test suite actually consist of, why was it built
this way rather than some other way, and what does it buy when adding features
to a program that takes hours to run?*

### The problem it solves

The pipeline had grown to roughly 3,000 lines across 13 modules with **no tests
at all**, and more functionality was coming.

The risk was never breaking something obviously. It was breaking something
*quietly*: change one thing, and a book transcribed overnight comes out subtly
wrong — noticed days later, after hours of compute were spent producing it. On a
job this slow, late discovery is what costs.

### What was built — three layers, fastest first

| Layer | Time | Catches |
|---|---|---|
| `ruff` (a linter) | ~1s | names used but never defined, unused imports |
| Unit tests | ~1s | individual pieces: gutter detection, text transforms, queue state |
| Integration + golden tests | seconds | whole stages wired together, compared against saved known-good output |

The linter reads the code without running it. It earns its place because the
failure it catches happened here twice: code that compiled cleanly and only blew
up when execution finally reached that line — for one bug, once per chunk, hours
into a run.

**79 tests in about 1.4 seconds.** The speed is deliberate, not incidental: a
suite that takes a minute is one you skip when in a hurry, and a skipped suite
protects nothing. The few slow tests — which boot the Streamlit app — are marked
separately and run in CI instead.

Real OCR is deliberately **excluded**. It needs the models and hours per run, so
it could never be part of a suite anyone actually runs. Output quality is still
checked the way it always was, by `bt/verify.py` against real output.

### Why *these* tests

Most of them pin a defect that **actually happened** while building this, and say
so in the docstring. Among them: chunk files quietly duplicating passages,
running-head removal eating real body text, finished jobs still looking like they
were running, one book's output folder adopted by another, and a verification
check that passed for the wrong reason.

This follows characterization testing (Feathers, *Working Effectively with Legacy
Code*): with working code and no tests, first pin the behaviour you have, *then*
change things. The tests describe what the code does today — which is exactly
what "prove I did not break it" requires.

The reasoning is simple: finding those bugs cost real time. Nobody should spend
that time twice.

### Proving the suite actually works

A test suite that has never been seen failing is not known to work. Plenty pass
because they are not really checking anything.

So two of the real bugs were **deliberately reintroduced** — the chunk-directory
glob and the bare-rule page separator — and the suite was confirmed to go red for
each (four failures and two respectively). Both files were then restored
byte-identically.

### Proving it is safe to run

Parts of this code have teeth. `stop_pids()` runs a global `pkill -f
llama-server`, sentinel cleanup deletes from the real model cache, and the worker
lock lives in the repo's `output/`. A worker is usually mid-book while tests run,
so a careless test could destroy hours of real work.

An autouse fixture in `tests/conftest.py` redirects every piece of that at a
temporary directory. The suite was then run *while the worker was processing*,
and its lock, its chunks and the three real sentinel files were all confirmed
untouched afterwards.

Fixtures are synthetic and generated rather than copied from real output — the
real output is book text, which cannot be committed. `conftest.py` builds
marker-shaped markdown and draws test PDFs with PyMuPDF, including synthetic
two-page spreads with a blank gutter. There are no binary fixtures in the repo.

### Two real bugs it found while being written

- `app.py` called `Path.relative_to()` unguarded, so pointing the GUI at an
  output folder outside the project would have taken the whole page down with an
  unhandled `ValueError`.
- Three `zip()` calls had no `strict=`. The one pairing queue rows with edited
  rows would have silently written a *Skip* toggle to the **wrong document** if
  the two ever diverged.

Neither had surfaced yet. Both are fixed.

### What it buys, and what it does not

Before committing a change, one command takes a second. Green means the
behaviour already relied on is intact; red names which behaviour broke and why.
The same runs automatically on GitHub on every push.

What it does **not** cover: OCR *quality*. No test here can tell you the Greek
came out right — that needs the models, and it stays with `bt/verify.py` against
real output.

### One-line summary

Three layers — lint, unit, integration — that run in about a second, built mostly
out of bugs that already happened, proven to fail when those bugs return, and
isolated so they can run safely while a real transcription is in progress.

---

## Q4 — How do the tests change the way new features get added?

**As asked:** *"How will it affect our next code improvement, for example adding
new features?"*

**Sharpened:** *With a suite now in place, what actually changes when adding
functionality — where is the work genuinely safer, and where is it no safer at
all?*

### The day-to-day change

The loop becomes: **write code → `ruff check` → `pytest` → then watch it run.**
Two seconds, before starting anything that takes hours.

The real shift is *when* a mistake is found. Previously a fault in text handling
surfaced only after a book finished, by reading the output and noticing something
odd. Now it surfaces before the run starts.

### What is genuinely protected

**The golden test does the heavy lifting.** Add a new cleanup step to
`postprocess` — say stripping page numbers — and if it also quietly eats footnote
markers, the golden test fails and shows the exact diff. Without it, catching
that needs someone reading a transcript closely enough to notice.

When a change to the output *is* intended, `pytest --golden-update` rewrites the
file and the diff gets reviewed. That is the value: the change becomes a
deliberate, visible decision rather than a silent one.

**New tests inherit the safety.** The autouse fixture redirects every test away
from the real `output/`, the worker lock and the model cache. It is not possible
to add a test that accidentally kills a running book.

**The old bugs cannot return quietly.** Each of the dozen has a test naming it.
That matters most during refactors, where reintroducing something while
"simplifying" is easy.

### Where features plug in

| Adding… | Lands in | Safety net |
|---|---|---|
| a text transform | `postprocess.process()` | strong — unit tests plus golden |
| queue behaviour | `queue.py` | good — tests build a directory layout and assert what is derived |
| a GUI panel | `app.py` | smoke test catches exceptions an HTTP 200 hides |
| an analysis signal | `analyze.py` | **none yet** |

### Where it will not help

Over-trusting a suite is its own risk, so the gaps matter as much as the cover:

- **It says nothing about OCR quality.** Green means the plumbing is intact, not
  that the Greek came out right. That still needs `bt/verify.py` against real
  output, and human eyes on a page.
- **It will not catch marker API changes.** marker is never imported in the fast
  suite, so an upgrade that breaks the real pipeline passes every test.
- **`bt/analyze.py` is untested** — and it decides *split or not* and *OCR or
  not*, the two choices that cost hours when wrong. A bug there is expensive
  precisely because it is silent.
- **The worker loop is untested**, and it is what runs unattended overnight.

Covered today: `postprocess`, `split_spreads`, `verify`, `jobs`, `queue`,
`transcribe`. Not covered: `analyze`, `warmup`, `preflight`, `gpu_setup`, and the
`worker` loop itself.

### What to add next

Where the remaining risk actually is, in order:

1. **`analyze.py`** — cheapest to test, since `conftest` can already generate
   PDFs, and the most expensive to get wrong.
2. **The worker loop** — `run_one()` with `jobs.start` stubbed, pinning that an
   early stop counts as progress and that a stalled job does not burn attempts.

Neither is large, and both cover code making consequential decisions with nothing
currently checking them.

### One-line summary

Adding features is meaningfully safer in text processing and the queue, about as
risky as before in analysis and the worker, and entirely unchanged for anything
that depends on real OCR.

---

## Q5 — What is CI, why does it matter, and what happens without it?

**As asked:** *"What is CI that we did previously? Why does it matter to do it?
What if we don't do it?"*

**Sharpened:** *From first principles: what problem does Continuous Integration
solve, what did we actually configure, and what goes wrong if it is skipped?*

### Start with the problem

We wrote 79 tests. They only protect anything **if someone actually runs them.**

That sounds trivial, but consider how it fails in practice. It is late, the
change is "obviously fine", and it gets committed unrun. Or the tests are run
but a file was never saved. Or they pass because of something specific to *this*
machine.

So there is a gap between "tests exist" and "tests were run against what was
actually shipped."

### What CI is

**Continuous Integration** is one idea: *every time the code changes, a computer
that is not yours automatically runs the checks.* The name is grander than the
concept.

Both words carry weight:

- **Continuous** — every time, automatically, not when someone remembers.
- **Integration** — it checks the code as committed and combined, not as it
  happens to exist in one person's working folder.

### What was configured here

One file, `.github/workflows/tests.yml`, telling GitHub what to do on every push:

```
1. Get a fresh, empty Linux machine
2. Check out the code from the repository
3. Install the dependencies
4. Run ruff
5. Run pytest (including the slow tests)
```

Red if anything fails. It takes about 30 seconds.

### Why "a fresh machine" is the whole point

This is the part that matters.

A working machine has *history*: packages installed months ago, stray files,
forgotten environment settings. When tests pass there, what has been proven is
"this works **on a machine like this one, with all its accumulated history**."
That is a weaker claim than it feels.

A fresh machine proves something stronger: **the code works from what is
actually in the repository.** A file that was never committed, or a hidden
dependency on something only present locally, simply is not there — and it fails
at once.

### It proved this on its first run

Not hypothetical. The very first CI run **failed**, usefully.

The PDFs live on the local machine but are not in the repository, so CI ran with
**no PDFs at all** — a situation the local machine has never been in. In that
state the app behaves differently: with no document to work on it shows "Choose
a PDF in the sidebar to begin" and stops. The test had assumed a document always
exists.

A local run could never have caught that, because locally there are always PDFs.
The fresh machine found it in 45 seconds.

The tests now describe *both* situations, each skipping where the other applies.
The empty-library case is checked in CI, the has-documents case locally —
together covering more than either environment can alone.

### What happens without it

Nothing dramatic on day one. It decays quietly:

- Tests get skipped when rushed. Usually fine; occasionally not, and the news
  arrives late.
- Code creeps in that only works on one machine. Six months later it will not run
  on a new laptop, or on Colab, and the context for debugging it is long gone.
- Broken code sits in the repository unnoticed, because nothing checks it until
  someone tries to use it.

None of these hurt *today*. They hurt once the context is forgotten — which is
exactly when they are expensive.

### The honest limits

- **It only runs the checks that were written.** Green means "the tests passed",
  not "the code is correct".
- **It cannot test everything.** This one never touches real OCR, which needs the
  models and hours, so it says nothing about whether a transcription is *good*.
- **It does not replace reading the output.** Confirming the Greek came out right
  is still a human job.

A smoke alarm, not a fire inspector: it says something is obviously burning, not
that the building is safe.

### One-line summary

Tests catch mistakes; CI makes sure those tests are actually run, on a clean
machine, every time — which is what turns "we have tests" into "we know the code
works".

---

## Q6 — Why is the worker loop the most important thing left to test?

**As asked:** *"Elaborate in plain English: the biggest remaining gap is the
worker loop — `run_one()` with `jobs.start` stubbed, to pin that an early stop
counts as progress and a stalled job doesn't burn attempts."*

**Sharpened:** *What does the worker loop actually decide, why can those
decisions go wrong silently, and how do you test something whose real work takes
hours?*

### What the worker loop is

The worker is the program that keeps books transcribing overnight without
supervision. Its job is a repeating cycle:

1. Is anything already running? Wait if so.
2. Is there enough disk? Wait if not.
3. Pick the next book that needs work.
4. Run it.
5. Look at what happened and decide what that means.
6. Repeat.

`run_one()` is steps 3–5 — **one turn of that cycle**, on one book.

### Step 5 is where the risk is

Steps 1–4 are mechanical. Step 5 is a *judgement*, and judgements are where bugs
hide.

When a run ends, the worker must decide whether it went well. It cannot simply
check that the program exited normally, because on this machine runs stop early
**all the time** — the disk guard halts them once space runs low, which is
normal here, not exceptional. A book finishes as many short cycles rather than
one long run.

So the judgement is made on **chunks**: did the number of finished pages go up?

| Outcome | Meaning | Action |
|---|---|---|
| chunks went up | real work happened | retry the same book |
| chunks did not move | nothing was achieved | count a failed attempt |
| three failures in a row | something is wrong with this book | mark stalled, move on |

### The two silent failures

**If an early stop were miscounted as failure.** Every normal run here stops
early, so every book would accumulate "failures", reach three, and be abandoned
— while transcribing perfectly well. You would wake to a queue where every book
had been marked stalled, with nothing visibly broken.

**If a genuinely stuck book never counted as failure.** The worker would retry it
forever, and one bad file would block every other book indefinitely. Again
nothing looks wrong: the worker is "busy", just achieving nothing.

Neither throws an error. Neither appears in a log as a crash.

### What "stubbing `jobs.start`" means

`jobs.start()` launches a real transcription — loads the model, runs OCR, takes
hours. A test cannot do that.

So it is replaced with a **stand-in**: *do not launch anything, just write two
chunk files and return*. Then check the worker concluded "progress was made".
Another stand-in writes nothing, and the worker should conclude "no progress"
and count an attempt.

What is being tested is the **judgement**, not the OCR. That runs in
milliseconds instead of hours — and the judgement is the part at risk, since the
OCR itself is marker's code rather than this project's.

### Why this gap matters more than the others

Everything else tested here runs while somebody is watching: you make a change,
run it, look at the result.

The worker runs when nobody is watching — that is its entire purpose. A wrong
judgement there produces no error message. It produces **a night of nothing
happening**, or a night of work thrown away, discovered hours later, having lost
exactly the time the worker exists to save.

It is also the only remaining untested piece that makes decisions on its own.
The other gaps (`warmup`, `preflight`, `gpu_setup`) mostly either work or fail
loudly at startup, where it would be noticed immediately.

### The honest caveat

None of this checks that OCR produces good text — nothing in the suite does. It
checks only that the worker *counts* and *decides* correctly. A narrower claim,
but the one thing currently unguarded.

### One-line summary

The worker's only real decision is "did that run accomplish anything?", it makes
it unattended, and getting it wrong silently wastes the nights it was built to
use — so it is worth testing with the expensive part faked out.
