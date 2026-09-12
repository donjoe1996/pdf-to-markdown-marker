---
name: adding-a-feature
description: "**[REQUIRED]** Load BEFORE writing any new feature, module, behaviour change, refactor or bug fix in this repository. Encodes how work is added here safely: find the seam, prove the test fails first, keep the suite fast and hermetic, never disturb a running transcription. Triggers: add a feature, new feature, new module, new stage, implement, build, extend, refactor, fix a bug, change behaviour, improve, optimise, bt/, app.py, pipeline, worker, queue."
---

# Adding a feature to this repository

A blueprint, not a style guide. Follow the steps in order; each exists because
skipping it has already cost time here.

## Why this exists

This project does long, expensive work — a single book takes hours — and often
has a worker processing books unattended while code is being edited. The
failures that hurt are not crashes. They are **silent**: output that looks
perfectly fine and is quietly wrong, discovered days later after hours of
compute.

Everything below is aimed at that.

## The workflow

```
0. Check what is running        -> do not break work in progress
1. Find the seam                -> reuse before adding
2. Pin current behaviour        -> characterize before changing
3. Write the failing test       -> red before green
4. Build it
5. Verify: ruff -> pytest -> isolation
6. Commit the reasoning, not just the change
```

---

### 0. Check what is running — first, every time

```bash
pgrep -f "bt.worker" >/dev/null && echo "worker RUNNING"
pgrep -f "m bt.run"   # an active transcription
df -h /Users/yogi | tail -1
```

**Editing `bt/*.py` reaches a running worker within ~30 seconds.** It launches a
fresh `python -m bt.run` each cycle and calls `queue.survey()` on every loop, so
modules are re-imported from disk continuously. A syntax error mid-edit crashes
it.

If the change is large or risky and a worker is running, either stop it first or
work in a git worktree (a branch alone does **not** isolate — branches share one
working tree).

Never start a second transcription alongside a running one: two `llama-server`
processes push this machine into swap, where throughput collapses. See
`QnA.md` Q2.

---

### 1. Find the seam — reuse before adding

Read before writing. The common seams:

| Adding… | Goes in | Notes |
|---|---|---|
| a text transform | `postprocess.process()` | strongest test cover; golden test will show the diff |
| a pipeline stage | `run.py` orchestration | keep each stage independently runnable |
| queue behaviour | `queue.py` | state is **derived from disk**, see below |
| process/lock handling | `jobs.py` | has global side effects — read the hazards section |
| a document signal | `analyze.py` | currently untested; add tests with it |
| GUI | `app.py` | **also load the `developing-with-streamlit` skill** |

Check whether the thing already exists. Examples that were nearly duplicated
here: `queue.resolve_out_dir()` (the GUI and worker must agree on where chunks
live), `verify.RUN_ON` (reused as a text-damage metric), `find_gutter()`'s band
width (already computed, just unreturned).

---

### 2. Pin current behaviour before changing it

If the code works and has no test, **write the characterization test first** —
one that describes what it does today, then change it. Otherwise "did I break
it?" has no answer.

This is why most tests here name a real defect in their docstring. A test whose
purpose is recorded survives a refactor; one that merely asserts something true
gets deleted when inconvenient.

---

### 3. Write the failing test first

**For a bug fix:** reproduce it red *before* fixing. A test that has never
failed is not known to test anything.

**For a feature:** write the test for the behaviour you intend, watch it fail
for the right reason, then implement.

**For a refactor:** the existing suite is the test. Run it before and after.

Verify the net still works when touching test-critical code:

```bash
# reintroduce a real bug, confirm red, restore
pytest tests/test_postprocess.py
```

#### Test rules in this repo

**Hermetic — never touch live state.** Several functions have teeth:

| Function | Side effect |
|---|---|
| `jobs.stop_pids()` | global `pkill -f llama-server` — kills a live OCR server |
| `jobs.clear_stale_sentinels()` | deletes from the real `~/.cache/datalab/surya` |
| `jobs.claim/release_worker_lock()` | writes the live `output/.worker.lock` |

The autouse `isolate` fixture in `tests/conftest.py` redirects all of it at
`tmp_path`. When adding tests:

- Never signal a pid the test did not spawn itself.
- Call `stop_pids(..., kill_inference=False)`.
- Never write under the repo's `output/`.

**Fixtures are synthetic and generated.** Real output is book text — it cannot
be committed or used as a golden file. `conftest.py` builds marker-shaped
markdown (`{N}----` separators, `<sup>` markers, running heads) with placeholder
prose, and draws test PDFs with PyMuPDF. Add to those builders rather than
committing binary fixtures.

**Keep the default suite fast.** It runs in ~1s and must stay that way — a slow
suite gets skipped, and a skipped suite protects nothing. Anything that boots
Streamlit or reads real PDFs gets `@pytest.mark.slow` and runs in CI.

**Golden changes are deliberate.** If output *should* change, run
`pytest --golden-update` and **review the diff**. Never regenerate to make a
failure go away without reading what moved.

---

### 4. Build it

Project-specific traps, all of which have bitten:

- **Keep `marker`/`torch` imports function-local** in `transcribe.py` and
  `warmup.py`. Those modules set env vars at import time that must land before
  torch loads; hoisting an import silently breaks it.
- **Do not add a progress field anywhere.** Chunk files *are* the progress
  record. A parallel store can only drift from them.
- **Completion is `chunks_done >= chunks_total`** — never "the output file
  exists". `run.py` writes a finished-looking `.md` even from a part-done run.
- **"Process alive" is not "work happening."** A job waiting on a dead model
  server looks perfectly healthy. Judge liveness by *output*.
- **Guards must fail loudly.** Prefer raising or stopping cleanly over silently
  continuing. `concatenate()` takes explicit bounds for exactly this reason.

---

### 5. Verify

```bash
uv run ruff check .      # layer 0 — run BEFORE tests
uv run pytest            # fast suite, ~1s
uv run pytest -m ""      # everything, as CI runs it
```

**Run `ruff` first.** Two real bugs here were names used but never imported;
they compile cleanly and fail only when that line finally executes — hours in,
for one of them.

If a worker was running, confirm it survived:

```bash
pgrep -f "bt.worker" && ls output/*/chunks/*.md | wc -l
```

For anything touching OCR config, the suite is **not** enough — it never imports
marker. Run a short real range and check the output against the page image.

---

### 6. Commit the reasoning

Record *why*, not just what. Specifically: what failure the change prevents,
what was measured, and what was rejected and on what evidence. A future reader
(often you) needs the reasoning more than the diff, which git already has.

---

## What this suite does not cover

Say so rather than implying more safety than exists:

- **OCR quality.** Green means the plumbing is intact, not that the output is
  right. Use `bt/verify.py` against real output, and read a page.
- **marker API changes.** Never imported in the fast suite.
- **`analyze.py`, `warmup.py`, `preflight.py`, `gpu_setup.py` and the worker
  loop** are untested. If the change lands there, add tests with it.

## Scaling toward production

Principles that earned their place here, worth keeping as the program grows:

1. **Derive state, don't track it.** Anything stored twice will disagree.
2. **Make failure loud and early.** Silent wrongness is the expensive kind.
3. **Prove the safety net catches things** by making it fail on purpose.
4. **Keep the fast path fast** so the checks are actually run.
5. **Isolate tests from live state** so verification is never a risk in itself.
6. **Measure before optimising.** DPI looked like the obvious speed lever here
   and turned out to change nothing — the measurement is in `CLAUDE.md`.
