---
name: designing-the-gui
description: "**[REQUIRED]** Load BEFORE changing the layout, order, labels, grouping or wording of the GUI (app.py), and whenever someone says a screen is confusing, cluttered, badly ordered, hard to find, or 'not intuitive'. A first-principles procedure for UX review and redesign in this app: name the task, inventory every element, classify it, place it by rule, verify with AppTest and a screenshot. Load together with developing-with-streamlit (the API) and adding-a-feature (the test discipline). Triggers: UI, UX, layout, menu order, intuitive, confusing, cluttered, friction, redesign, reorganise, tab, sidebar, button placement, label, wording, app.py."
---

# Designing the GUI

How to decide **where things go and what they are called** in `app.py`.
`developing-with-streamlit` says *how* to draw a widget, and this skill says
*whether it belongs on screen, where, and in what order*.

## Start from first principles

A screen exists to let one person finish one task with the least thinking. Each
time they open it, they need answers to three questions, in this order:

1. **Where am I, and on what?** That means the document, its output folder and
   the current step.
2. **What is the state?** Is a job running or stopped, is the output done or
   failed?
3. **What do I do next?** One obvious primary action.

Every element on screen either answers one of these questions or competes with
them. Most "this feels unintuitive" complaints come down to one of three causes:

- **Order does not follow the task.** The user's next step sits below something
  they don't need yet.
- **Too many equal-looking things.** Choice time grows with the number of
  options (Hick's law, Proctor & Schneider 2018), and anything on the first
  display tells the user "this matters" (NN/g, progressive disclosure).
- **State is hidden or ambiguous.** The user can't tell what the system is
  doing, which breaks Nielsen heuristic #1 (visibility of system status).

## The procedure (do it in this order)

### 1. Write the task as a sentence sequence, before touching code

Example for the Transcribe tab: *"I check the guessed settings → I glance at a
page to confirm the guess → I start → I watch progress."* The screen's top-to-
bottom (and left-to-right) order must match this sentence. If a step needs
scrolling past something from a later step, the order is wrong.

### 2. Inventory every element on the screen

List each widget, metric, caption and message. Take a screenshot first (see
*Verify* below). Most redesigns fail because the designer reasons about
the code and forgets what the screen actually looks like.

### 3. Classify each element. One class only.

| Class | Meaning | Where it goes |
|---|---|---|
| **Primary action** | The one thing this screen exists for (Start, Download, Translate) | Visible without scrolling, `type="primary"`, **one per screen section** |
| **Decision** | Input the user must actually think about on most runs | Top of the section, in task order, defaults pre-filled from analysis |
| **State** | Running / stopped / failed / done | Near the action it concerns; colour only for state |
| **Rare setting** | Needed on few runs (DPI, chunk size, page range, output folder) | Collapsed `st.expander` labelled "Advanced …"; never removed |
| **Explanation** | Why a default was chosen, how a feature works | `help=` tooltip or a collapsed expander, not a paragraph of caption |
| **Diagnostic** | Logs, shell commands, per-check detail | Collapsed expander; expand automatically only while it is live or failing |
| **Destructive** | Deletes work (Clear progress) | Secondary styling, **confirmation dialog** naming what will be lost |

If an element fits no class, delete it. If it fits two, it is doing two jobs, so
split it.

### 4. Place by rule

- **Scope separation (already established, keep it).** The *machine* (worker,
  queue, jobs) lives in the sidebar (read-only) and the Queue tab (controls).
  The *document* lives in the header and the Transcribe/Result/Translate tabs.
  Never put a machine control inside a document tab, or the reverse.
- **Primary action next to its decision.** Put Start directly under the
  settings it launches, in the same column, not after a tall preview.
- **Outcome first on result screens.** Show a summary of the outcome (one line:
  "All 5 checks passed" / "1 of 5 checks failed"), then the action (Download),
  then the details.
- **Progressive disclosure is one level deep.** Main screen plus one "Advanced"
  expander. Nested expanders and second-level advanced menus hide things for
  good (NN/g warns against multiple levels).
- **Coupled settings sit together and are labelled by outcome, not mechanism.**
  For example, label the engine choice by *what it is for* ("Docling, journals
  and tables") rather than only by tool name.
- **Recognition over recall.** Show the analysis verdict the defaults came
  from, next to the defaults. Don't make the user remember it from the header.
- **Destructive actions need confirmation.** Use `@st.dialog`, name the count
  of what will be lost, and make the cancel path the easy one.
- **Header is compact.** Document facts go in one line of badges or captions,
  not four large metric cards. Big cards push the task below the fold on every
  tab, including tabs where those facts don't matter.

### 5. Words

- Sentence case; verbs on buttons ("Start transcription", not "OK").
- Say the same thing with the same word everywhere ("chunk", "page",
  "document"). Don't call the same concept by two names (Nielsen #4,
  consistency).
- Messages say what happened **and what to do** ("Stopped to protect disk
  space. Press Resume, finished chunks are kept.").
- A caption longer than two lines is documentation. Move it to `help=` or an
  expander.

## Project rules that constrain any redesign

These are load-bearing. Read `CLAUDE.md` → *The GUI* for the reasons.

- **Tabs, not `st.navigation` pages**: AppTest must execute every branch in
  one run.
- **The document picker stays first among sidebar selectboxes**
  (`at.sidebar.selectbox[0]`), and the output folder stays the first sidebar
  text input (`at.sidebar.text_input[0]`). Both can move into an expander,
  because AppTest finds them there.
- **Tab names `Transcribe`, `Result`, `Translate`, `Queue` are addressed by
  tests.** Rename only together with the tests, and only for a reason.
- **Labels are API.** Tests find buttons by label ("Start worker", "Download
  Markdown", "Translate to …", "page audit"). Changing a label means changing
  its test in the same commit, deliberately.
- **No `icon=` on an `st.expander` that a test looks inside.** `AppTest`
  (Streamlit 1.63) stops listing an expander under `.expander` once it has an
  icon, so `next(e for e in tab.expander if "Advanced" in e.label)` finds
  nothing. Every assertion scoped to that expander then fails, or worse,
  silently checks nothing. Found while moving the rare settings into "Advanced".
- **Colour is reserved for state** (`.streamlit/config.toml`). Do not add
  decorative colour, CSS, or emoji.
- **Long work never runs in the script.** A button only launches a detached
  job, and the page re-reads state from disk.

## Verify

1. `uv run ruff check .` and then `uv run pytest -m slow tests/test_app.py`.
2. Add an AppTest that pins the **behaviour** the redesign promised, not the
   pixels. Examples: "one click on Clear progress deletes nothing", "DPI is
   inside the Advanced expander", "the primary button is in the same tab as the
   settings".
3. Take before/after screenshots and look at them:
   `bash .claude/skills/designing-the-gui/scripts/screenshot.sh before` (and
   then `after`), then open the PNGs. It seeds a demo library in a scratch copy
   of the app, so it never touches `output/` or a running worker.
4. Above the fold, at 1400×1000, can you answer the three questions
   (where am I, what is the state, what next) on every tab? If not, go back to
   step 3.

## Anti-patterns already met in this app

| Seen | Why it hurt | Fix applied |
|---|---|---|
| One long scroll column for every step | No way to tell where one section ended | Tabs per step plus a status strip |
| Start button below a tall page preview | The primary action was below the fold | Run card moved under the settings |
| DPI / chunk size / page range shown on every run | Rare settings looked as important as split/OCR, even though DPI is measured not to change speed | "Advanced" expander |
| Four big fact cards above every tab | Pushed every tab's content down, including Queue, where they are irrelevant | One line of badges |
| Five ok/FAIL cards before the Download button | The outcome took five glances, and the action sat at the bottom | Summary line → download → details |
| One click on "Clear progress" deleted hours of chunks | No error prevention on an irreversible action | Confirmation dialog naming the count |
| Shell commands always visible on the Queue tab | Diagnostic noise next to the one control | Collapsed "Shell commands" expander |

## Sources

- Nielsen, J. *10 Usability Heuristics for User Interface Design*. Nielsen
  Norman Group. https://www.nngroup.com/articles/ten-usability-heuristics/
- Nielsen, J. *Progressive Disclosure*. Nielsen Norman Group.
  https://www.nngroup.com/articles/progressive-disclosure/
- NN/g. *10 Usability Heuristics Applied to Complex Applications*.
  https://www.nngroup.com/articles/usability-heuristics-complex-applications/
- GOV.UK Service Manual. *Structuring forms* (one thing per page; split by
  decision). https://www.gov.uk/service-manual/design/form-structure
- GOV.UK Design System. *Question pages*.
  https://design-system.service.gov.uk/patterns/question-pages/
- Proctor, R. W. & Schneider, D. W. (2018). *Hick's law for choice reaction
  time: A review*. Quarterly Journal of Experimental Psychology (Purdue).
  https://web.ics.purdue.edu/~dws/pubs/ProctorSchneider_2018_QJEP.pdf
