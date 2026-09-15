"""Translation: structure survives the model, and a run survives a failure.

The model is never called here. Every test passes a fake translator, which is
the point of the seam: the risky part of this stage is not the API call but
what happens to the document *around* it -- page anchors, footnote ids, resume
boundaries -- and all of that is testable offline in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bt import translate
from bt.postprocess import PAGE_ANCHOR, PAGE_ANCHOR_FMT


def upper(text: str, target: str) -> str:
    """A translator that is obviously not the identity function."""
    del target
    return text.upper()


def source(md_path, pages: list[str]) -> None:
    md_path.write_text(
        "\n\n".join(
            f"{PAGE_ANCHOR_FMT.format(n=i)}\n\n{body}" for i, body in enumerate(pages)
        ),
        encoding="utf-8",
    )


@pytest.fixture
def run(tmp_path):
    """Translate a list of page bodies; return (output text, stats)."""

    def go(pages: list[str], translator=upper, **kw):
        src = tmp_path / "book.md"
        source(src, pages)
        out = tmp_path / "book.english.md"
        _, stats = translate.translate_document(
            src,
            out,
            tmp_path / "tchunks",
            translator=translator,
            **kw,
        )
        return out.read_text(encoding="utf-8"), stats

    return go


# --------------------------------------------------------------------------
# the structural guarantee
# --------------------------------------------------------------------------
def test_page_anchors_are_re_emitted_not_translated(run):
    """The model never sees an anchor, so it cannot lose or renumber one.

    ``verify.check_not_embedded_layer`` aligns output pages to PDF pages by
    these numbers, and every passage is traced back to a page through them. A
    model asked to "preserve" them would drop one eventually, and the loss
    would be invisible -- the translation would still read perfectly.
    """
    seen: list[str] = []

    def spy(text: str, target: str) -> str:
        seen.append(text)
        return upper(text, target)

    text, stats = run(["uno", "dos", "tres"], translator=spy)

    assert [int(m.group(1)) for m in PAGE_ANCHOR.finditer(text)] == [0, 1, 2]
    assert not any("<!-- page" in sent for sent in seen)
    assert stats.pages_translated == 3


def test_page_numbers_come_from_the_source_not_a_counter(tmp_path):
    """A part-run or a page range leaves gaps; the numbers must still match."""
    src = tmp_path / "book.md"
    src.write_text(
        "{40}" + "-" * 40 + "\n\nquarante\n\n{41}" + "-" * 40 + "\n\nquarante et un\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.md"
    translate.translate_document(src, out, tmp_path / "c", translator=upper)

    assert [int(m.group(1)) for m in PAGE_ANCHOR.finditer(out.read_text())] == [40, 41]


def test_unpaginated_source_gains_no_anchor(tmp_path):
    """Nothing to align by means nothing to invent."""
    src = tmp_path / "note.md"
    src.write_text("a single page with no separators\n", encoding="utf-8")
    out = tmp_path / "out.md"
    translate.translate_document(src, out, tmp_path / "c", translator=upper)

    assert "<!-- page" not in out.read_text()
    assert "NO SEPARATORS" in out.read_text()


def test_blank_pages_are_not_sent_to_the_model(run):
    """Six halves of the Heidegger scan are blank; an empty prompt buys nothing.

    Sending one costs a request and invites the model to fill the silence.
    """
    calls: list[str] = []

    def spy(text: str, target: str) -> str:
        calls.append(text)
        return upper(text, target)

    _, stats = run(["real text", "   \n\n  ", "more text"], translator=spy)

    assert len(calls) == 2
    assert stats.pages_translated == 2
    assert stats.pages_blank == 1


# --------------------------------------------------------------------------
# what the model hands back
# --------------------------------------------------------------------------
def test_code_fence_is_stripped(run):
    """Models wrap Markdown in a fence when asked for Markdown."""
    text, _ = run(["hola"], translator=lambda t, lang: f"```markdown\n{t}\n```")
    assert "```" not in text
    assert "hola" in text


def test_footnote_drift_is_recorded(run):
    """A renumbered footnote is silent damage: it still renders, wrongly.

    Not fatal -- a page can legitimately have none -- but it must be counted
    and reported rather than discovered months later in the finished book.
    """
    text, stats = run(
        ["see the note[^p0-1]\n\n[^p0-1]: the note"],
        translator=lambda t, lang: t.replace("p0-1", "1"),
    )
    assert stats.markup_drift == [0]
    assert "[^1]" in text  # kept as the model returned it; flagged, not patched


def test_intact_footnotes_are_not_flagged(run):
    """Ids are compared verbatim -- even a case change would break the link."""
    text, stats = run(
        ["see the note[^p0-1]\n\n[^p0-1]: the note"],
        translator=lambda t, lang: t.replace("the note", "la nota"),
    )
    assert stats.markup_drift == []
    assert text.count("[^p0-1]") == 2


# --------------------------------------------------------------------------
# resume, and failure part-way through
# --------------------------------------------------------------------------
def test_resume_skips_chunks_already_translated(tmp_path):
    src = tmp_path / "book.md"
    source(src, ["one", "two", "three", "four"])
    chunks = tmp_path / "c"

    translate.translate_document(
        src, tmp_path / "out.md", chunks, translator=upper, pages_per_chunk=2
    )

    calls: list[str] = []

    def spy(text: str, target: str) -> str:
        calls.append(text)
        return upper(text, target)

    _, stats = translate.translate_document(
        src, tmp_path / "out.md", chunks, translator=spy, pages_per_chunk=2
    )
    assert calls == []
    assert stats.chunks_done == 2 and stats.chunks_pending == 0


def test_a_failure_keeps_the_finished_chunks_and_stops(tmp_path):
    """A rate limit at page 300 must not cost the first 299 pages.

    It must also not pretend to have succeeded: the run reports the error and
    the partial output, and re-running picks up at the failed chunk.
    """
    src = tmp_path / "book.md"
    source(src, ["one", "two", "three", "four"])
    chunks = tmp_path / "c"

    def fails_on_three(text: str, target: str) -> str:
        if "three" in text:
            raise RuntimeError("429 rate limited")
        return upper(text, target)

    out = tmp_path / "out.md"
    _, stats = translate.translate_document(
        src, out, chunks, translator=fails_on_three, pages_per_chunk=2
    )

    assert stats.stopped_early and "429" in stats.error
    assert sorted(p.name for p in chunks.iterdir()) == ["0000-0001.md"]
    assert "ONE" in out.read_text() and "THREE" not in out.read_text()


def test_a_failed_chunk_leaves_no_partial_file(tmp_path):
    """REGRESSION shape: a half-written chunk looks complete on resume.

    Same hazard as the OCR chunks -- the file is the only progress record, so
    it may exist only once it is whole.
    """
    src = tmp_path / "book.md"
    source(src, ["one", "two"])
    chunks = tmp_path / "c"

    def boom(text: str, target: str) -> str:
        raise RuntimeError("no")

    translate.translate_document(
        src, tmp_path / "out.md", chunks, translator=boom, pages_per_chunk=1
    )
    assert list(chunks.glob("*.partial")) == []
    assert list(chunks.glob("*.md")) == []


def test_changing_chunk_size_does_not_duplicate_the_book(tmp_path):
    """The `concatenate` trap, at this layer: stale bounds must be ignored."""
    src = tmp_path / "book.md"
    source(src, ["one", "two", "three", "four"])
    chunks = tmp_path / "c"

    translate.translate_document(
        src, tmp_path / "out.md", chunks, translator=upper, pages_per_chunk=4
    )
    out = tmp_path / "out.md"
    translate.translate_document(
        src, out, chunks, translator=upper, pages_per_chunk=2
    )

    assert out.read_text().count("ONE") == 1


# --------------------------------------------------------------------------
# the prompt that reaches the model
# --------------------------------------------------------------------------
def test_target_language_reaches_the_translator(run):
    seen: list[str] = []
    run(["hola"], translator=lambda t, lang: seen.append(lang) or t, target="Indonesian")
    assert seen == ["Indonesian"]


def test_system_prompt_names_the_target_language():
    prompt = translate.system_prompt("Indonesian")
    assert "Indonesian" in prompt
    assert "[^" in prompt  # footnote ids are called out explicitly


def test_output_name_keeps_a_dotted_filename_whole():
    """`with_suffix` strips the last dot segment, not the extension.

    A document called `Calakmul v1.2.pdf` produced `Calakmul v1.english.md` --
    a different book's name, quietly, and a second version would overwrite it.
    """
    assert translate.default_out_path(Path("out/book.md"), "English").name == (
        "book.english.md"
    )
    assert translate.default_out_path(Path("out/Calakmul v1.2.md"), "English").name == (
        "Calakmul v1.2.english.md"
    )
    assert translate.default_out_path(Path("out/b.md"), "Bahasa Indonesia").name == (
        "b.bahasa-indonesia.md"
    )
