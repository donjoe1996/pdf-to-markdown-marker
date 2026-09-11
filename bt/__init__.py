"""Transcribe the scanned *Being and Time* PDF (2-page spreads) to Markdown.

Pipeline: split_spreads -> transcribe (marker) -> postprocess.
"""

__all__ = ["preflight", "split_spreads", "transcribe", "postprocess"]
