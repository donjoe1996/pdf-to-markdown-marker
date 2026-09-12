"""Environment checks that must pass before a long OCR run.

marker 2.0 no longer runs its models in-process: it auto-spawns a surya VLM
inference server, which on Apple Silicon is ``llama-server`` from llama.cpp.
marker does not install that for you, so a missing binary only surfaces hours
in. Everything expensive is gated behind these checks.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass

# venv (~2 GB incl. torch) + surya GGUF pair ~1.5 GB + rf-detr layout ~135 MB,
# plus headroom for the split PDF and chunk output.
REQUIRED_FREE_GB = 4.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 1e9


def check_python() -> Check:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10) and (major, minor) < (3, 14)
    return Check(
        "python",
        ok,
        f"{sys.version.split()[0]} at {sys.executable}"
        + ("" if ok else "  -- need >=3.10,<3.14 (torch has no 3.14 wheels)"),
    )


def check_disk() -> Check:
    gb = free_gb()
    ok = gb >= REQUIRED_FREE_GB
    detail = f"{gb:.1f} GB free (need ~{REQUIRED_FREE_GB:.0f} GB)"
    if not ok:
        detail += (
            "\n      Reclaimable candidates (nothing is deleted for you):\n"
            "        du -sh ~/Library/Caches ~/anaconda3/pkgs ~/.cache/uv\n"
            "        uv cache clean ; pip cache purge ; conda clean --all"
        )
    return Check("disk", ok, detail)


def check_marker() -> Check:
    spec = importlib.util.find_spec("marker")
    if spec is None:
        return Check("marker-pdf", False, "not importable -- run: uv sync")
    try:
        from importlib.metadata import version

        v = version("marker-pdf")
    except Exception:
        v = "unknown"
    return Check("marker-pdf", True, f"version {v}")


def check_llama_server() -> Check:
    """The surya VLM backend on Apple Silicon / CPU / CUDA."""
    # LLAMA_CPP_BINARY is how surya is pointed at a non-PATH build -- which is
    # the normal case on Colab, where llama.cpp has to be compiled with CUDA
    # because upstream ships no Linux CUDA binaries.
    exe = os.environ.get("LLAMA_CPP_BINARY") or shutil.which("llama-server")
    if exe and os.path.sep in exe and not os.access(exe, os.X_OK):
        return Check("llama-server", False, f"{exe} is not executable")
    if not exe:
        return Check(
            "llama-server",
            False,
            "not on PATH -- marker 2.0 spawns it to serve the surya VLM.\n"
            "      Install with:  brew install llama.cpp",
        )
    # Deliberately no `--version` probe: llama-server starts serving rather
    # than printing a version and exiting, so probing it just costs a timeout.
    return Check("llama-server", True, exe)


def check_torch_device() -> Check:
    """Non-fatal: MPS just makes the small local models faster."""
    try:
        import torch
    except Exception as exc:
        return Check("torch", False, f"import failed: {exc}")
    if torch.backends.mps.is_available():
        dev = "mps (Apple Silicon)"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu -- will be slow"
    return Check("torch", True, f"{torch.__version__}, device {dev}", fatal=False)


def run(require_marker: bool = True) -> list[Check]:
    checks = [check_python(), check_disk(), check_llama_server()]
    if require_marker:
        checks += [check_marker(), check_torch_device()]
    return checks


def report(checks: list[Check]) -> bool:
    """Print results; return True if no fatal check failed."""
    failed = False
    for c in checks:
        if c.ok:
            mark = "ok  "
        elif c.fatal:
            mark = "FAIL"
            failed = True
        else:
            mark = "warn"
        print(f"  [{mark}] {c.name:<13} {c.detail}")
    return not failed


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Preflight checks for the OCR run.")
    ap.add_argument(
        "--no-marker",
        action="store_true",
        help="skip marker/torch checks (for the split stage, which needs neither)",
    )
    args = ap.parse_args(argv)

    print("Preflight:")
    ok = report(run(require_marker=not args.no_marker))
    print("\nAll required checks passed." if ok else "\nFix the FAIL items above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
