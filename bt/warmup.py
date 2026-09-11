"""Pre-download every model marker needs, before any timed spawn.

Why this stage exists. marker 2.0 runs its models in separate server
subprocesses and waits for each to report healthy, with a startup timeout of
300s (``OCR_ERROR_SERVER_STARTUP_TIMEOUT`` and friends). On a first run those
subprocesses have to *download* their weights first -- the ocr-error model
alone is 258 MB. On a slow link the download outlives the health check, marker
force-kills the server, and the run dies with a ``SpawnError`` that reads like
a crash rather than what it is:

    SpawnError: ocr_error server failed to become healthy at
    http://127.0.0.1:65071 within 300.0s.

Fetching the weights up front means every later spawn starts from cache and
comes up in seconds. Downloads are resumable and idempotent, so re-running this
after an interrupted attempt costs nothing.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# The Hub resolves large files through its Xet CDN bridge
# (us.aws.cdn.hf.co/xet-bridge-us), which on some networks accepts the
# connection and then delivers nothing -- the download sits at 0 bytes
# indefinitely, with or without an auth token, and hf_transfer does not help.
# Forcing the classic CDN path makes the same file download normally. This is
# the single most important line in this module; without it a first run hangs.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Note: HF_HUB_ENABLE_HF_TRANSFER is deliberately not set. It is deprecated in
# current huggingface_hub (which warns and points at HF_XET_HIGH_PERFORMANCE),
# and it is moot here anyway since Xet is disabled above.


def _s3_models() -> list[str]:
    """Checkpoints served from the datalab CDN, read from surya's settings."""
    from surya.settings import settings

    return [
        settings.OCR_ERROR_MODEL_CHECKPOINT,  # s3://ocr_error_detection/...
        settings.DETECTOR_MODEL_CHECKPOINT,  # s3://text_detection/...
    ]


def warm_s3(checkpoint: str) -> None:
    """Download one ``s3://`` checkpoint into surya's model cache."""
    from surya.common.s3 import check_manifest, download_directory
    from surya.settings import settings

    remote = checkpoint.removeprefix("s3://")
    local = os.path.join(settings.MODEL_CACHE_DIR, remote)
    os.makedirs(local, exist_ok=True)

    if check_manifest(local):
        print(f"  cached   {checkpoint}")
        return
    print(f"  fetching {checkpoint} -> {local}")
    download_directory(remote, local)


def _has_hf_token() -> bool:
    """True if a Hub token is available from the env or a previous login."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _curl_into_hf_cache(repo_id: str, filename: str, attempts: int = 40) -> bool:
    """Fetch one Hub file with curl, writing it into the HF cache layout.

    Fallback for when the hub client stalls at 0 bytes -- which it does on a
    rate-limited link for the ~1.3 GB VLM weights, even with hf_transfer, while
    plain ranged GETs still succeed. Writes the exact layout
    ``hf_hub_download`` expects, so afterwards it is a cache hit:

        models--<org>--<repo>/blobs/<etag>
        models--<org>--<repo>/snapshots/<commit>/<filename> -> ../../blobs/<etag>
        models--<org>--<repo>/refs/main

    Each curl attempt resumes from what is already on disk (``-C -``), so a
    stalled connection costs only the time to notice it.
    """
    import subprocess
    from pathlib import Path

    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.constants import HF_HUB_CACHE

    url = hf_hub_url(repo_id, filename)
    meta = get_hf_file_metadata(url)
    etag = (meta.etag or "").strip('"')
    if not etag or not meta.commit_hash or not meta.size:
        return False

    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    blob = repo_dir / "blobs" / etag
    snapshot = repo_dir / "snapshots" / meta.commit_hash / filename
    partial = blob.with_suffix(blob.suffix + ".partial")

    if blob.exists() and blob.stat().st_size == meta.size:
        print(f"  cached   {repo_id}/{filename}")
    else:
        blob.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, attempts + 1):
            have = partial.stat().st_size if partial.exists() else 0
            if have >= meta.size:
                break
            print(
                f"  fetching {filename}: {have / 1e6:.0f}/{meta.size / 1e6:.0f} MB "
                f"(attempt {attempt}/{attempts})"
            )
            subprocess.run(
                [
                    "curl", "-sL", "-C", "-",
                    # Give up on a connection that delivers nothing, so the
                    # retry loop can reconnect instead of hanging.
                    "--speed-limit", "1024", "--speed-time", "30",
                    "--retry", "3", "--retry-delay", "5",
                    "-o", str(partial), meta.location or url,
                ],
                check=False,
            )
        if not partial.exists() or partial.stat().st_size != meta.size:
            got = partial.stat().st_size if partial.exists() else 0
            print(f"  INCOMPLETE {filename}: {got}/{meta.size} bytes")
            return False
        partial.rename(blob)

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(meta.commit_hash)
    return True


def warm_gguf() -> None:
    """Download the surya VLM weights used by the llama.cpp backend."""
    from huggingface_hub import hf_hub_download
    from surya.settings import settings

    authed = _has_hf_token()
    if not authed:
        print(
            "  note: no HF token found. Unauthenticated downloads are rate-limited\n"
            "        and the hub client tends to stall on the ~1.3 GB weights.\n"
            "        `huggingface-cli login` makes this much faster."
        )

    for filename in (settings.SURYA_GGUF_MODEL_FILE, settings.SURYA_GGUF_MMPROJ_FILE):
        if authed:
            # With a token the official client gets full-rate downloads, so use
            # it directly; curl stays as the fallback.
            try:
                hf_hub_download(settings.SURYA_GGUF_REPO, filename)
                print(f"  done     {filename}")
                continue
            except Exception as exc:
                print(f"  hub client failed ({type(exc).__name__}); trying curl")
        # Unauthenticated, the hub client does not fail on a throttled link --
        # it hangs at 0 bytes indefinitely, so a try/except would never fire.
        # curl's --speed-time turns that same stall into a fast retry.
        if _curl_into_hf_cache(settings.SURYA_GGUF_REPO, filename):
            continue
        print("  curl fallback incomplete; trying the hub client")
        hf_hub_download(settings.SURYA_GGUF_REPO, filename)


def warm_layout() -> None:
    """Download the rf-detr layout/order model used by ``mode='fast'``."""
    from huggingface_hub import snapshot_download
    from surya.settings import settings

    repo = settings.FAST_LAYOUT_MODEL_CHECKPOINT.removeprefix("hf://").split("/")
    repo_id = "/".join(repo[:2])
    print(f"  fetching {repo_id}")
    snapshot_download(repo_id)


def warm_all(include_vlm: bool = True) -> None:
    print("Pre-downloading models (resumable; re-runs are free):")
    for ckpt in _s3_models():
        warm_s3(ckpt)
    warm_layout()
    if include_vlm:
        warm_gguf()
    print("All models cached.")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Pre-download marker/surya models.")
    ap.add_argument(
        "--no-vlm",
        action="store_true",
        help="skip the ~1.5 GB VLM weights (only the small local models)",
    )
    args = ap.parse_args(argv)
    warm_all(include_vlm=not args.no_vlm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
