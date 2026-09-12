"""Set up the surya VLM backend on an NVIDIA GPU (Google Colab and similar).

Why this module exists. marker 2.0 has two inference backends:

* **vllm** -- the default on NVIDIA, but it spawns the ``vllm/vllm-openai``
  **Docker image**. Colab has no Docker daemon, so this path is unavailable.
* **llamacpp** -- used on macOS/CPU, and the one this project already runs. Its
  spawn command passes ``-ngl 99`` (``LLAMA_CPP_NGL``), which offloads every
  layer to **CUDA on Linux** exactly as it offloads to Metal on a Mac.

So the GPU route is the llama.cpp backend, forced on, pointed at a CUDA build.
That reuses the code path already validated on Apple Silicon rather than
introducing a second one.

The catch: upstream llama.cpp publishes CUDA binaries for **Windows only** --
the Linux assets are x64/arm64/vulkan/rocm/sycl, with no CUDA variant. So the
server has to be compiled here. That takes several minutes, which is why the
result is cached: point ``cache_dir`` at Google Drive and later sessions reuse
the binary instead of rebuilding it.

Typical use in a notebook::

    from bt import gpu_setup
    gpu_setup.report()
    gpu_setup.configure(cache_dir="/content/drive/MyDrive/bt-cache")
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"

# One OCR request needs ~12k ctx (surya's SURYA_INFERENCE_CTX_PER_SLOT), and
# llama-server allocates ctx x slots of KV cache. The GGUF weights are ~1.5 GB,
# so on a 16 GB T4 there is ample room for several concurrent pages -- which is
# the main reason a GPU beats the Mac here, beyond raw bandwidth.
PARALLEL_BY_VRAM_GB = [(40, 8), (20, 6), (14, 4), (8, 2)]


def gpu_info() -> dict:
    """What GPU (if any) torch can see."""
    try:
        import torch
    except Exception as exc:
        return {"available": False, "reason": f"torch import failed: {exc}"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch reports no CUDA device"}

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "available": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "capability": f"{major}.{minor}",
        "arch": f"{major}{minor}",  # CMAKE_CUDA_ARCHITECTURES value
    }


def suggest_parallel(vram_gb: float) -> int:
    for floor, slots in PARALLEL_BY_VRAM_GB:
        if vram_gb >= floor:
            return slots
    return 1


def build_llama_server(
    cache_dir: str | Path,
    arch: str,
    jobs: int | None = None,
    force: bool = False,
) -> Path:
    """Compile ``llama-server`` with CUDA, caching the binary in ``cache_dir``.

    Returns the binary path. If a cached build is already there, it is reused
    without rebuilding -- put ``cache_dir`` on Drive to survive session resets.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / "llama-server"

    if cached.exists() and not force:
        cached.chmod(0o755)
        print(f"Reusing cached llama-server: {cached}")
        return cached

    if not shutil.which("nvcc") and not Path("/usr/local/cuda/bin/nvcc").exists():
        raise RuntimeError(
            "nvcc not found -- a CUDA toolkit is required to build llama.cpp. "
            "On Colab, make sure the runtime type is set to a GPU."
        )

    src = Path("/content/llama.cpp") if Path("/content").is_dir() else Path("llama.cpp")
    if not src.exists():
        print(f"Cloning llama.cpp into {src} ...")
        _run(["git", "clone", "--depth", "1", LLAMA_REPO, str(src)])

    build = src / "build"
    jobs = jobs or (os.cpu_count() or 2)
    print(f"Building llama-server for CUDA arch {arch} with {jobs} jobs "
          "(several minutes; cached afterwards) ...")
    _run([
        "cmake", "-S", str(src), "-B", str(build),
        "-DGGML_CUDA=ON",
        "-DLLAMA_CURL=OFF",          # avoids a libcurl dev dependency
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",  # one arch = much faster build
    ])
    _run([
        "cmake", "--build", str(build),
        "--config", "Release", "-j", str(jobs),
        "--target", "llama-server",
    ])

    built = build / "bin" / "llama-server"
    if not built.exists():
        raise RuntimeError(f"build finished but {built} is missing")
    shutil.copy2(built, cached)
    cached.chmod(0o755)
    print(f"Built and cached: {cached}")
    return cached


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-25:]
        raise RuntimeError(
            f"command failed: {' '.join(cmd[:3])}...\n" + "\n".join(tail)
        )


def configure(
    cache_dir: str | Path = "/content/llama-cache",
    parallel: int | None = None,
    binary: str | Path | None = None,
) -> dict:
    """Point surya at a CUDA llama-server and tune concurrency. Returns the env set."""
    info = gpu_info()
    if not info["available"]:
        raise RuntimeError(
            f"No CUDA GPU visible ({info['reason']}).\n"
            "On Colab: Runtime > Change runtime type > GPU (T4), then rerun."
        )

    exe = Path(binary) if binary else build_llama_server(cache_dir, info["arch"])
    slots = parallel or suggest_parallel(info["vram_gb"])

    env = {
        # Force the non-Docker backend; the NVIDIA default would want Docker.
        "SURYA_INFERENCE_BACKEND": "llamacpp",
        "LLAMA_CPP_BINARY": str(exe),
        "LLAMA_CPP_NGL": "99",  # all layers on GPU
        "SURYA_INFERENCE_PARALLEL": str(slots),
        "SURYA_INFERENCE_KEEP_ALIVE": "1",
        "HF_HUB_DISABLE_XET": "1",  # see bt/warmup.py -- stalls downloads at 0 bytes
        "TORCH_DEVICE": "cuda",
    }
    os.environ.update(env)
    print(
        f"GPU: {info['name']} ({info['vram_gb']} GB, sm_{info['arch']})\n"
        f"llama-server: {exe}\nparallel pages: {slots}"
    )
    return env


def report() -> None:
    info = gpu_info()
    if not info["available"]:
        print(f"No CUDA GPU: {info['reason']}")
        print("On Colab: Runtime > Change runtime type > GPU (T4).")
        return
    print(
        f"GPU            : {info['name']}\n"
        f"VRAM           : {info['vram_gb']} GB\n"
        f"Compute        : {info['capability']} (sm_{info['arch']})\n"
        f"Suggested slots: {suggest_parallel(info['vram_gb'])}"
    )


if __name__ == "__main__":
    report()
