# Runs the pipeline as a public web app on Hugging Face Spaces (Docker SDK,
# free CPU tier). See the "Deploy your own copy" section in README.md.
#
# Why Docker and not the Streamlit SDK: marker 2.0 spawns `llama-server`
# itself (see CLAUDE.md / bt/preflight.py) and Spaces' managed Streamlit
# runtime has no way to install it. A custom image can build it.
#
# Why CPU, not GPU: the free tier has no GPU. Free Spaces top out at 16 GB RAM
# / 2 vCPU, which is enough headroom for torch + surya + llama-server's ~2.4 GB
# working set, but OCR runs far slower here than the Metal-accelerated path
# this repo was built around (measured 82-95 s/page on Apple Silicon) --
# expect single-digit minutes per page. That is a real product tradeoff of
# using free infrastructure, not a bug in this image.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- llama-server (CPU build) ---------------------------------------------
# marker does not install this for you. Upstream ships no CPU binary release
# guaranteed to match every Spaces host's CPU, so it is built once, at image
# build time, and baked in -- mirrors bt/gpu_setup.py's CUDA build for Colab,
# same idea, CPU instead of CUDA.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DGGML_CUDA=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/llama.cpp/build --config Release -j"$(nproc)" --target llama-server \
    && install -m 0755 /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server \
    && rm -rf /opt/llama.cpp

RUN pip install --no-cache-dir uv

# Spaces containers should not run as root.
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser . .
RUN mkdir -p uploads output && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser \
    HF_HUB_DISABLE_XET=1 \
    OCR_ERROR_SERVER_STARTUP_TIMEOUT=1800 \
    DETECTOR_SERVER_STARTUP_TIMEOUT=1800 \
    FAST_LAYOUT_SERVER_STARTUP_TIMEOUT=1800 \
    SURYA_INFERENCE_STARTUP_TIMEOUT=1800 \
    SURYA_INFERENCE_KEEP_ALIVE=1

RUN uv sync --frozen --no-dev

# Pre-download every model into the image (bt/warmup.py) so a visitor's first
# run does not pay a multi-minute cold-start download, and so a Space that
# goes to sleep and restarts (free tier, after inactivity) comes back warm --
# only uploads/ and output/ are lost on restart, not the model cache.
RUN uv run python -m bt.warmup

EXPOSE 7860

# XSRF/CORS protection is disabled because Spaces serves the app through its
# own iframe proxy, which a same-origin check would otherwise block -- this
# is what Hugging Face's own Streamlit template does for the same reason.
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false", \
     "--server.fileWatcherType=none"]
