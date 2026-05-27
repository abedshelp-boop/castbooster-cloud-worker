# Dockerfile
# -----------------------------------------------------------------------------
# Base: styler00dollar/vsgan_tensorrt:minimal (linux/amd64, ~6.1 GB compressed).
#
# This community-maintained image (updated within the last 3 weeks) ships:
#   - nvcr.io/nvidia/tensorrt:26.04-py3 base (CUDA 13 + TensorRT 10 + Python 3.12)
#   - VapourSynth R73 built from source (libvapoursynth.so + python bindings)
#   - vs-mlrt's libvstrt.so plugin built from source (at /usr/local/lib/vapoursynth/)
#   - lsmas / lsmashsource / bestsource / ffms2 VS plugins
#   - ffmpeg static build with NVDEC/NVENC + libdav1d
#   - g++-14 toolchain, zimg, cmake 4.1
#   - WORKDIR /workspace/tensorrt, CUDA_MODULE_LOADING=LAZY
#
# Why this base instead of building from nvidia/cuda:12.6.1-cudnn-ubuntu24.04:
# Noble (24.04) doesn't have `vapoursynth` in apt. Jammy (22.04) only has R55
# via the savoury1 PPA — too old for vs-mlrt v15. Building VapourSynth + vstrt
# from scratch takes 20+ min of CI time. The styler image already does it and
# tracks upstream closely. See: github.com/styler00dollar/VSGAN-tensorrt-docker
#
# What this layer adds on top:
#   - pip deps from requirements.txt (fastapi, uvicorn, httpx, pytest)
#   - nginx (HLS edge)
#   - vsmlrt.py Python wrapper module (NOT installed by the base — only libvstrt.so is)
#   - RIFE v4.6 ONNX model (pruned from vs-mlrt v15.13 models bundle)
#   - Our app code + start.sh
#
# Trade-offs / risks (DONE_WITH_CONCERNS):
#   1. CUDA 13 in base, not 12.6 as previously pinned. RTX 4090 (sm_89) is fully
#      supported by CUDA 13 and TRT 10. Drivers on RunPod 4090 pods (560+) support
#      CUDA 13 runtime.
#   2. Third-party image dependency. Mitigation: pin to a digest later (Phase 2);
#      until then pin to the `:minimal` tag.
#   3. Final image size will be ~6.5-7 GB compressed (~13-15 GB uncompressed).
#      Acceptable for RunPod cold-start budget (24s with SW warming).
# -----------------------------------------------------------------------------

FROM docker.io/styler00dollar/vsgan_tensorrt:minimal

ARG VSMLRT_VERSION=15.13

# Our app's WORKDIR (overrides base's /workspace/tensorrt)
WORKDIR /app

# System deps not in the base: nginx, p7zip-full (for vs-mlrt 7z archives),
# curl/ca-certificates may already be present but cheap to ensure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        p7zip-full \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (Python 3.12 + pip already in base image).
# Base ships pip system-wide; --break-system-packages avoids PEP-668 refusal.
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# vsmlrt.py — Python wrapper for vs-mlrt. The base image builds and installs
# libvstrt.so (the C++ TRT plugin) but does NOT include the Python wrapper that
# our run_rife.py imports (`from vsmlrt import RIFE, Backend`). We fetch it from
# the upstream release archive.
RUN curl -L -o /tmp/vsmlrt-scripts.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/scripts.v${VSMLRT_VERSION}.7z \
    && 7zz x /tmp/vsmlrt-scripts.7z -o/tmp/vsmlrt-scripts \
    && cp /tmp/vsmlrt-scripts/vsmlrt.py /usr/local/lib/python3.12/site-packages/ \
    && rm -rf /tmp/vsmlrt-scripts /tmp/vsmlrt-scripts.7z

# RIFE v4.6 ONNX model — vsmlrt.py searches `${MODEL_DIR}` or its built-in default
# (alongside the .so plugin). The styler image installs libvstrt.so to
# /usr/local/lib/vapoursynth/, so the standard model search path is
# /usr/local/lib/vapoursynth/models/rife/ (vsmlrt.py's plugin_dir convention).
# Extract only the RIFE subtree to keep the layer small (~850 MB archive → ~50 MB
# kept for RIFE v4.6 model).
RUN mkdir -p /usr/local/lib/vapoursynth/models \
    && curl -L -o /tmp/vsmlrt-models.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/models.v${VSMLRT_VERSION}.7z \
    && 7zz x /tmp/vsmlrt-models.7z -o/tmp/vsmlrt-models -y \
    && cp -r /tmp/vsmlrt-models/models/rife /usr/local/lib/vapoursynth/models/ \
    && rm -rf /tmp/vsmlrt-models /tmp/vsmlrt-models.7z

# Pre-bake the RIFE v4.6 TRT FP16 engine for RTX 4090 (Ada Lovelace = sm_89).
# At build time there's no GPU in the CI runner, so this just exits cleanly and
# the engine compiles on first /process call. Kept here so a future GPU-equipped
# CI runner (or a multi-stage `RUN --device=nvidia.com/gpu=all` build) can bake
# it ahead of time.
RUN mkdir -p /root/.cache/vs-mlrt
COPY trt_engine_builder.py /tmp/
RUN python3 /tmp/trt_engine_builder.py \
    || echo "Engine bake skipped (no GPU at build time — will compile on first /process)"

# App code
COPY server.py auth.py pipeline.py pipeline_types.py run_rife.py idle_watcher.py start.sh ./
COPY nginx.conf /etc/nginx/nginx.conf

# RunPod port — single port per pod (nginx terminates HTTPS via RunPod proxy,
# forwards / → uvicorn:8001, /hls/* → file-system).
EXPOSE 8080

ENV HLS_SERVE_DIR=/var/hls

CMD ["./start.sh"]
