# Dockerfile
# -----------------------------------------------------------------------------
# Base: styler00dollar/vsgan_tensorrt:minimal_no_avx512 (linux/amd64, ~5.5 GB).
#
# v0.1.8 — Switched from :minimal to :minimal_no_avx512 to fix SIGILL on
# RunPod RTX 4090 pods landing on AMD EPYC 7C13 (Milan/Zen 3, no AVX-512).
# The :minimal tag is built with -march=native on an AVX-512 host (7950x),
# producing AVX-512 binaries that crash on EPYC 7C13 with -4 (SIGILL).
# The :minimal_no_avx512 tag is a manual rebuild without AVX-512 — works
# on Zen 3. See gotchas/2026-05-27-cloud-worker-week1-acceptance.md.
#
# Tag details (per Docker Hub manifest, 2026-05-27):
#   - minimal_no_avx512  : 5.55 GB compressed, last updated 2025-10-30,
#     TensorRT 10.13, ffmpeg+mlrt+ffms2+lsmash+bestsource.
#   - minimal            : 6.24 GB, last updated 2026-05-08, TensorRT 10.16,
#     but built with AVX-512 — SIGILLs on Milan/Zen 3.
#
# This community-maintained image ships:
#   - nvcr.io/nvidia/tensorrt:* base (CUDA 13 + TensorRT 10.13 + Python 3.12)
#   - VapourSynth R73 built from source (libvapoursynth.so + python bindings)
#   - vs-mlrt's libvstrt.so plugin built from source (at /usr/local/lib/libvstrt.so)
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
# Trade-offs / risks:
#   1. TRT 10.13 in :minimal_no_avx512 vs 10.16 in :minimal. vsmlrt 15.13
#      supports both. RIFE v4.6 ONNX builds on both without ABI break.
#   2. Tag last updated 7 months ago (2025-10-30). Acceptable for MVP;
#      pin to digest at Phase 2.
#   3. Third-party image dependency. Mitigation: pin to a digest later.
#   4. Final image size ~5.5-6 GB compressed. Slightly smaller than v0.1.7.
# -----------------------------------------------------------------------------

FROM docker.io/styler00dollar/vsgan_tensorrt:minimal_no_avx512

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
    && 7z x /tmp/vsmlrt-scripts.7z -o/tmp/vsmlrt-scripts \
    && cp /tmp/vsmlrt-scripts/vsmlrt.py /usr/local/lib/python3.12/dist-packages/ \
    && rm -rf /tmp/vsmlrt-scripts /tmp/vsmlrt-scripts.7z

# RIFE v4.6 ONNX model — vsmlrt.py resolves models at:
#   models_path = os.path.dirname(<vstrt_plugin_path>) + "/models"
# In the styler base image, vstrt CMake installs to /usr/local/lib/libvstrt.so
# (see styler Dockerfile L629 and vstrt's CMakeLists default LIBDIR), so the
# computed models_path is /usr/local/lib/models/. For RIFE v4.6, the file lookup
# is at <models_path>/rife/rife_v4.6.onnx (vsmlrt.py v15.13 L1099-1103).
#
# vsmlrt.py has NO env-var or model_dir override — the path is fully derived
# from where the .so plugin was loaded. So we install to /usr/local/lib/models/
# (the actually-computed path) AND mirror to /usr/local/lib/vapoursynth/models/
# (the convention used by some other VS plugin layouts) as belt-and-suspenders,
# in case styler ever relocates libvstrt.so to the autoload dir in a later tag.
# Extract only the RIFE subtree to keep the layer small (~850 MB archive → ~50 MB
# kept for RIFE v4.6 model).
RUN mkdir -p /usr/local/lib/models /usr/local/lib/vapoursynth/models \
    && curl -L -o /tmp/vsmlrt-models.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/models.v${VSMLRT_VERSION}.7z \
    && 7z x /tmp/vsmlrt-models.7z -o/tmp/vsmlrt-models -y \
    && cp -r /tmp/vsmlrt-models/models/rife /usr/local/lib/models/ \
    && cp -r /tmp/vsmlrt-models/models/rife /usr/local/lib/vapoursynth/models/ \
    && rm -rf /tmp/vsmlrt-models /tmp/vsmlrt-models.7z

# libvstrt.so autoload fix: styler base installs it to /usr/local/lib/ (build artifact
# location) but VapourSynth's autoload only scans /usr/local/lib/vapoursynth/. Symlink
# so `from vsmlrt import RIFE` finds the plugin at runtime. Verified failing in v0.1.4
# pod test (see vault gotchas/2026-05-27-cloud-worker-week1-acceptance.md).
RUN mkdir -p /usr/local/lib/vapoursynth \
    && if [ -f /usr/local/lib/libvstrt.so ]; then \
           ln -sf /usr/local/lib/libvstrt.so /usr/local/lib/vapoursynth/libvstrt.so; \
           echo "Symlinked libvstrt.so to /usr/local/lib/vapoursynth/"; \
       else \
           echo "WARNING: /usr/local/lib/libvstrt.so not found in base image"; \
       fi \
    && ls -la /usr/local/lib/vapoursynth/

# Smoke-check: verify the vstrt namespace can be probed at all from Python.
# Don't FAIL the build on this — it'll fail without a GPU at CI time anyway —
# but log the result for debugging via docker history / build logs.
RUN python3 -c "import vapoursynth as vs; namespaces = sorted([p for p in dir(vs.core) if not p.startswith('_')]); print('VS namespaces:', namespaces); print('trt loaded:', 'trt' in namespaces)" 2>&1 || true

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
