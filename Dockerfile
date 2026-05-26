# Dockerfile
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

# System deps: ffmpeg with NVDEC/NVENC, nginx, Python 3.12.
# p7zip-full provides `7zz` for extracting vs-mlrt's .7z archives (no `unzip`
# needed — vs-mlrt does not publish .zip artefacts).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    ffmpeg nginx \
    ca-certificates curl p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# VapourSynth runtime + lsmas (source reader)
RUN apt-get update && apt-get install -y --no-install-recommends \
    vapoursynth vapoursynth-plugin-lsmas \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# vs-mlrt with TensorRT backend (Linux x64, RIFE v4.6 model).
#
# IMPORTANT: vs-mlrt does NOT publish a prebuilt Linux x64 release archive.
# Its release page only ships:
#   - vsmlrt-cuda.v${V}.7z              (Windows DLLs — NOT usable on Linux)
#   - vsmlrt-windows-x64-*.7z[.001|.002] (Windows full bundles)
#   - scripts.v${V}.7z                  (cross-platform: vsmlrt.py)
#   - models.v${V}.7z                   (cross-platform: ONNX models incl. RIFE)
# The Linux vstrt .so is only produced as a workflow-only CI artefact in
# .github/workflows/linux-trt.yml of the upstream repo. To deploy on Linux we
# must either (a) build vstrt from source against TensorRT 10, or (b) extract
# the workflow artefact. We do (a) here — it's reproducible, fast (<2 min on
# x86_64), and pins TRT/CUDA to the same major as the upstream CI.
#
# We pin to v15.13 — last single-file release (v15.14+ split vsmlrt-cuda into
# two 7z parts; not needed on Linux but keeps versioning simple). v15.13 ships
# the RIFE v4.6 model.
#
# TODO (Task 11 CI): verify these exact paths and the vstrt source build land
# correctly. If the build fails, fall back to extracting VSTRT-Linux-x64 from
# the upstream workflow artefacts via `gh run download`.
# -----------------------------------------------------------------------------
ARG VSMLRT_VERSION=15.13

# 1) TensorRT 10 runtime via NVIDIA's apt repo (cuda 12.6 keyring → libnvinfer10)
RUN curl -L -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         libnvinfer10 libnvinfer-plugin10 libnvinfer-headers-dev libnvinfer-dev \
         cuda-cudart-dev-12-6 \
         cmake ninja-build g++ git \
    && rm -rf /var/lib/apt/lists/*

# 2) Build vstrt plugin from source against TensorRT 10
#    (mirrors upstream linux-trt.yml). Output: /usr/local/lib/vapoursynth/libvstrt.so
RUN git clone --depth 1 --branch v${VSMLRT_VERSION} \
        https://github.com/AmusementClub/vs-mlrt.git /tmp/vs-mlrt \
    && cd /tmp/vs-mlrt/vstrt \
    && curl -L -o vs.zip https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R57.zip \
    && 7zz x vs.zip && mv vapoursynth-R57 vapoursynth \
    && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DVAPOURSYNTH_INCLUDE_DIRECTORY=/tmp/vs-mlrt/vstrt/vapoursynth/include \
        -DCMAKE_CXX_FLAGS="-Wall -ffast-math -march=x86-64-v3" \
    && cmake --build build \
    && mkdir -p /usr/lib/x86_64-linux-gnu/vapoursynth \
    && cp build/*.so /usr/lib/x86_64-linux-gnu/vapoursynth/ \
    && rm -rf /tmp/vs-mlrt/vstrt/build /tmp/vs-mlrt/vstrt/vs.zip

# 3) vsmlrt.py — Python wrapper module. Lives in `scripts.v${V}.7z` (single file).
#    Goes on Python's site-packages so `from vsmlrt import RIFE, Backend` works.
RUN curl -L -o /tmp/vsmlrt-scripts.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/scripts.v${VSMLRT_VERSION}.7z \
    && 7zz x /tmp/vsmlrt-scripts.7z -o/tmp/vsmlrt-scripts \
    && cp /tmp/vsmlrt-scripts/vsmlrt.py /usr/lib/python3.12/dist-packages/ \
    && rm -rf /tmp/vsmlrt-scripts /tmp/vsmlrt-scripts.7z

# 4) RIFE v4.6 ONNX model — ships in models.v${V}.7z (~850 MB; pruned to RIFE).
#    vs-mlrt's default model dir on Linux is /usr/local/share/vsmlrt or the dir
#    set by `MODEL_DIR` env (vsmlrt.py picks it up). We extract only the RIFE
#    subtree to keep the image small.
RUN mkdir -p /usr/local/share/vsmlrt \
    && curl -L -o /tmp/vsmlrt-models.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/models.v${VSMLRT_VERSION}.7z \
    && 7zz x /tmp/vsmlrt-models.7z -o/tmp/vsmlrt-models -y \
    && cp -r /tmp/vsmlrt-models/models/rife /usr/local/share/vsmlrt/ \
    && rm -rf /tmp/vsmlrt-models /tmp/vsmlrt-models.7z

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

# Pre-bake the RIFE v4.6 TRT FP16 engine for RTX 4090 (Ada Lovelace = sm_89)
# vs-mlrt caches engines; first invocation compiles and saves to /root/.cache
RUN mkdir -p /root/.cache/vs-mlrt
COPY trt_engine_builder.py /tmp/
# Skip engine bake on architectures without GPU; runtime will compile on first call
RUN python3 /tmp/trt_engine_builder.py || echo "Engine bake skipped (no GPU at build time — will compile on first /process)"

# App code
COPY server.py auth.py pipeline.py pipeline_types.py run_rife.py idle_watcher.py start.sh ./
COPY nginx.conf /etc/nginx/nginx.conf

# RunPod port — single port per pod
EXPOSE 8080

ENV HLS_SERVE_DIR=/var/hls

CMD ["./start.sh"]
