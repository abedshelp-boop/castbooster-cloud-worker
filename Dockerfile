# Dockerfile
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

# System deps: ffmpeg with NVDEC/NVENC, nginx, Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    ffmpeg nginx \
    ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# VapourSynth + vs-mlrt + TensorRT
# vs-mlrt bundles TRT in its release zip — see github.com/AmusementClub/vs-mlrt/releases
RUN apt-get update && apt-get install -y --no-install-recommends \
    vapoursynth vapoursynth-plugin-lsmas \
    && rm -rf /var/lib/apt/lists/*

# vs-mlrt with TRT backend (release v15.7+ ships RIFE v4.6 model)
ARG VSMLRT_VERSION=15.7
RUN curl -L https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/vsmlrt-cuda-${VSMLRT_VERSION}-linux-x64.zip -o /tmp/vsmlrt.zip \
    && unzip /tmp/vsmlrt.zip -d /usr/lib/x86_64-linux-gnu/vapoursynth/ \
    && rm /tmp/vsmlrt.zip

# Pre-bake the RIFE v4.6 TRT FP16 engine for RTX 4090 (Ada Lovelace = sm_89)
# vs-mlrt caches engines; first invocation compiles and saves to /root/.cache
RUN mkdir -p /root/.cache/vs-mlrt
COPY trt_engine_builder.py /tmp/
# Skip engine bake on architectures without GPU; runtime will compile on first call
RUN python3 /tmp/trt_engine_builder.py || echo "Engine bake skipped (no GPU at build time — will compile on first /process)"

# App code
COPY server.py auth.py pipeline.py run_rife.py idle_watcher.py start.sh ./
COPY nginx.conf /etc/nginx/nginx.conf

# RunPod port — single port per pod
EXPOSE 8080

ENV HLS_SERVE_DIR=/var/hls

CMD ["./start.sh"]
