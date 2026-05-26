# Dockerfile
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

# System deps: ffmpeg with NVDEC/NVENC, nginx, Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    ffmpeg nginx \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# App code
COPY server.py auth.py pipeline.py idle_watcher.py start.sh ./
COPY nginx.conf /etc/nginx/nginx.conf

# RunPod port — single port per pod
EXPOSE 8080

ENV HLS_SERVE_DIR=/var/hls

CMD ["./start.sh"]
