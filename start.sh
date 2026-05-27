#!/bin/bash
# start.sh — pod entrypoint
set -e

# Validate required env
: "${CLOUD_API_KEY:?CLOUD_API_KEY must be set}"

# Auto-derive PUBLIC_BASE_URL from RUNPOD_POD_ID if not explicitly set.
# RunPod injects RUNPOD_POD_ID into every pod's env automatically.
if [ -z "${PUBLIC_BASE_URL:-}" ]; then
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        export PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8080.proxy.runpod.net"
        echo "[start.sh] Auto-derived PUBLIC_BASE_URL=$PUBLIC_BASE_URL from RUNPOD_POD_ID"
    else
        echo "[start.sh] ERROR: neither PUBLIC_BASE_URL nor RUNPOD_POD_ID is set." >&2
        echo "[start.sh] One of them must be set so the worker knows its public hostname." >&2
        exit 1
    fi
fi

mkdir -p /var/hls

# Start FastAPI on :8001 (nginx proxies to it)
uvicorn server:app --host 127.0.0.1 --port 8001 &
UVICORN_PID=$!

# Start nginx in foreground (so docker logs work)
nginx -g 'daemon off;' &
NGINX_PID=$!

# Wait for either to exit
wait -n $UVICORN_PID $NGINX_PID
