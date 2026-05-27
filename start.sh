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

# v0.1.7: wrap uvicorn in a restart loop so a native crash in /process doesn't
# take down the whole pod. Without this, a single segfault during clip-build
# kills uvicorn and the container exits — RunPod restarts the whole container
# (~30-90s downtime). With this loop, the next /process call gets a fresh
# uvicorn within seconds.
uvicorn_loop() {
    while true; do
        echo "[start.sh] Starting uvicorn (server:app) on :8001"
        uvicorn server:app --host 127.0.0.1 --port 8001 || true
        echo "[start.sh] uvicorn EXITED. Restarting in 2s." >&2
        sleep 2
    done
}

uvicorn_loop &
UVICORN_LOOP_PID=$!

# Start nginx in foreground (so docker logs work)
nginx -g 'daemon off;' &
NGINX_PID=$!

# Wait for either to exit. nginx exiting still kills the pod, but the
# uvicorn_loop only exits if SIGKILL'd, so it'll keep respawning uvicorn.
wait -n $UVICORN_LOOP_PID $NGINX_PID
