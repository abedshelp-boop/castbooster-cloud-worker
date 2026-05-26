#!/bin/bash
# start.sh — pod entrypoint
set -e

# Validate required env
: "${CLOUD_API_KEY:?CLOUD_API_KEY must be set}"
: "${PUBLIC_BASE_URL:?PUBLIC_BASE_URL must be set (e.g. https://pod-id-8080.proxy.runpod.net)}"

mkdir -p /var/hls

# Start FastAPI on :8001 (nginx proxies to it)
uvicorn server:app --host 127.0.0.1 --port 8001 &
UVICORN_PID=$!

# Start nginx in foreground (so docker logs work)
nginx -g 'daemon off;' &
NGINX_PID=$!

# Wait for either to exit
wait -n $UVICORN_PID $NGINX_PID
