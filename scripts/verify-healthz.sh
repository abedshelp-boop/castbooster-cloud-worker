#!/bin/bash
# scripts/verify-healthz.sh — local smoke test before any image push.
#
# Builds the local Docker image, runs it with the env vars the orchestrator
# passes on RunPod, polls /healthz until 200, dumps logs, tears down.
# Exits 0 if /healthz returned 200 within the timeout, 1 otherwise.
#
# Why this exists: 2026-05-29 — v0.3.0 + v0.3.1 shipped to GHCR with the
# Dockerfile COPY list missing newly-added modules. pytest passed (tests run
# from the repo root with all .py on sys.path), but the production image
# uvicorn'd a server.py whose top-level imports were missing in /app, so
# uvicorn died with ModuleNotFoundError and start.sh restart-looped forever
# behind a permanent nginx 502. Took ~24h to detect on RunPod ($0.91 burned).
# This script catches that class of bug: it exercises the actual built image.
#
# Usage:
#   bash scripts/verify-healthz.sh
#   TAG=foo HOST_PORT=18080 TIMEOUT_S=90 bash scripts/verify-healthz.sh
#
# Run from repo root.
set -euo pipefail

TAG="${TAG:-local-verify}"
HOST_PORT="${HOST_PORT:-18080}"
TIMEOUT_S="${TIMEOUT_S:-60}"
CONTAINER_NAME="cbw-verify-$$"

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[verify] Building image castbooster-cloud-worker:${TAG} (linux/amd64)..."
docker build --platform linux/amd64 -t "castbooster-cloud-worker:${TAG}" .

echo "[verify] Starting ${CONTAINER_NAME} on host port ${HOST_PORT}..."
docker run -d --rm --name "${CONTAINER_NAME}" \
    --platform linux/amd64 \
    -p "${HOST_PORT}:8080" \
    -e CLOUD_API_KEY=verify-test \
    -e PUBLIC_BASE_URL="http://localhost:${HOST_PORT}" \
    -e DISABLE_IDLE_WATCHER=1 \
    "castbooster-cloud-worker:${TAG}"

echo "[verify] Polling http://localhost:${HOST_PORT}/healthz for up to ${TIMEOUT_S}s..."
for i in $(seq 1 "${TIMEOUT_S}"); do
    if curl -fsS "http://localhost:${HOST_PORT}/healthz" >/dev/null 2>&1; then
        echo "[verify] /healthz returned 200 after ${i}s — image is healthy."
        echo "[verify] --- container logs (tail 20) ---"
        docker logs "${CONTAINER_NAME}" 2>&1 | tail -n 20
        exit 0
    fi
    sleep 1
done

echo "[verify] FATAL: /healthz never returned 200 after ${TIMEOUT_S}s." >&2
echo "[verify] --- full container logs ---" >&2
docker logs "${CONTAINER_NAME}" >&2 2>&1 || true
exit 1
