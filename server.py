# server.py
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from auth import require_api_key
from idle_watcher import IdleWatcher
from run_rife import PipelineResult, run_rife

app = FastAPI(title="castbooster-cloud-worker", version="0.1.0")

_watcher: IdleWatcher | None = None


def _self_terminate_pod():
    """Best-effort RunPod self-termination via the pod's RUNPOD_POD_ID env var.

    This works because RunPod injects RUNPOD_POD_ID into every pod's environment;
    we call the RunPod REST API from inside the pod to terminate ourselves.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not pod_id or not runpod_api_key:
        # Local dev / test mode — just log
        print(f"[idle_watcher] Would terminate pod {pod_id!r} (no API key set)")
        return
    try:
        import httpx
        resp = httpx.post(
            f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
            headers={"Authorization": f"Bearer {runpod_api_key}"},
            timeout=10.0,
        )
        if resp.status_code >= 400:
            print(
                f"[idle_watcher] RunPod stop returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        else:
            print(
                f"[idle_watcher] Self-terminate requested for pod {pod_id} "
                f"(status {resp.status_code})"
            )
    except Exception as e:
        print(f"[idle_watcher] Self-terminate failed: {e}")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/_test_protected", dependencies=[Depends(require_api_key)])
async def _test_protected():
    return {"authorized": True}


class ProcessRequest(BaseModel):
    source_url: str
    source_headers: dict[str, str] = {}
    output_resolution: str = "1080p"


def _hls_dir() -> Path:
    return Path(os.environ.get("HLS_SERVE_DIR", "/var/hls"))


def _public_base_url() -> str:
    return os.environ.get("PUBLIC_BASE_URL", "")


# Wrapped indirection so tests can monkeypatch a fake
def run_pipeline_for_request(**kwargs) -> PipelineResult:
    return run_rife(**kwargs)


@app.post("/process", dependencies=[Depends(require_api_key)])
def process(req: ProcessRequest):
    out_dir = _hls_dir()
    base = _public_base_url()
    if not base:
        raise HTTPException(500, "PUBLIC_BASE_URL not configured")

    try:
        result = run_pipeline_for_request(
            source_url=req.source_url,
            output_dir=out_dir,
            extra_input_headers=req.source_headers or None,
        )
    except ValueError as e:
        raise HTTPException(400, f"invalid request: {e}")
    if result.returncode != 0:
        # Truncate to last 500 chars; ffmpeg errors are noisy
        raise HTTPException(502, detail=result.stderr[-500:] or "ffmpeg failed with no stderr")

    hls_url = f"{base.rstrip('/')}/hls/playlist.m3u8"
    return {"hls_url": hls_url}


@app.post("/stop", dependencies=[Depends(require_api_key)])
async def stop():
    _self_terminate_pod()
    return {"status": "shutting_down"}


@app.on_event("startup")
def _start_idle_watcher():
    global _watcher
    if os.environ.get("DISABLE_IDLE_WATCHER") == "1":
        return  # tests
    _watcher = IdleWatcher(
        watch_dir=_hls_dir(),
        idle_seconds=10 * 60,
        on_shutdown=_self_terminate_pod,
        check_interval_s=30.0,
        hard_max_lifetime_s=6 * 60 * 60,
    )
    _watcher.start()


@app.on_event("shutdown")
def _stop_idle_watcher():
    if _watcher:
        _watcher.stop()
