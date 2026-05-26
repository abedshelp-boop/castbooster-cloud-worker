# server.py
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from auth import require_api_key
from pipeline import PipelineResult, run_passthrough

app = FastAPI(title="castbooster-cloud-worker", version="0.1.0")

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
    return run_passthrough(**kwargs)


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
