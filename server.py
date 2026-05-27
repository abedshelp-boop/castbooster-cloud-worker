# server.py
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from auth import require_api_key
from idle_watcher import IdleWatcher
from pipeline_types import PipelineResult
from run_rife import run_rife

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
    import sys
    import traceback as _tb
    print(f"[/process] ENTER source_url={req.source_url[:200]}", file=sys.stderr, flush=True)

    out_dir = _hls_dir()
    base = _public_base_url()
    print(f"[/process] out_dir={out_dir} base={base}", file=sys.stderr, flush=True)

    if not base:
        print(f"[/process] ABORT: PUBLIC_BASE_URL not configured", file=sys.stderr, flush=True)
        raise HTTPException(500, "PUBLIC_BASE_URL not configured")

    try:
        print(f"[/process] Calling run_pipeline_for_request...", file=sys.stderr, flush=True)
        result = run_pipeline_for_request(
            source_url=req.source_url,
            output_dir=out_dir,
            extra_input_headers=req.source_headers or None,
        )
        print(f"[/process] pipeline returned: returncode={result.returncode} stderr_tail={result.stderr[-200:]!r}", file=sys.stderr, flush=True)
    except (ValueError, NotImplementedError) as e:
        print(f"[/process] 400 invalid request: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise HTTPException(400, f"invalid request: {e}")
    except Exception as e:
        tb_tail = "".join(_tb.format_exception(type(e), e, e.__traceback__))[-1500:]
        print(f"[/process] 502 pipeline EXCEPTION: {type(e).__name__}: {e}\n{tb_tail}", file=sys.stderr, flush=True)
        raise HTTPException(
            502,
            detail=f"pipeline error: {type(e).__name__}: {e}\n--- traceback (tail) ---\n{tb_tail}",
        )

    if result.returncode != 0:
        print(f"[/process] 502 pipeline nonzero: rc={result.returncode}", file=sys.stderr, flush=True)
        raise HTTPException(502, detail=result.stderr[-500:] or "ffmpeg failed with no stderr")

    hls_url = f"{base.rstrip('/')}/hls/playlist.m3u8"
    print(f"[/process] 200 SUCCESS hls_url={hls_url}", file=sys.stderr, flush=True)
    return {"hls_url": hls_url}


@app.post("/stop", dependencies=[Depends(require_api_key)])
async def stop():
    _self_terminate_pod()
    return {"status": "shutting_down"}


@app.get("/diag", dependencies=[Depends(require_api_key)])
def diag():
    """Diagnostic endpoint: returns environment, tool versions, and vsmlrt state.

    Used to localize /process failures without container shell access.
    """
    import os
    import shutil
    import subprocess

    def _check(cmd: list[str]) -> dict:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return {
                "returncode": r.returncode,
                "stdout": r.stdout[:500],
                "stderr": r.stderr[:500],
            }
        except FileNotFoundError as e:
            return {"error": f"FileNotFoundError: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # Probe vsmlrt import + RIFE model resolution
    vsmlrt_info: dict = {}
    try:
        import vsmlrt as _vsmlrt  # type: ignore
        vsmlrt_info["import_ok"] = True
        try:
            vsmlrt_info["plugins_path"] = _vsmlrt.get_plugins_path()
        except Exception as e:
            vsmlrt_info["plugins_path_error"] = f"{type(e).__name__}: {e}"
        vsmlrt_info["models_path"] = getattr(_vsmlrt, "models_path", None)
    except Exception as e:
        vsmlrt_info["import_ok"] = False
        vsmlrt_info["import_error"] = f"{type(e).__name__}: {e}"

    # Probe vapoursynth core directly to see which plugins are actually loaded
    vs_core_info: dict = {}
    try:
        import vapoursynth as vs  # type: ignore
        vs_core_info["version"] = str(vs.core.version()).split("\n")[0] if hasattr(vs.core, "version") else "?"
        loaded_plugins = []
        try:
            for ns in dir(vs.core):
                if not ns.startswith("_"):
                    loaded_plugins.append(ns)
        except Exception as e:
            loaded_plugins.append(f"<probe error: {e}>")
        vs_core_info["namespaces"] = loaded_plugins[:50]  # cap to avoid huge responses
    except Exception as e:
        vs_core_info["error"] = f"{type(e).__name__}: {e}"

    # Probe what model files we can find
    model_dirs_to_check = [
        "/usr/local/lib/models/rife",
        "/usr/local/lib/vapoursynth/models/rife",
        "/usr/local/share/vsmlrt/rife",
    ]
    model_files_found = {}
    for d in model_dirs_to_check:
        try:
            if os.path.isdir(d):
                model_files_found[d] = sorted(os.listdir(d))[:20]
            else:
                model_files_found[d] = None  # dir doesn't exist
        except Exception as e:
            model_files_found[d] = f"<error: {e}>"

    # Probe libvstrt.so location (the actual one loaded, if any)
    libvstrt_candidates = []
    for path in [
        "/usr/local/lib/libvstrt.so",
        "/usr/local/lib/vapoursynth/libvstrt.so",
        "/usr/lib/x86_64-linux-gnu/vapoursynth/libvstrt.so",
    ]:
        libvstrt_candidates.append({
            "path": path,
            "exists": os.path.exists(path),
            "is_link": os.path.islink(path) if os.path.exists(path) else None,
        })

    return {
        "env": {
            "RUNPOD_POD_ID": os.environ.get("RUNPOD_POD_ID", ""),
            "PUBLIC_BASE_URL": os.environ.get("PUBLIC_BASE_URL", ""),
            "HLS_SERVE_DIR": os.environ.get("HLS_SERVE_DIR", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
        "binaries": {
            "vspipe": shutil.which("vspipe"),
            "ffmpeg": shutil.which("ffmpeg"),
            "nginx": shutil.which("nginx"),
            "python3": shutil.which("python3"),
        },
        "ffmpeg_version": _check(["ffmpeg", "-version"]),
        "vspipe_version": _check(["vspipe", "--version"]),
        "vsmlrt": vsmlrt_info,
        "vapoursynth_core": vs_core_info,
        "model_files": model_files_found,
        "libvstrt_candidates": libvstrt_candidates,
    }


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
