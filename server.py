# server.py
import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from auth import require_api_key
from idle_watcher import IdleWatcher
from pipeline_types import PipelineResult
from run_rife import run_rife

app = FastAPI(title="castbooster-cloud-worker", version="0.1.0")


# Catch-all error logger: if any unhandled exception escapes a handler, log it
# to stdout (visible in RunPod web UI logs) BEFORE FastAPI's error middleware
# emits a 500. This is critical because Cloudflare 502s in <1s suggest the
# upstream connection is severed before FastAPI can emit a response — but if
# the handler raises a normal Python exception, this catches it and forces
# a JSON 500 through nginx instead of a TCP RST.
@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception):
    import traceback as _tb
    tb = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    msg = (
        f"[UNHANDLED] {request.method} {request.url.path} -> "
        f"{type(exc).__name__}: {exc}\n{tb}"
    )
    print(msg, file=sys.stderr, flush=True)
    print(msg, file=sys.stdout, flush=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path,
            "method": request.method,
            "traceback_tail": tb[-1500:],
        },
    )

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
    print("[/_test_protected] HIT", file=sys.stdout, flush=True)
    print("[/_test_protected] HIT", file=sys.stderr, flush=True)
    return {"authorized": True}


# -----------------------------------------------------------------------------
# v0.1.6 DIAGNOSTIC ENDPOINTS — added 2026-05-27 to isolate the Cloudflare 502
# pattern affecting /_test_protected and /process. Each varies ONE dimension
# at a time vs. routes that work.
#
# Hits ALL get logged to BOTH stdout and stderr so RunPod's web UI tail
# captures them. If a route gets a 502 from CF but we see the log line, the
# response is being dropped between FastAPI->nginx->RunPod proxy->CF. If we
# DON'T see the log line, the request never reached the handler.
# -----------------------------------------------------------------------------

def _log(tag: str, **fields):
    """Log to both stdout and stderr with flush, so RunPod logs catch it."""
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
    msg = f"[{tag}] {parts}"
    print(msg, file=sys.stdout, flush=True)
    print(msg, file=sys.stderr, flush=True)


# /raw1: GET, no auth, JSON. Pure control. Should always work.
@app.get("/raw1")
async def raw1():
    _log("/raw1 HIT")
    return {"ok": True}


# /raw2: GET, no auth, PlainTextResponse. Bypass JSON serialization.
@app.get("/raw2")
async def raw2():
    _log("/raw2 HIT")
    return PlainTextResponse(content="ok")


# /raw3: GET, auth, JSON. Same shape as /_test_protected.
@app.get("/raw3", dependencies=[Depends(require_api_key)])
async def raw3():
    _log("/raw3 HIT")
    return {"ok": True}


# /raw4: GET, auth, PlainTextResponse. Same as /raw3 but no JSON.
@app.get("/raw4", dependencies=[Depends(require_api_key)])
async def raw4():
    _log("/raw4 HIT")
    return PlainTextResponse(content="ok")


# /raw5: POST, auth, JSON body, JSON response. Same shape as /process minus pipeline.
class _RawBody(BaseModel):
    msg: str = ""


@app.post("/raw5", dependencies=[Depends(require_api_key)])
async def raw5(req: _RawBody):
    _log("/raw5 HIT", msg=req.msg[:80])
    return {"ok": True, "received": req.msg}


# /raw6: GET, auth, sync def (uses threadpool). Same as /raw3 but sync.
@app.get("/raw6", dependencies=[Depends(require_api_key)])
def raw6():
    _log("/raw6 HIT (sync)")
    return {"ok": True}


# /raw7: GET, auth, explicit Response object. Bypasses FastAPI return-value handling.
@app.get("/raw7", dependencies=[Depends(require_api_key)])
async def raw7():
    _log("/raw7 HIT")
    return Response(content=b'{"ok":true}', media_type="application/json")


# /raw8: POST, no auth, JSON body. Test if auth dep + POST + body together is the problem.
@app.post("/raw8")
async def raw8(req: _RawBody):
    _log("/raw8 HIT", msg=req.msg[:80])
    return {"ok": True, "received": req.msg}


# /raw9: POST, no body (empty). Tests if just POST works without body.
@app.post("/raw9", dependencies=[Depends(require_api_key)])
async def raw9():
    _log("/raw9 HIT")
    return {"ok": True}


# -----------------------------------------------------------------------------
# v0.1.7 NATIVE-CALL ISOLATION ENDPOINTS — added 2026-05-27 to identify WHICH
# native call in run_rife()'s _build_clip() segfaults the uvicorn worker.
#
# Each endpoint isolates ONE step. If any of these crash the pod (502 + pod
# restart), that step is the segfault source.
#
# Run them in order on a fresh pod: ffmpeg-check, vs-import, lsmas-construct,
# resize-construct, rife-construct, trt-backend.
# -----------------------------------------------------------------------------

@app.get("/probe/ffmpeg", dependencies=[Depends(require_api_key)])
def probe_ffmpeg():
    """Run ffmpeg --version directly. If returncode is -N (negative), the
    binary was killed by signal N. -4 = SIGILL (-march=native CPU mismatch).
    """
    import subprocess
    _log("/probe/ffmpeg ENTER")
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        result = {
            "returncode": r.returncode,
            "killed_by_signal": -r.returncode if r.returncode < 0 else None,
            "stdout_head": r.stdout[:500],
            "stderr_head": r.stderr[:500],
        }
        _log("/probe/ffmpeg DONE", returncode=r.returncode)
        return result
    except Exception as e:
        _log("/probe/ffmpeg EXC", err=str(e))
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/probe/cpuinfo", dependencies=[Depends(require_api_key)])
def probe_cpuinfo():
    """Return the host CPU's flags so we can correlate ffmpeg SIGILL with
    missing instruction sets (avx512, etc.).
    """
    _log("/probe/cpuinfo ENTER")
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        # Parse first processor block
        first_block = content.split("\n\n")[0] if "\n\n" in content else content[:4000]
        return {"cpuinfo": first_block[:4000]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/probe/lsmas_safe", dependencies=[Depends(require_api_key)])
def probe_lsmas_safe():
    """Call core.lsmas.LWLibavSource() with a known-bad path, catching ALL
    Python exceptions. If this CRASHES THE POD (502 + restart), lsmas is
    segfaulting natively. If it returns a JSON error, lsmas raises cleanly.
    """
    _log("/probe/lsmas_safe ENTER")
    try:
        import vapoursynth as vs  # already imported via /diag earlier most likely
        core = vs.core
        _log("/probe/lsmas_safe calling LWLibavSource")
        try:
            src = core.lsmas.LWLibavSource(source="file:///nonexistent.ts")
            _log("/probe/lsmas_safe got back", n_frames=src.num_frames)
            return {"ok": True, "n_frames": src.num_frames}
        except Exception as e:
            _log("/probe/lsmas_safe caught", err=f"{type(e).__name__}: {e}")
            return {"ok": False, "error_type": type(e).__name__, "error_msg": str(e)}
    except Exception as e:
        _log("/probe/lsmas_safe outer EXC", err=str(e))
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/probe/rife_construct", dependencies=[Depends(require_api_key)])
def probe_rife_construct():
    """Try to construct just the Backend.TRT(...) object without any clip.
    If THIS crashes the pod, the TRT backend init segfaults.
    """
    _log("/probe/rife_construct ENTER")
    try:
        from vsmlrt import Backend
        _log("/probe/rife_construct Backend imported")
        try:
            backend = Backend.TRT(fp16=True, num_streams=2)
            _log("/probe/rife_construct Backend.TRT instantiated", backend=str(backend))
            return {"ok": True, "backend_repr": str(backend)}
        except Exception as e:
            _log("/probe/rife_construct caught", err=f"{type(e).__name__}: {e}")
            return {"ok": False, "error_type": type(e).__name__, "error_msg": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


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
    import traceback as _tb
    # Log to BOTH stdout and stderr — v0.1.5 had only stderr and we never saw it
    # in RunPod web UI logs. Mirror to stdout too.
    _log("/process ENTER", source_url=req.source_url[:200])
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
    _log("/stop HIT")
    _self_terminate_pod()
    return {"status": "shutting_down"}


@app.get("/diag", dependencies=[Depends(require_api_key)])
def diag():
    """Diagnostic endpoint: returns environment, tool versions, and vsmlrt state.

    Used to localize /process failures without container shell access.
    """
    _log("/diag HIT")
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
    _log("STARTUP", pid=os.getpid(), python=sys.version.split()[0])
    if os.environ.get("DISABLE_IDLE_WATCHER") == "1":
        _log("STARTUP idle_watcher disabled (DISABLE_IDLE_WATCHER=1)")
        return  # tests
    _watcher = IdleWatcher(
        watch_dir=_hls_dir(),
        idle_seconds=10 * 60,
        on_shutdown=_self_terminate_pod,
        check_interval_s=30.0,
        hard_max_lifetime_s=6 * 60 * 60,
    )
    _watcher.start()
    _log("STARTUP idle_watcher started")


@app.on_event("shutdown")
def _stop_idle_watcher():
    if _watcher:
        _watcher.stop()
