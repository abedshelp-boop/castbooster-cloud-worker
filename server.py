# server.py
import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from auth import require_api_key
from idle_watcher import IdleWatcher
from pipeline_manager import PipelineManager, PipelineState
from pipeline_types import PipelineResult
from run_rife import run_rife
from source_proxy import source_proxy
from source_registry import SourceRegistry

app = FastAPI(title="castbooster-cloud-worker", version="0.3.1")


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
    """Spawn the RIFE+NVENC pipeline in a background thread; return the
    predicted HLS URL within ~1s. Errors after spawn surface via
    /process_status as state="failed".

    Pre-flight (synchronous) validation:
      - PUBLIC_BASE_URL must be set (500 if not).
      - CRLF in source_headers raises ValueError -> 400.

    v0.3.1 (Pillar 3.6): non-empty source_headers no longer 400. They get
    registered with app.state.source_registry, and the pipeline source URL
    becomes a loopback /source_proxy?token=... URL so BestSource always
    reads through the header-injecting proxy.
    """
    _log("/process ENTER", source_url=req.source_url[:200])
    base = _public_base_url()
    if not base:
        return JSONResponse(
            status_code=500,
            content={"error": "PUBLIC_BASE_URL not configured"},
        )

    # CRLF guard — defense-in-depth against header smuggling. Keep
    # synchronous so the client gets 400 immediately (not buried inside
    # an async FAILED status). source_proxy stores headers verbatim, so
    # this must catch malformed input before registration.
    headers = req.source_headers or {}
    for k, v in headers.items():
        if "\r" in k or "\n" in k or "\r" in v or "\n" in v:
            return JSONResponse(
                status_code=400,
                content={
                    "error_type": "ValueError",
                    "error_msg": f"header {k!r} contains CR/LF (injection attempt)",
                },
            )

    # Always route through the loopback proxy — even for empty headers —
    # so behavior is uniform across egydead, mux.dev, and clean sources.
    # The proxy resolves to the original URL when no rewrite is needed,
    # and adds the PNG-wrapper strip + playlist rewrite path universally.
    registry: SourceRegistry = app.state.source_registry
    token = registry.register(req.source_url, headers)
    local_source_url = f"http://127.0.0.1:8001/source_proxy?token={token}"

    mgr: PipelineManager = app.state.pipeline_manager
    outcome = mgr.start(
        source_url=local_source_url,
        headers=None,  # headers are already injected by the proxy
        on_terminal=lambda t=token: registry.unregister(t),
        display_source_url=req.source_url,
    )
    if outcome.success:
        # Return only the public fields (drop "traceback" since it's null
        # here). The snapshot's source_url is the client-provided URL
        # (display_source_url), not the internal loopback proxy URL —
        # see PipelineManager.start() docs.
        snap = outcome.snapshot
        return {
            "hls_url": snap["hls_url"],
            "state": snap["state"],
            "started_at": snap["started_at"],
            "source_url": snap["source_url"],
        }
    # Already running -> 409 with snapshot of the running pipeline.
    snap = outcome.snapshot
    return JSONResponse(
        status_code=409,
        content={
            "state": snap["state"],
            "hls_url": snap["hls_url"],
            "source_url": snap["source_url"],
            "started_at": snap["started_at"],
        },
    )


@app.get("/process_status", dependencies=[Depends(require_api_key)])
def process_status():
    """Snapshot of the background pipeline thread's state.

    Used by the laptop orchestrator (Task 16) to gate Chromecast playback
    on playlist_ready and to surface pipeline failures back to the user.
    """
    mgr: PipelineManager = app.state.pipeline_manager
    return mgr.status()


# v0.3.1 (Pillar 3.6): in-pod source proxy for header injection. NOT
# bearer-gated — BestSource on localhost calls this with the loopback URL
# /process built; the token in the query string IS the capability. Outside
# 127.0.0.1 the endpoint is firewalled by nginx (only :8080 is exposed
# publicly; FastAPI binds :8001 on loopback). See source_proxy.py docstring.
app.add_api_route("/source_proxy", source_proxy, methods=["GET"])


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


# -----------------------------------------------------------------------------
# v0.1.9 GRANULAR VS GRAPH PROBES — added 2026-05-27 to isolate which step in
# run_rife._build_clip() crashes /process. Each probe wraps ONE additional
# stage of the VS pipeline on top of the previous probe, all wrapped in
# try/except so uvicorn never crashes. They return structured JSON either way.
#
# Run in order on a fresh pod:
#   /probe/lsmas_real     — lsmas LWLibavSource on real HTTP URL, NO realization
#   /probe/lsmas_realize  — same + force decode of frame 0
#   /probe/resize_rgbs    — add YUV->RGBS resize, realize frame 0
#   /probe/rife_eval      — full pipeline: lsmas+resize+RIFE+reconvert, frame 0
#                           (5-10 min TRT engine compile on first call)
#   /probe/rife_eval_padded — same as rife_eval but pads source to mult-of-32
# -----------------------------------------------------------------------------

@app.get("/probe/lsmas_real", dependencies=[Depends(require_api_key)])
def probe_lsmas_real():
    """Try lsmas on the real test URL with no further processing.
    Returns clip metadata if successful; structured error if not."""
    import traceback as _tb
    _log("/probe/lsmas_real ENTER")
    try:
        import vapoursynth as vs
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.lsmas.LWLibavSource(source=url)
        return {
            "ok": True,
            "num_frames": src.num_frames,
            "fps_num": src.fps_num,
            "fps_den": src.fps_den,
            "width": src.width,
            "height": src.height,
            "format_name": src.format.name if src.format else None,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


@app.get("/probe/lsmas_realize", dependencies=[Depends(require_api_key)])
def probe_lsmas_realize():
    """Try lsmas + force realization of frame 0 (actually decode).
    This is what would crash if there's a codec/decoder issue."""
    import traceback as _tb
    _log("/probe/lsmas_realize ENTER")
    try:
        import vapoursynth as vs
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.lsmas.LWLibavSource(source=url)
        # Force realization of frame 0
        frame = src.get_frame(0)
        return {
            "ok": True,
            "frame_format": str(frame.format),
            "frame_width": frame.width,
            "frame_height": frame.height,
            "num_planes": frame.format.num_planes,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


@app.get("/probe/resize_rgbs", dependencies=[Depends(require_api_key)])
def probe_resize_rgbs():
    """Add the YUV->RGBS resize on top of lsmas. Realize frame 0."""
    import traceback as _tb
    _log("/probe/resize_rgbs ENTER")
    try:
        import vapoursynth as vs
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.lsmas.LWLibavSource(source=url)
        src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s='709')
        frame = src.get_frame(0)
        return {
            "ok": True,
            "frame_format": str(frame.format),
            "frame_width": frame.width,
            "frame_height": frame.height,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


@app.get("/probe/rife_eval", dependencies=[Depends(require_api_key)])
def probe_rife_eval():
    """Full pipeline: lsmas + resize + RIFE + reconvert. Realize frame 0.
    This will trigger the TRT engine compile on first call (~5-10 min)."""
    import traceback as _tb
    _log("/probe/rife_eval ENTER")
    try:
        import vapoursynth as vs
        from vsmlrt import RIFE, Backend
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.lsmas.LWLibavSource(source=url)
        src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s='709')
        out = RIFE(src, multi=2, model=46, backend=Backend.TRT(fp16=True, num_streams=2))
        out = core.resize.Bilinear(out, format=vs.YUV420P8, matrix_s='709')
        frame = out.get_frame(0)
        return {
            "ok": True,
            "frame_format": str(frame.format),
            "frame_width": frame.width,
            "frame_height": frame.height,
            "out_num_frames": out.num_frames,
            "out_fps_num": out.fps_num,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


@app.get("/probe/rife_eval_padded", dependencies=[Depends(require_api_key)])
def probe_rife_eval_padded():
    """Same as rife_eval but explicitly pads the source so dimensions are
    multiples of 32 (RIFE's requirement)."""
    import traceback as _tb
    _log("/probe/rife_eval_padded ENTER")
    try:
        import vapoursynth as vs
        from vsmlrt import RIFE, Backend
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.lsmas.LWLibavSource(source=url)
        src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s='709')

        # Pad to multiple of 32 if needed
        pad_w = (32 - src.width % 32) % 32
        pad_h = (32 - src.height % 32) % 32
        if pad_w or pad_h:
            src = core.std.AddBorders(src, right=pad_w, bottom=pad_h)

        out = RIFE(src, multi=2, model=46, backend=Backend.TRT(fp16=True, num_streams=2))

        # Crop back if we padded
        if pad_w or pad_h:
            out = core.std.Crop(out, right=pad_w, bottom=pad_h)

        out = core.resize.Bilinear(out, format=vs.YUV420P8, matrix_s='709')
        frame = out.get_frame(0)
        return {
            "ok": True,
            "padded_w": pad_w,
            "padded_h": pad_h,
            "frame_format": str(frame.format),
            "frame_width": frame.width,
            "frame_height": frame.height,
            "out_num_frames": out.num_frames,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


# -----------------------------------------------------------------------------
# v0.1.10 BESTSOURCE PROBES — added 2026-05-27. v0.1.9 probes proved lsmas
# (LWLibavSource) fails on HTTP URLs ("failed to construct index"). bs
# (BestSource / VideoSource) supports HTTP/HLS natively. These probes verify
# bs works end-to-end on the production base image before we hit /process.
# -----------------------------------------------------------------------------

@app.get("/probe/bs_real", dependencies=[Depends(require_api_key)])
def probe_bs_real():
    """Try BestSource on the real test URL (no realization)."""
    import traceback as _tb
    try:
        import vapoursynth as vs
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.bs.VideoSource(source=url, cachemode=0)
        return {
            "ok": True,
            "num_frames": src.num_frames,
            "fps_num": src.fps_num,
            "fps_den": src.fps_den,
            "width": src.width,
            "height": src.height,
            "format_name": src.format.name if src.format else None,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


@app.get("/probe/bs_realize", dependencies=[Depends(require_api_key)])
def probe_bs_realize():
    """Try BestSource + decode frame 0."""
    import traceback as _tb
    try:
        import vapoursynth as vs
        core = vs.core
        url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        src = core.bs.VideoSource(source=url, cachemode=0)
        frame = src.get_frame(0)
        return {
            "ok": True,
            "frame_format": str(frame.format),
            "frame_width": frame.width,
            "frame_height": frame.height,
            "num_planes": frame.format.num_planes,
        }
    except BaseException as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


# -----------------------------------------------------------------------------
# v0.1.11 PARAMETERIZED SOURCE PROBE — added 2026-05-28. The hardcoded /probe/bs_*
# endpoints couldn't open the mux master HLS playlist URL ("Couldn't open"). This
# generic probe lets us URL-discovery-sweep different backends + URL shapes
# (direct .ts / .mp4 / media playlist / master playlist) without rebuilding the
# image. Returns the same structured JSON as the existing probes.
# -----------------------------------------------------------------------------

@app.get("/probe/source", dependencies=[Depends(require_api_key)])
def probe_source(backend: str = "bs", realize: bool = False, url: str = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"):
    """Parameterized source probe — try different backends and URLs without rebuilding."""
    import traceback as _tb
    _log("/probe/source ENTER", backend=backend, realize=realize, url=url[:200])
    try:
        import vapoursynth as vs
        core = vs.core
        if backend == "bs":
            src = core.bs.VideoSource(source=url, cachemode=0)
        elif backend == "lsmas":
            src = core.lsmas.LWLibavSource(source=url)
        else:
            return {"ok": False, "error": f"unknown backend: {backend}"}

        result = {
            "ok": True,
            "backend": backend,
            "url": url,
            "num_frames": src.num_frames,
            "fps_num": src.fps_num,
            "fps_den": src.fps_den,
            "width": src.width,
            "height": src.height,
            "format_name": src.format.name if src.format else None,
        }
        if realize:
            frame = src.get_frame(0)
            result["realized_frame_format"] = str(frame.format)
            result["realized_frame_width"] = frame.width
            result["realized_frame_height"] = frame.height
        return result
    except BaseException as e:
        return {
            "ok": False,
            "backend": backend,
            "url": url,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1000],
            "traceback": _tb.format_exc()[-1500:],
        }


# -----------------------------------------------------------------------------
# v0.1.12 NETWORK + FFMPEG-DIRECT PROBES — added 2026-05-28. v0.1.11 /probe/source
# showed bs.VideoSource fails identically ("VideoSource: Couldn't open ...") on
# every HTTP/HTTPS URL we threw at it (mux HLS, mux .ts, Google CDN BBB.mp4).
# Need to disambiguate: is the pod offline? Does ffmpeg itself have HTTPS? Does
# Python's stdlib reach the URL? These three probes test each layer independently.
# -----------------------------------------------------------------------------

@app.get("/probe/net", dependencies=[Depends(require_api_key)])
def probe_net(url: str = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"):
    """Pure-Python network reachability check via urllib + DNS. No video libs."""
    import traceback as _tb
    import socket
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen
    _log("/probe/net ENTER", url=url[:200])
    parsed = urlparse(url)
    host = parsed.hostname or ""
    result: dict = {"url": url, "host": host, "scheme": parsed.scheme}
    # DNS
    try:
        ips = socket.gethostbyname_ex(host)
        result["dns_ok"] = True
        result["dns_ips"] = ips[2][:5]
    except Exception as e:
        result["dns_ok"] = False
        result["dns_error"] = f"{type(e).__name__}: {e}"
    # HEAD / first 1KB
    try:
        req = Request(url, headers={"User-Agent": "castbooster-probe/0.1.12", "Range": "bytes=0-1023"})
        with urlopen(req, timeout=15) as resp:
            data = resp.read(1024)
            result["http_ok"] = True
            result["http_status"] = resp.status
            result["http_content_type"] = resp.headers.get("Content-Type")
            result["http_content_length"] = resp.headers.get("Content-Length")
            result["http_first_bytes_hex"] = data[:64].hex()
    except Exception as e:
        result["http_ok"] = False
        result["http_error_type"] = type(e).__name__
        result["http_error"] = str(e)[:500]
        result["http_traceback"] = _tb.format_exc()[-1000:]
    return result


@app.get("/probe/ffprobe", dependencies=[Depends(require_api_key)])
def probe_ffprobe(url: str = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"):
    """Run the bundled ffmpeg (as ffprobe) directly against the URL.
    If this works but /probe/source doesn't, the bug is in bs/lsmas. If this
    also fails, the bundled ffmpeg lacks TLS or there's a network/cert issue.
    """
    import subprocess
    _log("/probe/ffprobe ENTER", url=url[:200])
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_streams",
                "-show_format",
                "-of", "json",
                "-timeout", "10000000",  # microseconds -> 10s
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "returncode": r.returncode,
            "stdout_head": r.stdout[:2000],
            "stderr_head": r.stderr[:2000],
        }
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timeout 30s"}
    except Exception as e:
        return {"error_type": type(e).__name__, "error": str(e)[:500]}


@app.get("/probe/ffmpeg_protocols", dependencies=[Depends(require_api_key)])
def probe_ffmpeg_protocols():
    """Dump ffmpeg's compiled-in protocols and TLS provider info.
    Looks for https + (openssl|gnutls|libtls) markers in the build config + protocols list.
    """
    import subprocess
    _log("/probe/ffmpeg_protocols ENTER")
    out: dict = {}
    try:
        r = subprocess.run(["ffmpeg", "-protocols"], capture_output=True, text=True, timeout=10)
        out["protocols_rc"] = r.returncode
        out["protocols"] = r.stdout
        out["protocols_stderr"] = r.stderr[:500]
    except Exception as e:
        out["protocols_error"] = f"{type(e).__name__}: {e}"
    try:
        r2 = subprocess.run(["ffmpeg", "-buildconf"], capture_output=True, text=True, timeout=10)
        out["buildconf_rc"] = r2.returncode
        # buildconf often goes to stderr in old ffmpeg builds
        out["buildconf_stdout"] = r2.stdout[:3000]
        out["buildconf_stderr"] = r2.stderr[:3000]
    except Exception as e:
        out["buildconf_error"] = f"{type(e).__name__}: {e}"
    return out


# -----------------------------------------------------------------------------
# v0.1.13 PARAMETERIZED FULL-RIFE PROBE — added 2026-05-28. v0.1.12 confirmed
# bs.VideoSource works on the mux test URL (38075 frames decoded at frame 0),
# but /process with the same URL still crashes uvicorn (CF 502 in 13.5s). The
# existing /probe/rife_eval uses lsmas which fails immediately on HTTP, so it
# never tested the RIFE step itself.
#
# This endpoint runs the SAME stages as /process (bs source → RGBS → optional
# pad → RIFE → optional crop → YUV420P8) but exposes each stage as a parameter,
# and returns structured JSON. If it succeeds, the bug is in non-pipeline code
# (clip.output(y4m=True) → ffmpeg subprocess pipe). If it crashes uvicorn,
# the bug is in RIFE/TRT itself.
# -----------------------------------------------------------------------------

@app.get("/probe/rife_eval_url", dependencies=[Depends(require_api_key)])
def probe_rife_eval_url(
    url: str = "http://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
    pad_to_32: bool = True,
    realize: bool = True,
    use_trt: bool = True,
):
    """Run the full RIFE pipeline with a parameterized source URL.

    Tests the same stages as /process but as a separate endpoint to bypass
    POST-route-specific issues. Returns structured JSON; uses BestSource for
    HTTP support; optionally pads to 32 and optionally realizes frame 0.
    """
    import traceback as _tb
    _log(
        "/probe/rife_eval_url ENTER",
        url=url[:200],
        pad_to_32=pad_to_32,
        realize=realize,
        use_trt=use_trt,
    )
    try:
        import vapoursynth as vs
        from vsmlrt import RIFE, Backend
        core = vs.core

        # Stage 1: source via bs (HTTP-capable)
        src = core.bs.VideoSource(source=url, cachemode=0)
        original_w, original_h = src.width, src.height

        # Stage 2: convert to RGBS for RIFE
        src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s='709')

        # Stage 3: optional pad to multiple of 32
        pad_w, pad_h = 0, 0
        if pad_to_32:
            pad_w = (32 - src.width % 32) % 32
            pad_h = (32 - src.height % 32) % 32
            if pad_w or pad_h:
                src = core.std.AddBorders(src, right=pad_w, bottom=pad_h)

        # Stage 4: RIFE 2x
        backend = (
            Backend.TRT(fp16=True, num_streams=2)
            if use_trt
            else Backend.ORT_CUDA(fp16=True)
        )
        out = RIFE(src, multi=2, model=46, backend=backend)

        # Stage 5: optional crop back
        if pad_w or pad_h:
            out = core.std.Crop(out, right=pad_w, bottom=pad_h)

        # Stage 6: convert back to YUV420P8
        out = core.resize.Bilinear(out, format=vs.YUV420P8, matrix_s='709')

        result = {
            "ok": True,
            "url": url,
            "pad_to_32": pad_to_32,
            "use_trt": use_trt,
            "original_width": original_w,
            "original_height": original_h,
            "padded_width_added": pad_w,
            "padded_height_added": pad_h,
            "out_num_frames": out.num_frames,
            "out_fps_num": out.fps_num,
            "out_fps_den": out.fps_den,
            "out_width": out.width,
            "out_height": out.height,
        }

        if realize:
            # THIS is where the TRT engine compiles (~5-10 min first call)
            frame = out.get_frame(0)
            result["realized_frame_format"] = str(frame.format)
            result["realized_frame_width"] = frame.width
            result["realized_frame_height"] = frame.height

        return result
    except BaseException as e:
        return {
            "ok": False,
            "url": url,
            "pad_to_32": pad_to_32,
            "use_trt": use_trt,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:1500],
            "traceback": _tb.format_exc()[-2000:],
        }


# -----------------------------------------------------------------------------
# v0.2.5 TRTEXEC DIRECT PROBE — added 2026-05-28. v0.2.4 pod test showed
# /probe/rife_eval_url?realize=true crashes with:
#     RuntimeError: trtexec execution fails but no log is found
# raised by vsmlrt.py:2200 — meaning vsmlrt invoked trtexec and got a non-zero
# exit, and trtexec didn't write to $TRTEXEC_LOG_FILE. The most likely root
# cause: trtexec can't resolve its shared libs (libnvinfer / libcudnn /
# libcudart) because vsmlrt strips LD_LIBRARY_PATH when subprocessing it
# (vsmlrt.py:2192 passes a minimal env={TRTEXEC_LOG_FILE, CUDA_MODULE_LOADING}).
#
# v0.2.5 fixes this two ways:
#   (1) /usr/local/bin/trtexec is now a bash wrapper that sets LD_LIBRARY_PATH
#       explicitly before exec'ing /usr/local/bin/trtexec.real
#   (2) ENV LD_LIBRARY_PATH set in Dockerfile for any non-vsmlrt caller
#
# This endpoint runs trtexec directly (with full os.environ) AND runs ldd to
# show its dynamic-link state. It catches both:
#   - "the wrapper isn't on PATH" / symlink broken
#   - "trtexec exists but ldd shows missing libs"
# Either tells us whether the v0.2.5 fix worked or pinpoints the real issue.
# -----------------------------------------------------------------------------

def _ldd(binary_path: str):
    """Run ldd on a binary to see its library dependencies."""
    import os
    import subprocess
    if not os.path.exists(binary_path):
        return {"error": "binary not found"}
    try:
        r = subprocess.run(
            ["ldd", binary_path], capture_output=True, text=True, timeout=10
        )
        return {"returncode": r.returncode, "output": r.stdout[:3000]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/probe/trtexec", dependencies=[Depends(require_api_key)])
def probe_trtexec():
    """Run trtexec --help directly to see what error it produces.

    Helps debug the 'trtexec execution fails but no log is found' bug from
    vsmlrt.py:2200. Probes both the wrapper path and the real path."""
    import os
    import subprocess

    paths_to_try = [
        "/usr/local/lib/vapoursynth/vsmlrt-cuda/trtexec",  # vsmlrt's expected path (symlink)
        "/usr/local/bin/trtexec",        # our wrapper (or the real binary pre-v0.2.5)
        "/usr/local/bin/trtexec.real",   # v0.2.5+ real binary behind the wrapper
    ]

    results = []
    for p in paths_to_try:
        info: dict = {
            "path": p,
            "exists": os.path.exists(p),
            "is_link": os.path.islink(p) if os.path.exists(p) else None,
        }
        if info["is_link"]:
            try:
                info["link_target"] = os.readlink(p)
            except Exception as e:
                info["link_target_error"] = f"{type(e).__name__}: {e}"
        if info["exists"]:
            # Try to run it (--help is fast and doesn't need GPU)
            try:
                r = subprocess.run(
                    [p, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env={**os.environ},  # explicit env (inherits LD_LIBRARY_PATH)
                )
                info["returncode"] = r.returncode
                info["stdout_head"] = r.stdout[:1500]
                info["stderr_head"] = r.stderr[:1500]
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
        results.append(info)

    return {
        "results": results,
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "PATH": os.environ.get("PATH", ""),
        "ldd_trtexec_real": _ldd("/usr/local/bin/trtexec.real"),
        "ldd_trtexec": _ldd("/usr/local/bin/trtexec"),
    }


# -----------------------------------------------------------------------------
# v0.2.7 PROCESS-SHORT PROBE — added 2026-05-28. v0.2.6 /process call gets a
# CF 504 at 60s, but no HLS segments ever appear at /hls/playlist.m3u8 even
# after 15+ min. We can't view pod stdout/stderr (RunPod doesn't expose logs
# via API), so we need a probe that:
#   (a) runs the same pipeline as /process but trimmed to a few seconds of frames
#   (b) captures ffmpeg stdout/stderr in the JSON response
#   (c) lists the HLS output dir contents so we know what got written
# /process_short builds the same VS graph as run_rife._build_clip(), trims to
# N frames (default 60 = 0.5s of output), pipes to ffmpeg with the same
# command, and returns everything inline.
# -----------------------------------------------------------------------------

@app.get("/probe/process_short", dependencies=[Depends(require_api_key)])
def probe_process_short(
    url: str = "http://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
    n_frames: int = 60,
    use_trt: bool = True,
):
    """Run a short version of /process and return ffmpeg's stderr + output dir
    contents. Used to debug why /process never writes segments."""
    import shutil
    import subprocess
    import time
    import traceback as _tb
    from pathlib import Path

    _log("/probe/process_short ENTER", url=url[:200], n_frames=n_frames, use_trt=use_trt)
    out_dir = Path("/var/hls_probe")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    playlist = out_dir / "playlist.m3u8"
    segment_pattern = out_dir / "segment_%03d.ts"

    result: dict = {
        "url": url,
        "n_frames": n_frames,
        "use_trt": use_trt,
        "out_dir": str(out_dir),
    }

    # Stage 1: build the clip graph, same as run_rife but trimmed to n_frames
    t0 = time.monotonic()
    try:
        import vapoursynth as vs
        from vsmlrt import RIFE, Backend
        core = vs.core
        src = core.bs.VideoSource(source=url, cachemode=0)
        src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s="709")
        pad_w = (32 - src.width % 32) % 32
        pad_h = (32 - src.height % 32) % 32
        if pad_w or pad_h:
            src = core.std.AddBorders(src, right=pad_w, bottom=pad_h)
        backend = Backend.TRT(fp16=True, num_streams=2) if use_trt else Backend.ORT_CUDA(fp16=True)
        clip = RIFE(src, multi=2, model=46, backend=backend)
        if pad_w or pad_h:
            clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)
        clip = core.resize.Bilinear(clip, format=vs.YUV420P8, matrix_s="709")
        # Trim to first N frames of the RIFE OUTPUT (so we encode quickly)
        clip = core.std.Trim(clip, first=0, last=n_frames - 1)
        result["clip_built_s"] = round(time.monotonic() - t0, 2)
        result["clip_num_frames"] = clip.num_frames
        result["clip_fps_num"] = clip.fps_num
        result["clip_fps_den"] = clip.fps_den
        result["clip_w"] = clip.width
        result["clip_h"] = clip.height
    except BaseException as e:
        result["ok"] = False
        result["stage"] = "build_clip"
        result["error_type"] = type(e).__name__
        result["error_msg"] = str(e)[:1000]
        result["traceback"] = _tb.format_exc()[-2000:]
        return result

    # Stage 2: spawn ffmpeg, pipe clip.output() into stdin
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "yuv4mpegpipe",
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-tune", "ll",
        "-b:v", "8M",
        "-maxrate", "12M",
        "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "0",  # keep all segments for inspection
        "-hls_flags", "independent_segments",
        "-hls_segment_filename", str(segment_pattern),
        str(playlist),
    ]
    result["ffmpeg_cmd"] = " ".join(ffmpeg_cmd)

    t1 = time.monotonic()
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        clip.output(ffmpeg_proc.stdin, y4m=True)
        ffmpeg_proc.stdin.close()
        result["clip_output_done_s"] = round(time.monotonic() - t1, 2)
        # v0.2.8: use manual drain + wait instead of communicate() because
        # ffmpeg_proc.communicate() raises "ValueError: flush of closed file"
        # when stdin has been closed manually (confirmed v0.2.7 probe).
        stdout_b = ffmpeg_proc.stdout.read() if ffmpeg_proc.stdout else b""
        stderr_b = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr else b""
        ffmpeg_proc.wait(timeout=120)
        result["ffmpeg_returncode"] = ffmpeg_proc.returncode
        result["ffmpeg_stdout_tail"] = stdout_b.decode(errors="replace")[-2000:]
        result["ffmpeg_stderr_tail"] = stderr_b.decode(errors="replace")[-3000:]
        result["total_pipeline_s"] = round(time.monotonic() - t0, 2)
    except subprocess.TimeoutExpired:
        ffmpeg_proc.kill()
        try:
            stdout_b, stderr_b = ffmpeg_proc.communicate(timeout=5)
            result["ffmpeg_stdout_tail"] = stdout_b.decode(errors="replace")[-2000:]
            result["ffmpeg_stderr_tail"] = stderr_b.decode(errors="replace")[-3000:]
        except Exception:
            pass
        result["ok"] = False
        result["stage"] = "ffmpeg_timeout"
        result["error_msg"] = "ffmpeg.communicate() timed out at 120s"
    except BaseException as e:
        try:
            ffmpeg_proc.kill()
            stdout_b, stderr_b = ffmpeg_proc.communicate(timeout=5)
            result["ffmpeg_stdout_tail"] = stdout_b.decode(errors="replace")[-2000:]
            result["ffmpeg_stderr_tail"] = stderr_b.decode(errors="replace")[-3000:]
        except Exception:
            pass
        result["ok"] = False
        result["stage"] = "clip_output_or_ffmpeg"
        result["error_type"] = type(e).__name__
        result["error_msg"] = str(e)[:1000]
        result["traceback"] = _tb.format_exc()[-2000:]

    # Stage 3: list the output dir to see what got written
    files = []
    try:
        for p in sorted(out_dir.iterdir()):
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
            })
    except Exception as e:
        files = [{"error": f"{type(e).__name__}: {e}"}]
    result["hls_files"] = files
    if "ok" not in result:
        # Determine success: ffmpeg rc==0 AND playlist exists AND at least 1 segment
        playlist_exists = any(f.get("name") == "playlist.m3u8" for f in files)
        segs = [f for f in files if str(f.get("name", "")).startswith("segment_") and str(f.get("name", "")).endswith(".ts") and f.get("size", 0) > 0]
        result["ok"] = (
            result.get("ffmpeg_returncode") == 0
            and playlist_exists
            and len(segs) > 0
        )
        result["segments_written"] = len(segs)

    return result


@app.on_event("startup")
def _on_startup():
    global _watcher
    _log("STARTUP", pid=os.getpid(), python=sys.version.split()[0])

    # Pipeline manager — always created, even in DISABLE_IDLE_WATCHER mode,
    # because /process and /process_status read from it.
    app.state.pipeline_manager = PipelineManager(
        output_dir=_hls_dir(),
        public_base_url=_public_base_url(),
        run_pipeline=run_pipeline_for_request,
    )
    _log("STARTUP pipeline_manager ready", output_dir=str(_hls_dir()))

    # v0.3.1 (Pillar 3.6): SourceRegistry — token-keyed (source_url, headers)
    # map used by /source_proxy to inject Cookie/User-Agent/Referer into
    # BestSource fetches. /process registers an entry per request and wires
    # an on_terminal cleanup hook so the entry frees on COMPLETED / FAILED.
    app.state.source_registry = SourceRegistry()
    _log("STARTUP source_registry ready")

    if os.environ.get("DISABLE_IDLE_WATCHER") == "1":
        _log("STARTUP idle_watcher disabled (DISABLE_IDLE_WATCHER=1)")
        return
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
