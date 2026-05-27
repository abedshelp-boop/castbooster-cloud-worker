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

app = FastAPI(title="castbooster-cloud-worker", version="0.1.12")


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
        # NOTE: 500 stays HTTPException — short detail string, no risk of pipe-breaking.
        # Only the longer 400/502 errors below were swapped to JSONResponse in v0.1.11
        # because their multi-KB tracebacks were tripping FastAPI->uvicorn->nginx->CF
        # and surfacing as bare Cloudflare 502s.
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
        # v0.1.11: JSONResponse instead of HTTPException — see notes above.
        # `detail` key kept for backwards-compat with existing tests/clients;
        # structured fields added per the dispatch spec for richer diagnostics.
        print(f"[/process] 400 invalid request: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"invalid request: {e}",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:1500],
            },
        )
    except Exception as e:
        # v0.1.11: JSONResponse instead of HTTPException — multi-KB detail strings
        # via HTTPException were getting mangled / dropped before reaching CF,
        # producing bare Cloudflare 502s with no FastAPI body.
        tb_tail = "".join(_tb.format_exception(type(e), e, e.__traceback__))[-1500:]
        print(f"[/process] 502 pipeline EXCEPTION: {type(e).__name__}: {e}\n{tb_tail}", file=sys.stderr, flush=True)
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"pipeline error: {type(e).__name__}: {e}",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:1500],
                "traceback_tail": tb_tail,
            },
        )

    if result.returncode != 0:
        # v0.1.11: JSONResponse instead of HTTPException — ffmpeg stderr can be
        # multi-KB and was likely contributing to the same pipe issue.
        print(f"[/process] 502 pipeline nonzero: rc={result.returncode}", file=sys.stderr, flush=True)
        stderr_tail = result.stderr[-1500:] if result.stderr else ""
        return JSONResponse(
            status_code=502,
            content={
                "detail": stderr_tail[-500:] or "ffmpeg failed with no stderr",
                "error_type": "PipelineNonZeroExit",
                "error_msg": f"pipeline returncode={result.returncode}",
                "returncode": result.returncode,
                "stderr_tail": stderr_tail,
            },
        )

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
