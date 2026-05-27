# run_rife.py
"""RIFE v4.6 TensorRT FP16 pipeline using vs-mlrt's Python API directly.

Bypasses the bundled vspipe wrapper (broken in our base image) by building the
VapourSynth clip graph in-process and using clip.output() to stream y4m frames
to ffmpeg's stdin via subprocess.

This module runs only inside the cloud Docker image (GPU + vsmlrt required).
"""
import os
import subprocess
import sys
import traceback
from pathlib import Path

from pipeline_types import PipelineResult


def _build_clip(source_url: str):
    """Construct the VapourSynth clip graph. Lazy import so the module is
    importable on dev machines without VapourSynth installed (tests use mocks).
    """
    import vapoursynth as vs
    from vsmlrt import RIFE, Backend

    core = vs.core

    # Source: HTTP(S) HLS URL via BestSource (bs.VideoSource).
    # BestSource supports HTTP URLs / HLS playlists natively (unlike lsmas which
    # requires a local file + .lwi sidecar). cachemode=0 = no .ffi cache file
    # (we don't need seek; we stream once).
    src = core.bs.VideoSource(source=source_url, cachemode=0)
    src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s='709')

    # RIFE 2x temporal upsample (30fps source -> 60fps output)
    out = RIFE(
        src,
        multi=2,
        model=46,  # RIFE v4.6
        backend=Backend.TRT(fp16=True, num_streams=2),
    )

    # Convert back to YUV420P8 for NVENC
    out = core.resize.Bilinear(out, format=vs.YUV420P8, matrix_s='709')
    return out


def run_rife(
    source_url: str,
    output_dir: Path,
    timeout_s: int = 7200,
    extra_input_headers: dict[str, str] | None = None,
) -> PipelineResult:
    """Run RIFE v4.6 TRT FP16 pipeline using Python-direct VS API.

    Args:
        source_url: source HLS URL (http://... or https://...)
        output_dir: directory for HLS playlist + .ts segments
        timeout_s: kill pipeline if it runs longer than this
        extra_input_headers: NOT supported in this pipeline yet (see below)

    Returns:
        PipelineResult with returncode, stdout, stderr
    """
    print(f"[run_rife] START source_url={source_url[:200]} timeout_s={timeout_s}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    playlist = output_dir / "playlist.m3u8"
    segment_pattern = output_dir / "segment_%03d.ts"

    # CRLF injection guard kept for defense-in-depth even though headers are
    # rejected below — if Phase 2 wires headers through lsmas format_opts, the
    # guard already exists and catches malformed input upstream.
    if extra_input_headers:
        for k, v in extra_input_headers.items():
            if "\r" in k or "\n" in k or "\r" in v or "\n" in v:
                print(f"[run_rife] CRLF guard tripped on header {k!r}", flush=True)
                raise ValueError(f"header {k!r} contains CR/LF (injection attempt)")
        print(f"[run_rife] extra_input_headers passed but not yet supported", flush=True)
        raise NotImplementedError(
            "extra_input_headers not supported in RIFE pipeline yet — "
            "vapoursynth/lsmas reads the source, not ffmpeg. "
            "Phase 2: thread headers via lsmas format_opts."
        )

    # Build the clip graph
    print(f"[run_rife] Building VS clip graph...", flush=True)
    try:
        clip = _build_clip(source_url)
        print(f"[run_rife] clip built: {clip.num_frames} frames @ {clip.fps_num}/{clip.fps_den} fps, {clip.width}x{clip.height}", flush=True)
    except Exception as e:
        msg = f"clip build failed: {type(e).__name__}: {e}"
        print(f"[run_rife] {msg}", flush=True)
        return PipelineResult(
            returncode=2,
            stdout="",
            stderr=msg + "\n" + traceback.format_exc(),
        )

    # Build the ffmpeg command — reads y4m from stdin, encodes via NVENC, HLS-muxes
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
        "-hls_time", "4",
        "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments",
        "-hls_segment_filename", str(segment_pattern),
        str(playlist),
    ]
    print(f"[run_rife] Spawning ffmpeg: {' '.join(ffmpeg_cmd[:6])} ...", flush=True)

    # Start ffmpeg, pipe clip.output() into its stdin
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # clip.output(file, y4m=True) writes the y4m stream into the given file
        # object. We pass ffmpeg's stdin pipe. This is blocking for the duration
        # of the encoding.
        print(f"[run_rife] Streaming frames into ffmpeg...", flush=True)
        clip.output(ffmpeg_proc.stdin, y4m=True)
        ffmpeg_proc.stdin.close()
        print(f"[run_rife] clip.output() returned; waiting for ffmpeg...", flush=True)
        stdout_bytes, stderr_bytes = ffmpeg_proc.communicate(timeout=timeout_s)
        rc = ffmpeg_proc.returncode
        print(f"[run_rife] ffmpeg returncode={rc}", flush=True)
        return PipelineResult(
            returncode=rc,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
        )
    except subprocess.TimeoutExpired:
        print(f"[run_rife] timeout after {timeout_s}s; killing ffmpeg", flush=True)
        ffmpeg_proc.kill()
        try:
            ffmpeg_proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return PipelineResult(
            returncode=124,
            stdout="",
            stderr=f"pipeline timeout after {timeout_s}s",
        )
    except Exception as e:
        # clip.output() raises if VS error during processing (e.g. RIFE crash,
        # OOM, source fetch fail). Always kill ffmpeg, surface the error.
        print(f"[run_rife] EXCEPTION in clip.output() / communicate: {type(e).__name__}: {e}", flush=True)
        try:
            ffmpeg_proc.kill()
            ffmpeg_proc.communicate(timeout=5)
        except Exception:
            pass
        return PipelineResult(
            returncode=3,
            stdout="",
            stderr=f"pipeline error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
