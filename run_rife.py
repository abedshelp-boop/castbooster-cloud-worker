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

    # v0.2.6: pad source dims up to a multiple of 32. RIFE's TRT engine requires
    # input height/width divisible by 32; without this, vsmlrt raises:
    #   ValueError: vsmlrt.RIFEMerge: tile size must be divisible by 32 (W, H)
    # at clip-build time (caught upstream, but the partial-graph teardown then
    # segfaults VS → uvicorn restart loop fires → /process returns CF 502).
    # The /probe/rife_eval_url endpoint already does this; pulling the same
    # logic in here so /process works end-to-end. We pad on the right/bottom
    # so the visible frame stays in the top-left, then crop back after RIFE.
    pad_w = (32 - src.width % 32) % 32
    pad_h = (32 - src.height % 32) % 32
    if pad_w or pad_h:
        src = core.std.AddBorders(src, right=pad_w, bottom=pad_h)

    # RIFE 2x temporal upsample (30fps source -> 60fps output)
    out = RIFE(
        src,
        multi=2,
        model=46,  # RIFE v4.6
        backend=Backend.TRT(fp16=True, num_streams=2),
    )

    # v0.2.6: crop the RIFE output back to original dimensions so the final
    # HLS stream matches the source aspect (NVENC encodes whatever we hand it,
    # so a 1280x736 padded frame would leak black borders downstream).
    if pad_w or pad_h:
        out = core.std.Crop(out, right=pad_w, bottom=pad_h)

    # Convert back to YUV420P8 for NVENC
    out = core.resize.Bilinear(out, format=vs.YUV420P8, matrix_s='709')
    return out


def _build_ffmpeg_cmd(playlist: Path, segment_pattern: Path) -> list[str]:
    """Build the ffmpeg argv for NVENC h264 + HLS muxer.

    Extracted from run_rife() so the keyframe / GOP / bitrate args are
    unit-testable without spawning a real ffmpeg subprocess.
    """
    return [
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
        # Issue 2: force an IDR keyframe every 4 seconds so the HLS muxer
        # can cut clean 4-second segments. Time-based expression so it
        # works for any output framerate (RIFE doubles source, so output
        # is variable: 60fps source -> 120fps; 30fps source -> 60fps).
        "-force_key_frames", "expr:gte(t,n_forced*4)",
        "-f", "hls",
        "-hls_time", "4",
        "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments",
        "-hls_segment_filename", str(segment_pattern),
        str(playlist),
    ]


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
    ffmpeg_cmd = _build_ffmpeg_cmd(playlist, segment_pattern)
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
        # v0.2.8: ffmpeg_proc.communicate() raises "ValueError: flush of closed
        # file" because we already closed stdin above. communicate() internally
        # calls self.stdin.flush() and chokes on an already-closed pipe. Use
        # manual drain + wait instead. Confirmed by /probe/process_short in
        # v0.2.7 — error: subprocess.py:2067 self.stdin.flush() ValueError.
        stdout_bytes = ffmpeg_proc.stdout.read() if ffmpeg_proc.stdout else b""
        stderr_bytes = ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr else b""
        ffmpeg_proc.wait(timeout=timeout_s)
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
            ffmpeg_proc.wait(timeout=5)
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
        print(f"[run_rife] EXCEPTION in clip.output() / wait: {type(e).__name__}: {e}", flush=True)
        try:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait(timeout=5)
        except Exception:
            pass
        return PipelineResult(
            returncode=3,
            stdout="",
            stderr=f"pipeline error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
