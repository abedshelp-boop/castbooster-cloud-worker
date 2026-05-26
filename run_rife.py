# run_rife.py
"""RIFE v4.6 TensorRT FP16 pipeline using vs-mlrt.

This module runs only inside the cloud Docker image (GPU required).
Local dev machines without GPU should use run_passthrough instead.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineResult:
    returncode: int
    stdout: str
    stderr: str


# VapourSynth script template — interpolates 2x via RIFE v4.6
VPY_TEMPLATE = """
import vapoursynth as vs
from vsmlrt import RIFE, Backend

core = vs.core

src = core.lsmas.LWLibavSource(source='{source_path}')
src = core.resize.Bilinear(src, format=vs.RGBS, matrix_in_s='709')

# RIFE 2x temporal upsample (30fps source -> 60fps output)
out = RIFE(
    src,
    multi=2,
    model=46,  # RIFE v4.6
    backend=Backend.TRT(fp16=True, num_streams=2),
)

out = core.resize.Bilinear(out, format=vs.YUV420P8, matrix_s='709')
out.set_output()
"""


def run_rife(
    source_url: str,
    output_dir: Path,
    timeout_s: int = 7200,
    extra_input_headers: dict[str, str] | None = None,
) -> PipelineResult:
    """Run RIFE v4.6 TRT FP16 pipeline: source -> 2x temporal interp -> HLS.

    Args:
        source_url: source HLS URL (https://... or file://...)
        output_dir: directory for HLS playlist + .ts segments
        timeout_s: kill pipeline if it runs longer than this
        extra_input_headers: optional HTTP headers (forwarded via ffmpeg -headers)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    playlist = output_dir / "playlist.m3u8"
    segment_pattern = output_dir / "segment_%03d.ts"

    # Write VPY script
    vpy_path = output_dir / "pipeline.vpy"
    vpy_path.write_text(VPY_TEMPLATE.format(source_path=source_url))

    # vspipe streams raw frames -> ffmpeg encodes with NVENC -> HLS
    vspipe_cmd = ["vspipe", "-c", "y4m", str(vpy_path), "-"]

    ffmpeg_cmd = ["ffmpeg", "-y"]
    if extra_input_headers:
        for k, v in extra_input_headers.items():
            if "\r" in k or "\n" in k or "\r" in v or "\n" in v:
                raise ValueError(
                    f"header {k!r} contains CR/LF (injection attempt)"
                )
        header_lines = "".join(f"{k}: {v}\r\n" for k, v in extra_input_headers.items())
        ffmpeg_cmd += ["-headers", header_lines]

    ffmpeg_cmd += [
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

    # Pipe vspipe -> ffmpeg
    vspipe_proc = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=vspipe_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    vspipe_proc.stdout.close()  # allow vspipe to receive SIGPIPE if ffmpeg dies

    try:
        ffmpeg_stdout, ffmpeg_stderr = ffmpeg_proc.communicate(timeout=timeout_s)
        vspipe_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        ffmpeg_proc.kill()
        vspipe_proc.kill()
        return PipelineResult(
            returncode=124,
            stdout="",
            stderr=f"pipeline timeout after {timeout_s}s",
        )

    if ffmpeg_proc.returncode != 0:
        return PipelineResult(
            returncode=ffmpeg_proc.returncode,
            stdout=ffmpeg_stdout.decode(errors="replace"),
            stderr=ffmpeg_stderr.decode(errors="replace"),
        )
    if vspipe_proc.returncode != 0:
        vspipe_stderr = vspipe_proc.stderr.read().decode(errors="replace")
        return PipelineResult(
            returncode=vspipe_proc.returncode,
            stdout="",
            stderr=f"vspipe failed: {vspipe_stderr}",
        )

    return PipelineResult(
        returncode=0,
        stdout=ffmpeg_stdout.decode(errors="replace"),
        stderr=ffmpeg_stderr.decode(errors="replace"),
    )
