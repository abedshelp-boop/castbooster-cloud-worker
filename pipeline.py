import subprocess
from pathlib import Path

from pipeline_types import PipelineResult


def run_passthrough(
    source_url: str,
    output_dir: Path,
    timeout_s: int = 600,
    extra_input_headers: dict[str, str] | None = None,
) -> PipelineResult:
    """Run ffmpeg in passthrough mode: source HLS in, HLS out (no RIFE).

    Used as a smoke test before RIFE integration. NVDEC/NVENC NOT enabled here
    (works on any CPU); RIFE-enabled run_rife() in a later task uses GPU.

    Args:
        source_url: file:// or http(s):// URL to source content
        output_dir: directory to write playlist.m3u8 + *.ts segments
        timeout_s: kill ffmpeg if it runs longer than this
        extra_input_headers: optional HTTP headers (forwarded via -headers)

    Returns:
        PipelineResult with returncode, stdout, stderr
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    playlist = output_dir / "playlist.m3u8"
    segment_pattern = output_dir / "segment_%03d.ts"

    cmd = ["ffmpeg", "-y"]

    if extra_input_headers:
        for k, v in extra_input_headers.items():
            if "\r" in k or "\n" in k or "\r" in v or "\n" in v:
                raise ValueError(
                    f"header {k!r} contains CR/LF (injection attempt)"
                )
        header_lines = "".join(f"{k}: {v}\r\n" for k, v in extra_input_headers.items())
        cmd += ["-headers", header_lines]

    cmd += [
        "-i", source_url,
        "-c:v", "copy",
        "-c:a", "copy",
        "-f", "hls",
        "-hls_time", "4",
        "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments",
        "-hls_segment_filename", str(segment_pattern),
        str(playlist),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return PipelineResult(
            returncode=-1,
            stdout=(e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
            stderr=f"ffmpeg passthrough timeout after {timeout_s}s",
        )
    return PipelineResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
