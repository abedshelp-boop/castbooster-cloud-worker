# tests/test_run_rife.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_fake_clip(monkeypatch, num_frames=2):
    """Mock _build_clip to return a fake clip object whose .output() succeeds."""
    fake_clip = MagicMock()
    fake_clip.num_frames = num_frames
    fake_clip.fps_num = 60
    fake_clip.fps_den = 1
    fake_clip.width = 1920
    fake_clip.height = 1080
    # clip.output(file, y4m=True) is the call; we make it a no-op
    fake_clip.output = MagicMock(return_value=None)
    monkeypatch.setattr("run_rife._build_clip", lambda source_url: fake_clip)
    return fake_clip


def test_run_rife_calls_ffmpeg_with_nvenc(tmp_path, monkeypatch):
    from run_rife import run_rife

    fake_clip = _make_fake_clip(monkeypatch)

    fake_ffmpeg = MagicMock()
    fake_ffmpeg.returncode = 0
    # v0.2.8: run_rife now uses stdout.read() / stderr.read() / wait()
    # instead of communicate() — see comment in run_rife.py about
    # "ValueError: flush of closed file" when stdin is closed before
    # communicate is called.
    fake_ffmpeg.stdout.read.return_value = b""
    fake_ffmpeg.stderr.read.return_value = b""
    fake_ffmpeg.wait.return_value = 0
    fake_ffmpeg.stdin = MagicMock()

    popen_calls = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return fake_ffmpeg

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = run_rife(
        source_url="https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        output_dir=tmp_path,
        timeout_s=60,
    )

    assert result.returncode == 0
    assert len(popen_calls) == 1
    ffmpeg_cmd = popen_calls[0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert "h264_nvenc" in ffmpeg_cmd
    assert "-f" in ffmpeg_cmd and "hls" in ffmpeg_cmd
    # Verify clip.output() was called with y4m=True
    fake_clip.output.assert_called_once()
    assert fake_clip.output.call_args.kwargs.get("y4m") is True


def test_run_rife_returns_failure_on_ffmpeg_nonzero(tmp_path, monkeypatch):
    from run_rife import run_rife

    _make_fake_clip(monkeypatch)

    fake_ffmpeg = MagicMock()
    fake_ffmpeg.returncode = 1
    # v0.2.8: same API change as the success-case test above
    fake_ffmpeg.stdout.read.return_value = b""
    fake_ffmpeg.stderr.read.return_value = b"nvenc init failed"
    fake_ffmpeg.wait.return_value = 1
    fake_ffmpeg.stdin = MagicMock()

    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: fake_ffmpeg)

    result = run_rife(
        source_url="x",
        output_dir=tmp_path,
        timeout_s=60,
    )
    assert result.returncode == 1
    assert "nvenc init failed" in result.stderr


def test_run_rife_rejects_extra_input_headers(tmp_path, monkeypatch):
    """Non-empty headers should raise NotImplementedError (Phase 2)."""
    from run_rife import run_rife

    with pytest.raises(NotImplementedError, match="extra_input_headers"):
        run_rife(
            source_url="https://example.com/x.m3u8",
            output_dir=tmp_path,
            extra_input_headers={"Cookie": "session=abc"},
        )


def test_run_rife_crlf_guard_fires_before_notimplemented(tmp_path):
    """CRLF in headers should raise ValueError (caught earlier than NIE)."""
    from run_rife import run_rife

    with pytest.raises(ValueError, match="CR/LF"):
        run_rife(
            source_url="https://example.com/x.m3u8",
            output_dir=tmp_path,
            extra_input_headers={"X-Bad": "value\r\nX-Smuggled: pwned"},
        )


def test_run_rife_clip_build_failure_returns_2(tmp_path, monkeypatch):
    """If _build_clip raises (e.g., vsmlrt import error on dev machine), surface
    as returncode 2 with a useful traceback."""
    from run_rife import run_rife

    def fake_build_clip(source_url):
        raise RuntimeError("vsmlrt: cannot load any filters")

    monkeypatch.setattr("run_rife._build_clip", fake_build_clip)

    result = run_rife(
        source_url="https://example.com/x.m3u8",
        output_dir=tmp_path,
    )
    assert result.returncode == 2
    assert "clip build failed" in result.stderr
    assert "RuntimeError" in result.stderr


def test_build_ffmpeg_cmd_includes_hls_settings():
    from run_rife import _build_ffmpeg_cmd
    cmd = _build_ffmpeg_cmd(
        playlist=Path("/var/hls/playlist.m3u8"),
        segment_pattern=Path("/var/hls/segment_%03d.ts"),
    )
    assert cmd[0] == "ffmpeg"
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert "-hls_time" in cmd and cmd[cmd.index("-hls_time") + 1] == "4"
    assert "-hls_segment_filename" in cmd
    assert "-f" in cmd
    # Trailing arg must be the playlist path (compare as Path to handle OS separators).
    assert Path(cmd[-1]) == Path("/var/hls/playlist.m3u8")
